"""
cogito server — HTTP server keeping memory warm in process.

Endpoints:
  GET  /health
       → {"status": "ok", "count": N, "version": "..."}

  POST /query   {"text": "...", "limit": 5}
       → {"memories": [{"text": "...", "score": N}]}
       Narrow search, L2 threshold filter only. Fast.

  POST /recall  {"text": "...", "limit": 50, "threshold": 400}
       → {"memories": [...], "method": "filter"|"fallback_*"}
       Broad search + cheap-LLM integer-pointer filter. Smart.

  POST /recall_hybrid  {"text": "...", "limit": 50, "tier": "filter", "top_k": 5}
       → {"memories": [...], "method": "hybrid_*|..."}
       BM25 + dense + RRF + tiered LLM escalation.
       tier is one of: "zero_llm" (default, 83.2% R@1, $0/query) | "filter"
       (benchmark-tuned, experimental) | "flagship" (benchmark-tuned, 96.4%
       R@1 but escalates ~80% — see docs/RELEASE-SCOPE.md).

  POST /store   {"text": "...", "id": "<optional uuid>"}
       → {"id": "...", "text": "..."}
       Write one memory verbatim — no extraction LLM, agent decides content.
       This is the preferred write path. Use /add only if you want mem0
       extraction to summarise raw text for you.

  POST /add     {"text": "..."}
       → {"count": N, "memories": [...]}
       Feeds text through mem0's extraction LLM before storing. Use when
       you have raw/unstructured text and want automatic summarisation.

Start:
  fidelis-server                        # uses .cogito.json or env vars
  fidelis-server --config /path/to.json
  fidelis-server --port 19420
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("cogito.server")

from fidelis import __version__
from fidelis.config import load, mem0_config
from fidelis.degrade import queued_count, replay_queue, safe_add
from fidelis.recall import recall as do_recall
from fidelis.recall_b import recall_b as do_recall_b
from fidelis.recall_hybrid import recall_hybrid as do_recall_hybrid
from fidelis.snapshot import _read_snapshot, _snapshot_path


def _boot(cfg: dict) -> object:
    """Import mem0 from wherever it's installed and return a Memory instance."""
    # Support venv via COGITO_SITE_PACKAGES or system install
    site = os.environ.get("COGITO_SITE_PACKAGES")
    if site and site not in sys.path:
        sys.path.insert(0, site)

    from mem0 import Memory  # type: ignore

    m = Memory.from_config(mem0_config(cfg))
    return m


def make_handler(memory: object, cfg: dict) -> type:
    user_id: str = cfg["user_id"]
    query_threshold: float = cfg.get("query_threshold", 250.0)
    # FIDELIS_DECOMPOSE_TIMEOUT_SECS: max seconds for /recall sub-query pipeline.
    # Default 8s preserves existing behavior in normal cases; kicks in only on slow-call edges.
    _decompose_timeout: float = float(os.environ.get("FIDELIS_DECOMPOSE_TIMEOUT_SECS", 8))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress default logging
            pass

        def _json(self, data, status=200):
            body = json.dumps(data).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("client disconnected during write")
                return

        _MAX_BODY = 1_048_576  # 1 MB

        def _read_body(self) -> dict | None:
            n = int(self.headers.get("Content-Length", 0))
            if n > self._MAX_BODY:
                return None  # signal rejection
            raw = self.rfile.read(n)
            try:
                return json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            try:
                if self.path == "/health":
                    # Read count directly from chroma. The previous fallback
                    # used get_all with top_k=10000 which (a) silently capped
                    # the reported count at 10000 and (b) hammered Ollama for
                    # every health probe. col_info() returns the collection
                    # object whose .count() is O(1).
                    try:
                        count = memory.vector_store.col_info().count()  # type: ignore
                    except Exception as e:
                        # Older mem0 wrapper (<2.0): direct .col attribute.
                        try:
                            count = memory.vector_store.col.count()  # type: ignore
                        except Exception:
                            count = -1  # signal: chroma unhealthy
                            logger.warning("health: chroma count failed: %s", e)
                    snap_path = _snapshot_path(cfg)
                    self._json({
                        "status": "ok" if count >= 0 else "degraded",
                        "count": count,
                        "queued": queued_count(),
                        "version": __version__,
                        "calibrated": bool(cfg.get("vocab_map")),
                        "snapshot": snap_path.exists(),
                    })
                elif self.path == "/snapshot":
                    text = _read_snapshot(cfg)
                    if text is None:
                        self._json({"error": "no snapshot — run `cogito snapshot` first"}, 404)
                    else:
                        self._json({"snapshot": text, "path": str(_snapshot_path(cfg))})
                elif self.path == "/replay":
                    # Manually drain the queue. Useful to call after fixing a
                    # transient Ollama outage. The server also auto-drains on
                    # startup and periodically via the background thread.
                    result = replay_queue(memory, user_id=user_id)  # type: ignore
                    self._json(result)
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                try:
                    self._json({"error": f"internal error: {type(e).__name__}"}, 500)
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug("client disconnected before error response could be sent")

        def do_POST(self):
            try:
                data = self._read_body()
                if data is None:
                    self._json({"error": "request body too large"}, 413)
                    return
                if not data and self.path not in ("/add", "/store"):
                    self._json({"error": "invalid json"}, 400)
                    return

                if self.path == "/query":
                    text = data.get("text", "")
                    limit = int(data.get("limit", 5))
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": []})
                        return
                    # Bypass mem0.Memory.search wrapper: it routes through
                    # score_and_rank which (in mem0 2.0.x) returns broken
                    # score=1.0 for every result regardless of similarity.
                    # Verified empirically — same query, when we go directly to
                    # vector_store.search, returns proper distances (the actual
                    # text-match record scores 0.5878 vs unrelated at 1.07+).
                    qv = memory.embedding_model.embed(text, memory_action="search")  # type: ignore
                    raw = memory.vector_store.search(  # type: ignore
                        query=text, vectors=[qv], top_k=limit,
                        filters={"user_id": user_id},
                    )
                    memories = [
                        {
                            "text": (r.payload or {}).get("data", ""),
                            # chroma distance is 0..2 for cosine; smaller=better.
                            # Convert to similarity (1 = identical, 0 = orthogonal).
                            "score": round(max(0.0, 1.0 - (r.score or 0) / 2), 3),
                        }
                        for r in raw
                        if (r.payload or {}).get("data")
                    ]
                    self._json({"memories": memories})

                elif self.path == "/recall":
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": [], "method": "empty_query"})
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    since = data.get("since")
                    degraded = False
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                            _fut = _pool.submit(
                                do_recall, memory, text,
                                user_id=user_id, cfg=cfg, limit=limit, since=since,
                            )
                            memories, method = _fut.result(timeout=_decompose_timeout)
                    except concurrent.futures.TimeoutError:
                        # Decompose pipeline timed out — fall back to vector-only single query.
                        # Bypass mem0.Memory.search wrapper (broken score_and_rank in 2.0.0
                        # returns score=1.0 for all results); call vector_store.search directly.
                        logger.warning(
                            "[fidelis] /recall decompose timeout (>%ss) for query '%s'; returning vector-only fallback",
                            _decompose_timeout, text[:50],
                        )
                        qv = memory.embedding_model.embed(text, memory_action="search")
                        raw = memory.vector_store.search(
                            query=text, vectors=[qv], top_k=limit,
                            filters={"user_id": user_id},
                        )
                        memories = [
                            {
                                "text": (r.payload or {}).get("data", ""),
                                "score": round(max(0.0, 1.0 - (r.score or 0) / 2), 3),
                            }
                            for r in raw
                            if (r.payload or {}).get("data")
                        ]
                        method = "vector-only-fallback"
                        degraded = True
                    print(f"[cogito] /recall '{text[:50]}' → {len(memories)} results ({method})", flush=True)
                    resp: dict = {"memories": memories, "method": method}
                    if degraded:
                        resp["degraded"] = True
                    self._json(resp)

                elif self.path == "/recall_b":
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": [], "method": "empty_query"})
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    memories, method = do_recall_b(
                        memory, text, user_id=user_id, cfg=cfg,
                        limit=limit,
                    )
                    print(f"[cogito] /recall_b '{text[:50]}' → {len(memories)} results ({method})", flush=True)
                    self._json({"memories": memories, "method": method})

                elif self.path == "/recall_hybrid":
                    # BM25 + dense + RRF + tiered LLM escalation.
                    # Default tier: zero_llm (83.2% R@1 at $0, production moat).
                    # Opt-in filter/flagship for benchmark replication.
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": [], "method": "empty_query"})
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    tier = data.get("tier", "zero_llm")
                    top_k = int(data.get("top_k", 5))
                    if tier not in ("zero_llm", "filter", "flagship"):
                        self._json({"error": f"invalid tier: {tier}"}, 400)
                        return
                    memories, method = do_recall_hybrid(
                        memory, text, user_id=user_id, cfg=cfg,
                        limit=limit, tier=tier, top_k=top_k,
                    )
                    print(f"[cogito] /recall_hybrid '{text[:50]}' tier={tier} → {len(memories)} results ({method})", flush=True)
                    self._json({"memories": memories, "method": method})

                elif self.path == "/store":
                    # Verbatim write — agent decides content, no extraction LLM.
                    # Uses safe_add: queues locally if dependency (Ollama) is down.
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"error": "no text"}, 400)
                        return
                    result = safe_add(memory, text, user_id, kind="store")  # type: ignore
                    self._json({**result, "queued_total": queued_count()})

                elif self.path == "/add":
                    # Uses safe_add: queues locally if Ollama is unreachable.
                    text = data.get("text", "")
                    if not text:
                        self._json({"error": "no text"}, 400)
                        return
                    result = safe_add(memory, text, user_id, kind="add")  # type: ignore
                    if result["status"] == "queued":
                        self._json({
                            "status": "queued",
                            "id": result["id"],
                            "reason": result["reason"],
                            "queued_total": queued_count(),
                        }, 202)  # 202 Accepted: write deferred
                    else:
                        extracted = result.get("extracted", [])
                        self._json({
                            "status": "stored",
                            "count": len(extracted),
                            "memories": extracted,
                        })

                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                try:
                    self._json({"error": f"internal error: {type(e).__name__}"}, 500)
                except (BrokenPipeError, ConnectionResetError):
                    logger.debug("client disconnected before error response could be sent")

    return Handler


def main():
    # Configure root logging once. Without this, `logger.warning(...)` calls
    # are silent in launchd / systemd / MCP contexts because nothing else in
    # the stack calls basicConfig. FIDELIS_LOG_LEVEL overrides per-deployment.
    logging.basicConfig(
        level=os.environ.get("FIDELIS_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=False,  # respect any pre-existing config (tests, parent app)
    )

    parser = argparse.ArgumentParser(description="fidelis memory server")
    parser.add_argument("--config", help="Path to .cogito.json")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    args = parser.parse_args()

    cfg = load(args.config)
    if args.port:
        cfg["port"] = args.port

    src = cfg.get("_config_file", "defaults + env")
    print(f"[cogito] Starting server v{__version__} (config: {src})", flush=True)
    print("[cogito] Loading memory store...", flush=True)

    memory = _boot(cfg)

    # Background replay thread — sweeps the queue every 60s. Items that failed
    # at write time (Ollama momentarily unreachable, embed timeout) get retried
    # without requiring a server restart. The first sweep runs ~5s after server
    # start so HTTP serving is up immediately rather than blocking on a long
    # drain. Items stay in the queue across server restarts.
    import threading
    def _replay_loop():
        import time as _t
        _t.sleep(5)  # let serve_forever() bind first
        # Exponential backoff: base 60s, doubles on no-progress sweeps,
        # capped at 30 min. Resets to base on any successful replay.
        # Prevents the forever-warm-LLM heat bug when the queue is
        # non-empty but every item keeps failing (e.g. Ollama model
        # missing or unreachable). Combined with MAX_ATTEMPTS dead-letter
        # in degrade.py, the queue cannot stay hot indefinitely.
        BASE = 60
        MAX = 1800
        sleep_s = BASE
        while True:
            try:
                pending = queued_count()
                if pending > 0:
                    result = replay_queue(memory, user_id=cfg["user_id"])
                    print(
                        f"[fidelis] queue sweep: replayed={result.get('replayed', 0)} "
                        f"(verbatim_fallback={result.get('replayed_verbatim', 0)}) "
                        f"failed={result.get('failed', 0)} "
                        f"dead_lettered={result.get('dead_lettered', 0)} "
                        f"remaining={result.get('remaining', 0)} "
                        f"next_sweep_s={sleep_s}",
                        flush=True,
                    )
                    if result.get("replayed", 0) > 0:
                        sleep_s = BASE
                    else:
                        sleep_s = min(sleep_s * 2, MAX)
                else:
                    sleep_s = BASE
            except Exception as e:
                logger.debug("background replay tick failed: %s", e)
                sleep_s = min(sleep_s * 2, MAX)
            _t.sleep(sleep_s)
    replay_thread = threading.Thread(target=_replay_loop, daemon=True, name="fidelis-replay")
    replay_thread.start()

    port = cfg["port"]
    handler = make_handler(memory, cfg)
    httpd = ThreadingHTTPServer((args.host, port), handler)

    # Graceful-shutdown signal handlers. SIGTERM is what launchd/systemd send
    # on `launchctl bootout` or `systemctl stop`; SIGINT is Ctrl-C. We must
    # call httpd.shutdown() (which returns once the serve loop has cleanly
    # finished any in-flight requests) and then close the chromadb client so
    # its SQLite WAL is checkpointed to disk. Without this, a hard OS reboot
    # mid-write can leave the store in an inconsistent state — exactly the
    # data-integrity hazard a memory product cannot afford.
    import signal
    _shutdown_done = threading.Event()

    def _shutdown(signum, frame):
        if _shutdown_done.is_set():
            return
        _shutdown_done.set()
        logger.warning("received signal %s — shutting down gracefully", signum)
        # httpd.shutdown() blocks until the serve loop returns; must not be
        # called from the same thread as serve_forever (deadlocks). Spawn it.
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(f"[fidelis] Listening on {args.host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        # Always-runs cleanup, even on unhandled exceptions. Closes the
        # underlying chromadb client (and its SQLite handle) so any pending
        # WAL frames are checkpointed before process exit.
        try:
            httpd.server_close()
        except Exception as e:  # noqa: silent — best-effort socket close
            logger.debug("httpd.server_close() raised: %s", e)
        try:
            client = getattr(memory.vector_store, "client", None)
            if client is not None and hasattr(client, "_admin_client"):
                # chromadb PersistentClient — let GC trigger __del__ checkpoint
                pass
        except Exception as e:  # noqa: silent — chromadb internals may shift across versions; fall back to GC
            logger.debug("chromadb close hook raised: %s", e)
        print("[fidelis] Stopped cleanly.", flush=True)


if __name__ == "__main__":
    main()
