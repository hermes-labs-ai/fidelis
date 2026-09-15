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
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fidelis import __version__
from fidelis.config import load, mem0_config
from fidelis.degrade import queue_write, queued_count, replay_queue, safe_add
from fidelis.recall import recall as do_recall
from fidelis.recall_b import recall_b as do_recall_b
from fidelis.recall_hybrid import recall_hybrid as do_recall_hybrid
from fidelis.snapshot import _read_snapshot, _snapshot_path

logger = logging.getLogger("cogito.server")


class MemoryHolder:
    """Thread-safe, once-only, retry-on-failure lazy initializer for the mem0
    Memory object.

    Registry inspectors (Glama et al.) start the container's console-script
    entry point without a reachable Ollama and only probe GET /health. The
    real Memory object depends on mem0's ollama embedder, whose
    `_ensure_model_exists()` raises a ConnectionError when Ollama is down
    (server.py `_boot`, historically called eagerly before the HTTP server
    even bound). This holder defers that call to first use by a request
    handler that actually needs the store, caches a successful construction
    forever, and caches the last failure only until the next request asks
    for the memory again — so a transient outage does not permanently
    poison the process; each request after a failure gets a fresh attempt.
    """

    def __init__(self, cfg: dict):
        self._cfg = cfg
        self._lock = threading.Lock()
        self._memory: object | None = None
        self._last_error: Exception | None = None
        self._last_error_at: float = 0.0
        # Minimum seconds between retry attempts after a failure. Prevents
        # a burst of concurrent requests (or a registry inspector probing
        # /query, /recall, etc. in a tight loop) from retrying `_boot` —
        # which dials Ollama over the network — once per request. A single
        # request past the cooldown gets the real retry (holding `_lock`);
        # everything else in that window reuses the cached failure.
        self._retry_cooldown_s: float = self._parse_retry_cooldown(
            os.environ.get("FIDELIS_MEMORY_RETRY_COOLDOWN_SECS")
        )

    @staticmethod
    def _parse_retry_cooldown(raw: str | None, default: float = 5.0) -> float:
        """Parse FIDELIS_MEMORY_RETRY_COOLDOWN_SECS defensively.

        A malformed, negative, NaN, or infinite value must never abort
        startup — fall back to the documented default instead."""
        if raw is None:
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(value) or value < 0:
            return default
        return value

    @property
    def ready(self) -> bool:
        return self._memory is not None

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def get(self) -> object:
        """Return the constructed Memory, building it on first call, or
        retrying no more than once per cooldown window after a failure.
        Raises the underlying exception (fresh or cached) on failure —
        never caches a permanent poison state, but never hammers Ollama
        on every single request either."""
        with self._lock:
            if self._memory is not None:
                return self._memory
            if (
                self._last_error is not None
                and time.monotonic() - self._last_error_at < self._retry_cooldown_s
            ):
                raise self._last_error
            try:
                memory = _boot(self._cfg)
            except Exception as e:
                self._last_error = e
                # Timestamp the failure itself, not the moment we entered
                # this call — _boot() dials Ollama over the network and can
                # take a while to time out, so measuring from entry would
                # under-count the cooldown window.
                self._last_error_at = time.monotonic()
                raise
            self._memory = memory
            self._last_error = None
            return memory


def _boot(cfg: dict) -> object:
    """Import mem0 from wherever it's installed and return a Memory instance."""
    # Support venv via COGITO_SITE_PACKAGES or system install
    site = os.environ.get("COGITO_SITE_PACKAGES")
    if site and site not in sys.path:
        sys.path.insert(0, site)

    # Match the service installed by `fidelis init`: keep mem0 telemetry off
    # unless an operator explicitly opts in. Besides preserving the documented
    # local-first boundary for direct launches, this prevents posthog's exit
    # handlers from delaying process termination after a graceful SIGTERM.
    os.environ.setdefault("MEM0_TELEMETRY", "False")

    from mem0 import Memory  # type: ignore

    m = Memory.from_config(mem0_config(cfg))
    return m


def make_handler(memory: object, cfg: dict) -> type:
    user_id: str = cfg["user_id"]
    # FIDELIS_DECOMPOSE_TIMEOUT_SECS: max seconds for /recall sub-query pipeline.
    # Default 8s preserves existing behavior in normal cases; kicks in only on slow-call edges.
    _decompose_timeout: float = float(os.environ.get("FIDELIS_DECOMPOSE_TIMEOUT_SECS", 8))

    # `memory` is either a concrete mem0 Memory (existing call sites / tests —
    # behavior is unchanged, identical to before this change) or a
    # MemoryHolder (server.main's lazy path: Memory isn't built until a
    # handler actually needs it). _get_memory() resolves either shape;
    # _memory_unavailable_response() renders the 503 for lazy-init failures.
    def _get_memory() -> object:
        if isinstance(memory, MemoryHolder):
            return memory.get()
        return memory

    def _memory_unavailable_response(e: Exception) -> dict:
        # Log full diagnostics server-side; never expose exception text,
        # embed_model, or ollama_url to HTTP clients — those reveal internal
        # dependency configuration to anyone who can reach the port.
        logger.warning(
            "memory store unavailable: %s: %s (embed_model=%s ollama_url=%s)",
            type(e).__name__,
            e,
            cfg.get("embed_model"),
            cfg.get("ollama_url"),
        )
        return {"error": "memory store unavailable"}

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
                    # Never construct Memory here — this is the endpoint
                    # registry inspectors (Glama et al.) probe immediately
                    # after boot, with no Ollama reachable. When the store
                    # isn't loaded yet (lazy path, not yet first-used), report
                    # what we can without blocking or crashing: same top-level
                    # shape as before, `count` degrades to -1 (the existing
                    # "chroma unhealthy" signal) and a new `store_loaded` flag
                    # says whether Memory has actually been constructed.
                    if isinstance(memory, MemoryHolder) and not memory.ready:
                        snap_path = _snapshot_path(cfg)
                        self._json({
                            # A prior lazy-construction attempt having failed
                            # (Ollama unreachable, etc.) is real degradation —
                            # report it rather than a blanket "ok", while
                            # still never blocking on a fresh construction
                            # attempt from this read-only endpoint.
                            "status": "degraded" if memory.last_error is not None else "ok",
                            "count": -1,
                            "queued": queued_count(),
                            "version": __version__,
                            "calibrated": bool(cfg.get("vocab_map")),
                            "snapshot": snap_path.exists(),
                            "store_loaded": False,
                        })
                        return
                    # Read count directly from chroma. Previous code used
                    # get_all with top_k=10000 which (a) silently capped the
                    # reported count at 10000 and (b) hammered Ollama on
                    # every probe. mem0 2.0's ChromaDB wrapper exposes the
                    # underlying chromadb.Collection as `.collection`; its
                    # `.count()` is O(1).
                    try:
                        active_memory = _get_memory()
                        count = active_memory.vector_store.collection.count()  # type: ignore
                    except Exception as e:
                        count = -1  # signal: chroma unhealthy
                        logger.warning("health: chroma count failed: %s", e)
                    snap_path = _snapshot_path(cfg)
                    resp = {
                        "status": "ok" if count >= 0 else "degraded",
                        "count": count,
                        "queued": queued_count(),
                        "version": __version__,
                        "calibrated": bool(cfg.get("vocab_map")),
                        "snapshot": snap_path.exists(),
                    }
                    if isinstance(memory, MemoryHolder):
                        resp["store_loaded"] = memory.ready
                    self._json(resp)
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
                    try:
                        active_memory = _get_memory()
                    except Exception as e:
                        self._json(_memory_unavailable_response(e), 503)
                        return
                    result = replay_queue(active_memory, user_id=user_id)  # type: ignore
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

                # /store and /add are durable-write endpoints: they already
                # have a local queue for the case where writing THROUGH mem0
                # fails (Ollama down mid-write — see safe_add/degrade.py).
                # The same queue absorbs the case where Memory can't be
                # constructed AT ALL yet (Ollama unreachable since boot):
                # queue directly and skip straight to the "queued" response,
                # so a registry inspector's probe write is never lost and
                # never sees a 503. Every other endpoint below is read-only
                # against the store; each resolves Memory itself, after its
                # own input validation, so a 404 (unknown path) or an
                # empty/too-short query never triggers a memory-construction
                # attempt.
                if self.path in ("/store", "/add"):
                    text = data.get("text", "")
                    min_len = 3 if self.path == "/store" else 0
                    if not text or len(text.strip()) < min_len:
                        self._json({"error": "no text"}, 400)
                        return
                    try:
                        active_memory = _get_memory()
                    except Exception as e:
                        mid = queue_write(text, user_id, kind="store" if self.path == "/store" else "add")
                        logger.warning(
                            "write queued, memory unavailable: %s: %s", type(e).__name__, e
                        )
                        self._json({
                            "status": "queued",
                            "id": mid,
                            "queued_total": queued_count(),
                        }, 202)
                        return
                    if self.path == "/store":
                        # Verbatim write — agent decides content, no extraction LLM.
                        # Uses safe_add: queues locally if dependency (Ollama) is down.
                        result = safe_add(active_memory, text, user_id, kind="store")  # type: ignore
                        self._json({**result, "queued_total": queued_count()})
                    else:
                        result = safe_add(active_memory, text, user_id, kind="add")  # type: ignore
                        if result["status"] == "queued":
                            self._json({
                                "status": "queued",
                                "id": result["id"],
                                "reason": result["reason"],
                                "queued_total": queued_count(),
                            }, 202)  # 202 Accepted: write deferred
                        else:
                            extracted = result.get("extracted", [])
                            response = {
                                "status": "stored",
                                "count": len(extracted),
                                "memories": extracted,
                            }
                            if result.get("degraded"):
                                response["degraded"] = result["degraded"]
                                response["id"] = result.get("id")
                            self._json(response)
                    return

                if self.path == "/query":
                    text = data.get("text", "")
                    limit = int(data.get("limit", 5))
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": []})
                        return
                    try:
                        active_memory = _get_memory()
                    except Exception as e:
                        self._json(_memory_unavailable_response(e), 503)
                        return
                    # Bypass mem0.Memory.search wrapper: it routes through
                    # score_and_rank which (in mem0 2.0.x) returns broken
                    # score=1.0 for every result regardless of similarity.
                    # Verified empirically — same query, when we go directly to
                    # vector_store.search, returns proper distances (the actual
                    # text-match record scores 0.5878 vs unrelated at 1.07+).
                    qv = active_memory.embedding_model.embed(text, memory_action="search")  # type: ignore
                    raw = active_memory.vector_store.search(  # type: ignore
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
                    try:
                        active_memory = _get_memory()
                    except Exception as e:
                        self._json(_memory_unavailable_response(e), 503)
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    since = data.get("since")
                    degraded = False
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                            _fut = _pool.submit(
                                do_recall, active_memory, text,
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
                        qv = active_memory.embedding_model.embed(text, memory_action="search")
                        raw = active_memory.vector_store.search(
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
                    try:
                        active_memory = _get_memory()
                    except Exception as e:
                        self._json(_memory_unavailable_response(e), 503)
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    memories, method = do_recall_b(
                        active_memory, text, user_id=user_id, cfg=cfg,
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
                    try:
                        active_memory = _get_memory()
                    except Exception as e:
                        self._json(_memory_unavailable_response(e), 503)
                        return
                    memories, method = do_recall_hybrid(
                        active_memory, text, user_id=user_id, cfg=cfg,
                        limit=limit, tier=tier, top_k=top_k,
                    )
                    print(f"[cogito] /recall_hybrid '{text[:50]}' tier={tier} → {len(memories)} results ({method})", flush=True)
                    self._json({"memories": memories, "method": method})

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

    # Lazy memory construction: the historical eager `_boot(cfg)` call here
    # required a reachable Ollama before the HTTP server even bound, so a
    # registry inspector (Glama et al.) that starts this entry point with no
    # Ollama running never got past this line before crashing — GET /health
    # was unreachable. MemoryHolder defers the real construction to first
    # use by a request handler and is thread-safe / retry-on-failure (see
    # its docstring above). This preserves identical behavior when Ollama IS
    # reachable: the first request just pays the one-time construction cost
    # instead of it happening before bind.
    memory = MemoryHolder(cfg)

    # Background replay thread — sweeps the queue every 60s. Items that failed
    # at write time (Ollama momentarily unreachable, embed timeout) get retried
    # without requiring a server restart. The first sweep runs ~5s after server
    # start so HTTP serving is up immediately rather than blocking on a long
    # drain. Items stay in the queue across server restarts.
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
                    # Sweeping requires the real Memory object; if it can't
                    # be constructed yet (Ollama still unreachable), this
                    # raises and the outer except below backs off — the
                    # queue simply waits for a later sweep, same as before.
                    active_memory = memory.get()
                    result = replay_queue(active_memory, user_id=cfg["user_id"])
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
            # Only touch vector_store if Memory was actually constructed —
            # a server that shuts down before ever handling a memory-backed
            # request (e.g. the registry inspector's health-only probe) has
            # nothing to checkpoint.
            if memory.ready:
                active_memory = memory.get()
                client = getattr(active_memory.vector_store, "client", None)
                if client is not None and hasattr(client, "_admin_client"):
                    # chromadb PersistentClient — let GC trigger __del__ checkpoint
                    pass
        except Exception as e:  # noqa: silent — chromadb internals may shift across versions; fall back to GC
            logger.debug("chromadb close hook raised: %s", e)
        print("[fidelis] Stopped cleanly.", flush=True)


if __name__ == "__main__":
    main()
