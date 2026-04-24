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
  cogito-server                        # uses .cogito.json or env vars
  cogito-server --config /path/to.json
  cogito-server --port 19420
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fidelis import __version__
from fidelis.config import load, mem0_config
from fidelis.degrade import queued_count, safe_add
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

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress default logging
            pass

        def _json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
                    try:
                        count = memory.vector_store.col.count()  # type: ignore
                    except Exception:
                        result = memory.get_all(user_id=user_id, limit=10000)  # type: ignore
                        count = len(result.get("results", []))
                    snap_path = _snapshot_path(cfg)
                    self._json({
                        "status": "ok",
                        "count": count,
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
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as e:
                self._json({"error": f"internal error: {type(e).__name__}"}, 500)

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
                    raw = memory.search(text, user_id=user_id, limit=limit)  # type: ignore
                    memories = [
                        {"text": r["memory"], "score": round(r["score"], 3)}
                        for r in raw.get("results", [])
                        if r.get("memory") and r.get("score", 9999) < query_threshold
                    ]
                    self._json({"memories": memories})

                elif self.path == "/recall":
                    text = data.get("text", "")
                    if not text or len(text.strip()) < 3:
                        self._json({"memories": [], "method": "empty_query"})
                        return
                    limit = int(data.get("limit", cfg.get("recall_limit", 50)))
                    since = data.get("since")
                    memories, method = do_recall(
                        memory, text, user_id=user_id, cfg=cfg,
                        limit=limit, since=since,
                    )
                    print(f"[cogito] /recall '{text[:50]}' → {len(memories)} results ({method})", flush=True)
                    self._json({"memories": memories, "method": method})

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
                self._json({"error": f"internal error: {type(e).__name__}"}, 500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="cogito memory server")
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

    port = cfg["port"]
    handler = make_handler(memory, cfg)
    httpd = ThreadingHTTPServer((args.host, port), handler)
    print(f"[cogito] Listening on {args.host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[cogito] Stopped.")


if __name__ == "__main__":
    main()
