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
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from cogito import __version__
from cogito.config import load, mem0_config
from cogito.recall import recall as do_recall


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

        def _read_body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            try:
                return json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if self.path == "/health":
                try:
                    count = memory.vector_store.col.count()  # type: ignore
                except Exception:
                    result = memory.get_all(user_id=user_id, limit=10000)  # type: ignore
                    count = len(result.get("results", []))
                self._json({"status": "ok", "count": count, "version": __version__})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            data = self._read_body()
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
                memories, method = do_recall(
                    memory, text, user_id=user_id, cfg=cfg,
                    limit=limit,
                )
                print(f"[cogito] /recall '{text[:50]}' → {len(memories)} results ({method})", flush=True)
                self._json({"memories": memories, "method": method})

            elif self.path == "/store":
                # Verbatim write — agent decides content, no extraction LLM.
                text = data.get("text", "")
                if not text or len(text.strip()) < 3:
                    self._json({"error": "no text"}, 400)
                    return
                mem_id = data.get("id") or str(uuid.uuid4())
                try:
                    vector = memory.embedding_model.embed(text)  # type: ignore
                    memory.vector_store.insert(  # type: ignore
                        vectors=[vector],
                        payloads=[{"data": text, "user_id": user_id}],
                        ids=[mem_id],
                    )
                    self._json({"id": mem_id, "text": text})
                except Exception as e:
                    self._json({"error": str(e)}, 500)

            elif self.path == "/add":
                text = data.get("text", "")
                if not text:
                    self._json({"error": "no text"}, 400)
                    return
                result = memory.add(text, user_id=user_id)  # type: ignore
                extracted = result.get("results", [])
                self._json({
                    "count": len(extracted),
                    "memories": [m.get("memory", "") for m in extracted],
                })

            else:
                self._json({"error": "not found"}, 404)

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
    print(f"[cogito] Loading memory store...", flush=True)

    memory = _boot(cfg)

    port = cfg["port"]
    handler = make_handler(memory, cfg)
    httpd = HTTPServer((args.host, port), handler)
    print(f"[cogito] Listening on {args.host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[cogito] Stopped.")


if __name__ == "__main__":
    main()
