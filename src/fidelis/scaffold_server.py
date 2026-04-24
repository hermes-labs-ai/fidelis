"""
cogito scaffold server — scaffold memory for AI agents over HTTP.

Endpoints:
  POST /session  {"role": "...", "goal": "...", "style": "...", "constraints": [...]}
       → {"session_id": "...", "scaffold": "...", "turn": 0}
       Create a new session with initial state.

  POST /turn     {"session_id": "...", "user": "...", "assistant": "..."}
       → {"scaffold": "...", "turn": N}
       Process a conversation turn. Extracts state delta, rewrites scaffold.

  GET  /scaffold?session_id=...
       → {"scaffold": "...", "turn": N}
       Read current scaffold without modifying it.

  GET  /health
       → {"status": "ok", "sessions": N, "version": "..."}

Start:
  cogito-scaffold                        # default port 19421
  cogito-scaffold --port 19421
"""

from __future__ import annotations

import argparse
import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from fidelis import __version__
from fidelis.config import load
from fidelis.scaffold import (
    SessionState,
    apply_delta,
    extract_state_delta,
    load_session,
    save_session,
)


def make_handler(sessions: dict[str, SessionState], cfg: dict) -> type:
    budget = cfg.get("scaffold_budget", 7000)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict | None:
            n = int(self.headers.get("Content-Length", 0))
            if n > 1_048_576:
                return None
            raw = self.rfile.read(n)
            try:
                return json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return {}

        def _get_session(self, session_id: str) -> SessionState | None:
            """Look up session in memory, falling back to disk."""
            if session_id in sessions:
                return sessions[session_id]
            state = load_session(session_id)
            if state:
                sessions[session_id] = state
            return state

        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path == "/health":
                self._json({
                    "status": "ok",
                    "sessions": len(sessions),
                    "version": __version__,
                })

            elif parsed.path == "/scaffold":
                qs = parse_qs(parsed.query)
                sid = qs.get("session_id", [None])[0]
                if not sid:
                    self._json({"error": "session_id required"}, 400)
                    return
                state = self._get_session(sid)
                if not state:
                    self._json({"error": "session not found"}, 404)
                    return
                self._json({
                    "scaffold": state.to_scaffold(token_budget=budget),
                    "turn": state.turn,
                })

            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            data = self._read_body()
            if data is None:
                self._json({"error": "request body too large"}, 413)
                return

            if self.path == "/session":
                sid = str(uuid.uuid4())
                state = SessionState(
                    role=data.get("role", ""),
                    style=data.get("style", ""),
                    goal=data.get("goal", ""),
                    constraints=data.get("constraints", []),
                )
                sessions[sid] = state
                save_session(sid, state)
                print(f"[scaffold] new session {sid[:8]}  role={state.role[:30]}", flush=True)
                self._json({
                    "session_id": sid,
                    "scaffold": state.to_scaffold(token_budget=budget),
                    "turn": 0,
                })

            elif self.path == "/turn":
                sid = data.get("session_id", "")
                user_msg = data.get("user", "")
                asst_msg = data.get("assistant", "")

                if not sid:
                    self._json({"error": "session_id required"}, 400)
                    return
                if not user_msg and not asst_msg:
                    self._json({"error": "user and/or assistant message required"}, 400)
                    return

                state = self._get_session(sid)
                if not state:
                    self._json({"error": "session not found"}, 404)
                    return

                # Extract delta and apply
                delta = extract_state_delta(state, user_msg, asst_msg, cfg)
                apply_delta(state, delta)
                save_session(sid, state)

                scaffold = state.to_scaffold(token_budget=budget)
                delta_keys = [k for k in delta if k not in ("goal", "style")]
                print(
                    f"[scaffold] turn {state.turn}  session={sid[:8]}  "
                    f"delta={delta_keys}  scaffold={len(scaffold)} chars",
                    flush=True,
                )
                self._json({
                    "scaffold": scaffold,
                    "turn": state.turn,
                })

            else:
                self._json({"error": "not found"}, 404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="cogito scaffold memory server")
    parser.add_argument("--config", help="Path to .cogito.json")
    parser.add_argument("--port", type=int, default=19421, help="Port (default: 19421)")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    cfg = load(args.config)
    cfg["scaffold_port"] = args.port

    sessions: dict[str, SessionState] = {}
    handler = make_handler(sessions, cfg)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"[scaffold] cogito scaffold server v{__version__}", flush=True)
    print(f"[scaffold] model={cfg.get('scaffold_model', 'qwen3.5:4b')}  budget={cfg.get('scaffold_budget', 7000)} tokens", flush=True)
    print(f"[scaffold] listening on {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[scaffold] stopped.", flush=True)


if __name__ == "__main__":
    main()
