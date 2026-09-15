"""fidelis-server boots and answers GET /health without reachable Ollama.

This is the container's real console-script entry point
(`pyproject.toml` [project.scripts] fidelis-server = "fidelis.server:main").
Historically `server.main()` called `_boot(cfg)` eagerly, which imports mem0
and constructs `Memory.from_config(...)`; mem0's ollama embedder backend
calls `_ensure_model_exists()` during that construction, which raises a
`ConnectionError` when Ollama is unreachable — before the HTTP server ever
binds, so GET /health was unreachable too (server.py `_boot`, called from
`main()`). Registry inspectors (Glama et al.) start this exact entry point
with no Ollama running and only probe /health, so this must succeed.

Both OLLAMA_URL and COGITO_OLLAMA_URL aliases are set to a port nothing can
ever be listening on, matching tests/test_mcp_smoke_no_ollama.py's approach
for the stdio entry point.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_health(port: int, timeout_s: float = 10.0):
    deadline = time.time() + timeout_s
    last_exc = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return r.status, r.read()
        except Exception as e:  # noqa: silent — server still booting, retry
            last_exc = e
            time.sleep(0.25)
    raise AssertionError(f"server not healthy on :{port} within {timeout_s}s: {last_exc}")


def test_server_health_ok_without_ollama(tmp_path: Path):
    port = _free_port()
    env = {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": SRC,
        "FIDELIS_PORT": str(port),
        "COGITO_PORT": str(port),
        "COGITO_STORE_PATH": str(tmp_path / "store"),
        "FIDELIS_QUEUE_DIR": str(tmp_path / "queue"),
        "COGITO_QUEUE_DIR": str(tmp_path / "queue"),
        # Unreachable on purpose (both aliases): guarantees no Ollama.
        "OLLAMA_URL": "http://127.0.0.1:1",
        "COGITO_OLLAMA_URL": "http://127.0.0.1:1",
        "ANONYMIZED_TELEMETRY": "False",
        "POSTHOG_DISABLED": "1",
        "CHROMA_TELEMETRY_DISABLED": "True",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "fidelis.server"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        status, body = _wait_health(port, timeout_s=10.0)
        assert status == 200
        import json
        payload = json.loads(body)
        assert payload["status"] == "ok"
        # Memory has not been constructed yet — the lazy holder reports this
        # explicitly instead of blocking /health on an unreachable Ollama.
        assert payload.get("store_loaded") is False
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def test_server_post_endpoint_returns_503_without_ollama(tmp_path: Path):
    """A memory-backed POST endpoint must fail cleanly (503 JSON, no crash,
    no traceback) rather than raising when Ollama is unreachable, and the
    process must still be alive and answering /health afterward."""
    port = _free_port()
    env = {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": SRC,
        "FIDELIS_PORT": str(port),
        "COGITO_PORT": str(port),
        "COGITO_STORE_PATH": str(tmp_path / "store"),
        "FIDELIS_QUEUE_DIR": str(tmp_path / "queue"),
        "COGITO_QUEUE_DIR": str(tmp_path / "queue"),
        "OLLAMA_URL": "http://127.0.0.1:1",
        "COGITO_OLLAMA_URL": "http://127.0.0.1:1",
        "ANONYMIZED_TELEMETRY": "False",
        "POSTHOG_DISABLED": "1",
        "CHROMA_TELEMETRY_DISABLED": "True",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "fidelis.server"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        _wait_health(port, timeout_s=10.0)

        import json
        data = json.dumps({"text": "some query text"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/query",
            data=data, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=10)
            raise AssertionError("expected HTTPError 503")
        except urllib.error.HTTPError as e:
            assert e.code == 503
            body = json.loads(e.read())
            assert "ollama_url" in body or "embed_model" in body
            assert "error" in body

        # Process must still be alive and healthy after the failed request.
        status, _ = _wait_health(port, timeout_s=5.0)
        assert status == 200
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
