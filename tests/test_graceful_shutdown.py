"""Graceful shutdown — SIGTERM/SIGINT handlers must trigger a clean
httpd.shutdown() so in-flight requests complete, then close the chromadb
connection so SQLite WAL is checkpointed before process exit.

Without this, a `launchctl bootout` mid-write or an OS reboot can leave
the store in an inconsistent state — exactly the data-integrity hazard a
memory product cannot afford.
"""
from __future__ import annotations

import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ollama_reachable() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:  # noqa: silent — connectivity probe, any failure means not reachable
        return False


def _wait_health(port: int, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: silent — server still booting
            pass
        time.sleep(0.5)
    return False


@pytest.mark.skipif(not _ollama_reachable(), reason="requires local Ollama on :11434")
def test_sigterm_triggers_clean_shutdown(tmp_path: Path):
    """Server must respond to SIGTERM by exiting cleanly within 5s, printing
    the 'Stopped cleanly' marker. Hard regression for the graceful-shutdown
    pillar: without the signal handlers, SIGTERM kills mid-write."""
    port = _free_port()
    env = {
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        "FIDELIS_PORT": str(port),
        "COGITO_PORT": str(port),
        "COGITO_STORE_PATH": str(tmp_path / "store"),
        "FIDELIS_QUEUE_DIR": str(tmp_path / "queue"),
        "COGITO_LLM_MODEL": "qwen3.5:0.8b",
        "ANONYMIZED_TELEMETRY": "False",
        "POSTHOG_DISABLED": "1",
        "CHROMA_TELEMETRY_DISABLED": "True",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "fidelis.server"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        if not _wait_health(port, timeout_s=60.0):
            output = proc.stdout.read(2000) if proc.stdout else "(none)"
            pytest.fail(f"server not healthy on :{port} within 60s. output:\n{output}")

        # Send SIGTERM and verify the process exits cleanly within 5s.
        proc.send_signal(signal.SIGTERM)
        try:
            return_code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("SIGTERM did not stop the server within 5s — graceful shutdown regressed")

        # Process must have exited with 0 (clean) — anything else means an
        # unhandled exception killed it instead of the signal handler.
        assert return_code == 0, f"server exited with {return_code} after SIGTERM"

        # Output must contain the clean-stop marker — proves the finally block ran.
        output = proc.stdout.read() if proc.stdout else ""
        assert "Stopped cleanly" in output, (
            f"server exited but did NOT print 'Stopped cleanly' — finally block "
            f"may not have run, SQLite WAL may not be checkpointed.\n"
            f"output tail:\n{output[-1000:]}"
        )
    finally:
        if proc.poll() is None:
            proc.kill()


def test_signal_handlers_registered_in_main():
    """Light-weight test: verify the handler-registration code path exists in
    server.main(). Catches accidental removal of the signal hooks even if the
    Ollama-backed integration test is skipped on a CI runner."""
    src = (Path(__file__).resolve().parent.parent / "src/fidelis/server.py").read_text()
    assert "signal.SIGTERM" in src, "SIGTERM handler missing from server.main()"
    assert "signal.SIGINT" in src, "SIGINT handler missing from server.main()"
    assert "httpd.shutdown" in src, "graceful httpd.shutdown call missing"
    assert "Stopped cleanly" in src, "clean-stop marker missing — friend will not see exit signal"
