"""End-to-end regression test for the v0.0.9 P0 fix:
write a verbatim memory via /store and recall it via /query.

This test would FAIL on v0.0.8 (mem0.Memory.search returned score=1.0 for
every result, drowning the actual match in unrelated memories with the same
score). It PASSES on v0.0.9 (server now bypasses mem0's broken score_and_rank
and queries vector_store.search directly).

Runs against a real fidelis-server. Marked skipif Ollama isn't reachable so
it doesn't break CI for contributors without a local Ollama.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest


def _ollama_reachable() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _http_post(url: str, payload: dict, timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _wait_for_health(port: int, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    return False


@pytest.mark.skipif(not _ollama_reachable(), reason="requires local Ollama on :11434")
def test_store_query_roundtrip(tmp_path: Path):
    """Write a marker via /store, then query for its content; the marker must
    appear in /query results. Regression for the v0.0.8 score_and_rank bug.
    """
    # Use a port + store + queue dir isolated from the user's running service.
    port = 19422
    env = os.environ.copy()
    env["FIDELIS_PORT"] = str(port)
    env["COGITO_PORT"] = str(port)
    env["COGITO_STORE_PATH"] = str(tmp_path / "store")
    env["COGITO_QUEUE_DIR"] = str(tmp_path / "queue")
    # Use a model that hangs minimally — the test doesn't exercise the LLM path,
    # but Memory.from_config still tries to init the LLM client. mistral isn't
    # installed; qwen3.5:0.8b is in the install path per v0.0.9 default.
    env["COGITO_LLM_MODEL"] = "qwen3.5:0.8b"

    # Spawn server. NOTE: Memory.from_config can take 10-30s on first boot due
    # to chroma init; tolerate that.
    proc = subprocess.Popen(
        [sys.executable, "-m", "fidelis.server", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        if not _wait_for_health(port, timeout_s=60.0):
            pytest.fail(
                f"fidelis-server on :{port} did not respond to /health within 60s. "
                f"server output: {proc.stdout.read(2000).decode(errors='replace') if proc.stdout else '(none)'}"
            )

        # Write a unique marker
        marker = f"E2EMARKER-{uuid.uuid4().hex[:12]}"
        text = (
            f"On 2026-04-27 the user mentioned {marker}. They prefer Thai food on weekends "
            f"and dislike pineapple on pizza. This is a regression test for fidelis."
        )
        store_resp = _http_post(f"http://127.0.0.1:{port}/store", {"text": text})
        assert store_resp.get("status") == "stored", f"store failed: {store_resp}"

        # Give chroma a moment to settle
        time.sleep(1.0)

        # Query for the marker — must be in results
        query_resp = _http_post(
            f"http://127.0.0.1:{port}/query", {"text": marker, "limit": 3},
        )
        memories = query_resp.get("memories", [])
        assert memories, f"/query returned no memories for marker {marker}"
        # The marker must appear verbatim in at least one result's text
        hit = any(marker in m.get("text", "") for m in memories)
        assert hit, (
            f"marker {marker!r} not in any /query result. "
            f"got: {[m.get('text', '')[:80] for m in memories]}"
        )

        # Also verify semantic query (richer signal) recovers the same memory
        semantic_resp = _http_post(
            f"http://127.0.0.1:{port}/query",
            {"text": "what does the user prefer to eat", "limit": 3},
        )
        sem_memories = semantic_resp.get("memories", [])
        assert sem_memories, "semantic /query returned no memories"
        sem_hit = any("Thai food" in m.get("text", "") for m in sem_memories)
        assert sem_hit, (
            f"semantic query did not recall the Thai-food memory. "
            f"got: {[m.get('text', '')[:80] for m in sem_memories]}"
        )

        # Scores must be in [0, 1] — regression for the score_and_rank bug
        # which returned 1.0 across all results
        scores = [m.get("score") for m in memories]
        assert all(isinstance(s, (int, float)) for s in scores), f"non-numeric scores: {scores}"
        assert all(0 <= s <= 1 for s in scores), f"scores out of [0,1]: {scores}"
        # If multiple results, scores must DIFFER (not all 1.0). At least one pair.
        if len(scores) > 1:
            assert len(set(scores)) > 1, (
                f"all scores identical ({scores}); v0.0.8 score_and_rank bug regressed"
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
