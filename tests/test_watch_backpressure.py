"""Backpressure tests for `fidelis watch`.

Three protections are required: a per-batch server-pressure check, a
per-session byte cap, and an early exit when /health is unreachable.
Without these, dropping 10,000 PDFs into a watched directory can pile
unbounded work on the embedding queue and OOM the user's machine.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from fidelis import watch_cmd


def _make_files(root: Path, n: int, body: str = "x" * 1000) -> list[Path]:
    paths = []
    for i in range(n):
        f = root / f"file_{i:04d}.md"
        f.write_text(body)
        paths.append(f)
    return paths


def test_session_byte_cap_stops_ingestion(tmp_path):
    """When the session byte cap is exhausted, ingestion stops mid-batch
    rather than continuing to push unbounded work at the server."""
    files = _make_files(tmp_path, 100, body="x" * 1000)  # 100 KB total worth
    ledger: dict = {}

    posted = []
    def fake_post(path, payload, timeout=30.0):
        posted.append(payload["text"])
        return {"status": "stored", "id": "x", "extracted": ["x"]}

    # Force "no pressure" so backpressure isn't the limiter — we want the
    # byte cap to be the gate.
    with patch.object(watch_cmd, "_post", side_effect=fake_post), \
         patch.object(watch_cmd, "_server_pressure", return_value=(False, 0)):
        ingested, pushed = watch_cmd._ingest_with_backpressure(
            files, ledger, verbose=False, bytes_budget=5000,  # 5 KB cap
        )

    # Each file is ~1 KB UTF-8 so the cap should fire after ~5 files
    assert ingested <= 6, f"byte cap not enforced: ingested {ingested} files past 5 KB cap"
    assert pushed >= 5000 or pushed >= 1000 * ingested, (
        f"bytes_pushed accounting wrong: ingested={ingested} pushed={pushed}"
    )
    # Most files should NOT have been pushed
    assert len(posted) <= 6, f"too many /store calls: {len(posted)}"


def test_server_pressure_aborts_batch(tmp_path):
    """When the server reports a deep graceful-degrade queue, the watcher
    pauses; if still pressured after the pause, it aborts the batch."""
    files = _make_files(tmp_path, watch_cmd.INGEST_BATCH_SIZE * 3)
    ledger: dict = {}

    def fake_post(path, payload, timeout=30.0):
        return {"status": "stored", "id": "x", "extracted": ["x"]}

    pressure_calls = [0]
    def fake_pressure():
        pressure_calls[0] += 1
        # First N calls: not pressured. After that: under pressure both checks.
        if pressure_calls[0] <= 1:
            return (False, 0)
        return (True, 999)

    with patch.object(watch_cmd, "_post", side_effect=fake_post), \
         patch.object(watch_cmd, "_server_pressure", side_effect=fake_pressure), \
         patch.object(watch_cmd, "time") as fake_time:
        # Don't actually sleep during the pause
        fake_time.sleep = lambda s: None
        ingested, _ = watch_cmd._ingest_with_backpressure(
            files, ledger, verbose=False, bytes_budget=10**9,
        )

    # Should have aborted before processing all files
    assert ingested < len(files), (
        f"batch did not abort under pressure: ingested {ingested}/{len(files)}"
    )
    # Should have ingested at least one batch worth
    assert ingested >= watch_cmd.INGEST_BATCH_SIZE - 1, (
        f"aborted too eagerly: ingested only {ingested} files"
    )


def test_health_unreachable_treated_as_pressure(tmp_path):
    """If /health can't be reached, _server_pressure returns under_pressure=True
    so the watcher doesn't pile on a server that may be down."""
    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("nope")):
        under, queued = watch_cmd._server_pressure()
    assert under is True
    assert queued == -1


def test_ingest_file_returns_bytes_pushed(tmp_path):
    """_ingest_file must return (success, bytes_pushed) so the caller can
    enforce session byte caps. Regression for the API change."""
    f = tmp_path / "x.md"
    body = "hello world"
    f.write_text(body)

    with patch.object(watch_cmd, "_post", return_value={"status": "stored", "id": "x"}):
        ok, pushed = watch_cmd._ingest_file(f, verbose=False)
    assert ok is True
    assert pushed == len(body.encode("utf-8"))


def test_ingest_file_skip_returns_zero_bytes(tmp_path):
    f = tmp_path / "big.md"
    f.write_text("x" * (watch_cmd.DEFAULT_MAX_FILE_BYTES + 1))
    ok, pushed = watch_cmd._ingest_file(f, verbose=False, max_bytes=watch_cmd.DEFAULT_MAX_FILE_BYTES)
    assert ok is False
    assert pushed == 0
