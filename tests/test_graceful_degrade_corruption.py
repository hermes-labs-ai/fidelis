"""Queue-corruption branch coverage for degrade.replay_queue.

degrade.py has a corrupted-file branch:

    try:
        rec = json.loads(qfile.read_text())
    except Exception:
        failed += 1
        continue

Without a test, the branch can rot silently. These tests pin it.
"""
from __future__ import annotations

import json

import pytest

from fidelis import degrade


class _FakeMemory:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.added: list[tuple[str, str]] = []

    def add(self, text: str, user_id: str):
        if self.fail:
            raise ConnectionError("still down")
        self.added.append((text, user_id))
        return {"results": [{"memory": text}]}


@pytest.fixture
def temp_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(degrade, "QUEUE_DIR", tmp_path / "queue")
    return tmp_path / "queue"


def test_replay_counts_corrupted_queue_file_as_failed(temp_queue):
    temp_queue.mkdir(parents=True, exist_ok=True)
    (temp_queue / "corrupt.json").write_text("{not valid json")
    (temp_queue / "good.json").write_text(
        json.dumps({"id": "x", "ts": 0, "kind": "add", "user_id": "u", "text": "ok"})
    )
    summary = degrade.replay_queue(_FakeMemory(), user_id="u")
    assert summary["replayed"] == 1
    assert summary["remaining"] == 1  # corrupt file kept on disk, not silently lost
    assert (temp_queue / "corrupt.json").exists()


def test_replay_corrupt_only_queue_is_noop(temp_queue):
    temp_queue.mkdir(parents=True, exist_ok=True)
    (temp_queue / "a.json").write_text("garbage")
    (temp_queue / "b.json").write_text("")
    summary = degrade.replay_queue(_FakeMemory(), user_id="u")
    assert summary["replayed"] == 0
    assert summary["remaining"] == 2
