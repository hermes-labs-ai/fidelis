"""Queue-corruption branch coverage for degrade.replay_queue.

Corrupted JSON files in the queue are moved to a dead-letter subdirectory
so the replay loop cannot stay hot indefinitely re-trying unparseable
records. The originals are preserved (in dead/) for forensic inspection.
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
    qdir = tmp_path / "queue"
    monkeypatch.setenv("FIDELIS_QUEUE_DIR", str(qdir))
    return qdir


def test_replay_dead_letters_corrupted_json(temp_queue):
    temp_queue.mkdir(parents=True, exist_ok=True)
    (temp_queue / "corrupt.json").write_text("{not valid json")
    (temp_queue / "good.json").write_text(
        json.dumps({"id": "x", "ts": 0, "kind": "add", "user_id": "u", "text": "ok"})
    )
    summary = degrade.replay_queue(_FakeMemory(), user_id="u")
    assert summary["replayed"] == 1
    assert summary["dead_lettered"] == 1
    assert summary["remaining"] == 0  # corrupt file moved to dead/, not in main queue
    assert not (temp_queue / "corrupt.json").exists()
    assert (temp_queue / "dead" / "corrupt.json").exists()


def test_replay_corrupt_only_queue_drains_to_dead(temp_queue):
    temp_queue.mkdir(parents=True, exist_ok=True)
    (temp_queue / "a.json").write_text("garbage")
    (temp_queue / "b.json").write_text("")
    summary = degrade.replay_queue(_FakeMemory(), user_id="u")
    assert summary["replayed"] == 0
    assert summary["dead_lettered"] == 2
    assert summary["remaining"] == 0
    assert (temp_queue / "dead").exists()
