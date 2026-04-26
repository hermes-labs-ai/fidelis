"""Per-item attempts cap + dead-letter behavior for replay_queue.

Without these tests, a permanently poisoned record could keep the replay
loop hot forever, firing the embedder + LLM on every 60s sweep. The
MAX_ATTEMPTS=5 cap and the dead-letter sub-directory prevent that.
"""
from __future__ import annotations

import json

import pytest

from fidelis import degrade


@pytest.fixture
def temp_queue(tmp_path, monkeypatch):
    qdir = tmp_path / "queue"
    monkeypatch.setenv("FIDELIS_QUEUE_DIR", str(qdir))
    return qdir


class _AlwaysFailingMemory:
    """Both add() and the verbatim path fail."""
    class _Embedder:
        def embed(self, text, *args, **kwargs):
            raise ConnectionError("ollama down")

    class _Store:
        def insert(self, **kwargs):
            raise RuntimeError("chroma write failed")

    def __init__(self):
        self.embedding_model = self._Embedder()
        self.vector_store = self._Store()

    def add(self, text, user_id):
        raise ConnectionError("ollama down")


def test_dead_letter_after_max_attempts(temp_queue):
    """A permanently failing record must move to dead/ after MAX_ATTEMPTS."""
    mem = _AlwaysFailingMemory()
    degrade.queue_write("poison", user_id="agent")
    assert degrade.queued_count() == 1

    for sweep in range(degrade.MAX_ATTEMPTS):
        summary = degrade.replay_queue(mem, user_id="agent")
        if sweep < degrade.MAX_ATTEMPTS - 1:
            assert summary["failed"] == 1, f"sweep {sweep}: expected failure, got {summary}"
            assert summary["remaining"] == 1
            assert summary["dead_lettered"] == 0
        else:
            assert summary["dead_lettered"] == 1
            assert summary["remaining"] == 0

    assert degrade.queued_count() == 0
    assert degrade.dead_count() == 1


def test_dead_letter_preserves_record_with_error(temp_queue):
    mem = _AlwaysFailingMemory()
    mid = degrade.queue_write("poison-with-id", user_id="agent")
    for _ in range(degrade.MAX_ATTEMPTS):
        degrade.replay_queue(mem, user_id="agent")

    dead_files = list((temp_queue / "dead").glob("*.json"))
    assert len(dead_files) == 1
    rec = json.loads(dead_files[0].read_text())
    assert rec["id"] == mid
    assert rec["text"] == "poison-with-id"
    assert rec["attempts"] == degrade.MAX_ATTEMPTS
    assert "last_error" in rec  # forensic trail


def test_attempts_counter_increments_atomically(temp_queue):
    """Each failed sweep must persist attempts back to the queue file
    atomically — no half-written files that would corrupt on reload."""
    mem = _AlwaysFailingMemory()
    degrade.queue_write("poison", user_id="agent")

    qfile = next(temp_queue.glob("*.json"))
    initial = json.loads(qfile.read_text())
    assert initial["attempts"] == 0

    degrade.replay_queue(mem, user_id="agent")
    after_one = json.loads(qfile.read_text())
    assert after_one["attempts"] == 1
    assert after_one["text"] == "poison"  # body preserved

    degrade.replay_queue(mem, user_id="agent")
    after_two = json.loads(qfile.read_text())
    assert after_two["attempts"] == 2


def test_atomic_write_no_tmp_leftovers(temp_queue):
    """queue_write must not leave .tmp files behind."""
    for i in range(5):
        degrade.queue_write(f"item {i}", user_id="agent")
    leftovers = list(temp_queue.glob("*.tmp"))
    assert leftovers == [], f"unexpected tmp files: {leftovers}"
    assert degrade.queued_count() == 5


class _IdempotentVerbatimMemory:
    """Simulates a partial prior insert — second call raises 'duplicate id'."""
    class _Embedder:
        def embed(self, text, *args, **kwargs):
            return [0.0] * 8

    class _Store:
        def __init__(self):
            self.calls = 0

        def insert(self, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("Insert of existing embedding ID: foo")

    def __init__(self):
        self.embedding_model = self._Embedder()
        self.vector_store = self._Store()


def test_replay_verbatim_idempotent_on_duplicate_id(temp_queue):
    """Retry of a partially-succeeded insert (duplicate-id error) must be
    treated as success, not as another failed attempt."""
    mem = _IdempotentVerbatimMemory()
    rec = {"id": "fixed-id", "text": "stuck-write", "user_id": "agent", "kind": "store"}
    degrade._replay_verbatim(mem, rec)  # first insert succeeds
    degrade._replay_verbatim(mem, rec)  # second raises duplicate, must NOT propagate
