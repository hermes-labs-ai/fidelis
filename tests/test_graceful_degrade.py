"""Graceful-degradation tests — Hermes Seal continuity category."""

from __future__ import annotations


import pytest

from fidelis import degrade


@pytest.fixture(autouse=True)
def temp_queue(tmp_path, monkeypatch):
    qdir = tmp_path / "queue"
    monkeypatch.setenv("FIDELIS_QUEUE_DIR", str(qdir))
    yield qdir


class _BrokenOllamaMemory:
    """Mem0 substitute that raises when Ollama is down."""

    def add(self, text, user_id):
        raise ConnectionError("ollama unreachable: socket EOF")


class _WorkingMemory:
    def __init__(self):
        self.added = []

    def add(self, text, user_id):
        self.added.append((text, user_id))
        return {"results": [{"memory": text}]}


def test_add_queues_when_ollama_down(temp_queue):
    """The 2026-04-19 incident regression test."""
    mem = _BrokenOllamaMemory()
    result = degrade.safe_add(mem, "an important memory", user_id="agent")
    assert result["status"] == "queued"
    assert "ConnectionError" in result["reason"]
    queued = list(temp_queue.glob("*.json"))
    assert len(queued) == 1


def test_add_succeeds_when_dependency_up(temp_queue):
    mem = _WorkingMemory()
    result = degrade.safe_add(mem, "another memory", user_id="agent")
    assert result["status"] == "stored"
    assert result["extracted"] == ["another memory"]
    assert list(temp_queue.glob("*.json")) == []


def test_replay_drains_queue_when_dependency_recovers(temp_queue):
    bad = _BrokenOllamaMemory()
    for i in range(3):
        degrade.safe_add(bad, f"memory {i}", user_id="agent")
    assert degrade.queued_count() == 3

    good = _WorkingMemory()
    summary = degrade.replay_queue(good, user_id="agent")
    assert summary["replayed"] == 3
    assert summary["remaining"] == 0
    assert len(good.added) == 3


def test_replay_keeps_records_when_dependency_still_down(temp_queue):
    bad = _BrokenOllamaMemory()
    degrade.safe_add(bad, "stuck memory", user_id="agent")
    summary = degrade.replay_queue(bad, user_id="agent")
    assert summary["replayed"] == 0
    assert summary["remaining"] == 1
