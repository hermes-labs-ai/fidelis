"""Graceful-degradation tests — Hermes Seal continuity category."""

from __future__ import annotations


import pytest
from chromadb.errors import DuplicateIDError

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


class _RecordingVectorStore:
    def __init__(self):
        self.inserted = []

    def insert(self, *, vectors, payloads, ids):
        self.inserted.append({"vectors": vectors, "payloads": payloads, "ids": ids})


class _WorkingEmbedder:
    def embed(self, text, *args, **kwargs):
        return [0.0] * 8


class _EmptyExtractionMemory:
    def __init__(self, add_result):
        self.add_result = add_result
        self.embedding_model = _WorkingEmbedder()
        self.vector_store = _RecordingVectorStore()

    def add(self, text, user_id):
        return self.add_result


class _BrokenFallbackMemory(_EmptyExtractionMemory):
    class _BrokenEmbedder:
        def embed(self, text, *args, **kwargs):
            raise ConnectionError("embedding unavailable")

    def __init__(self):
        super().__init__({"results": []})
        self.embedding_model = self._BrokenEmbedder()


class _BrokenInsertMemory(_EmptyExtractionMemory):
    class _BrokenStore:
        def insert(self, *, vectors, payloads, ids):
            raise RuntimeError("collection does not exist")

    def __init__(self):
        super().__init__({"results": []})
        self.vector_store = self._BrokenStore()


class _DuplicateInsertMemory(_EmptyExtractionMemory):
    class _DuplicateStore:
        def insert(self, *, vectors, payloads, ids):
            raise DuplicateIDError(f"Expected IDs to be unique, found duplicates of: {ids[0]}")

    def __init__(self):
        super().__init__({"results": []})
        self.vector_store = self._DuplicateStore()


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


@pytest.mark.parametrize(
    "add_result",
    [
        {"results": []},
        None,
        {"results": [{"memory": ""}]},
        {"results": [{"memory": None}]},
    ],
)
def test_empty_extraction_falls_back_to_verbatim(temp_queue, add_result):
    mem = _EmptyExtractionMemory(add_result)

    result = degrade.safe_add(mem, "durable original text", user_id="agent")

    assert result["status"] == "stored"
    assert result["extracted"] == ["durable original text"]
    assert result["degraded"] == "verbatim-fallback-empty-extraction"
    assert result["id"]
    assert len(mem.vector_store.inserted) == 1
    assert mem.vector_store.inserted[0]["payloads"] == [
        {"data": "durable original text", "user_id": "agent"}
    ]
    assert list(temp_queue.glob("*.json")) == []


def test_failed_verbatim_fallback_queues_original_text(temp_queue):
    mem = _BrokenFallbackMemory()

    result = degrade.safe_add(mem, "queue this exact text", user_id="agent")

    assert result["status"] == "queued"
    queued = list(temp_queue.glob("*.json"))
    assert len(queued) == 1
    assert "queue this exact text" in queued[0].read_text()


def test_non_duplicate_insert_failure_queues_original_text(temp_queue):
    mem = _BrokenInsertMemory()

    result = degrade.safe_add(mem, "preserve after insert failure", user_id="agent")

    assert result["status"] == "queued"
    queued = list(temp_queue.glob("*.json"))
    assert len(queued) == 1
    assert "preserve after insert failure" in queued[0].read_text()


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


@pytest.mark.parametrize(
    "add_result",
    [
        {"results": []},
        {"results": [{"memory": ""}]},
        {"results": [{"memory": None}]},
    ],
)
def test_replay_empty_extraction_falls_back_before_deleting_queue(temp_queue, add_result):
    degrade.queue_write("queued durable text", user_id="agent")
    mem = _EmptyExtractionMemory(add_result)

    summary = degrade.replay_queue(mem, user_id="agent")

    assert summary["replayed"] == 1
    assert summary["replayed_verbatim"] == 1
    assert summary["remaining"] == 0
    assert len(mem.vector_store.inserted) == 1
    assert mem.vector_store.inserted[0]["payloads"] == [
        {"data": "queued durable text", "user_id": "agent"}
    ]


def test_replay_non_duplicate_insert_failure_keeps_queue_record(temp_queue):
    degrade.queue_write("queued after insert failure", user_id="agent")
    mem = _BrokenInsertMemory()

    summary = degrade.replay_queue(mem, user_id="agent")

    assert summary["replayed"] == 0
    assert summary["failed"] == 1
    assert summary["remaining"] == 1
    assert degrade.queued_count() == 1


def test_replay_duplicate_id_acknowledges_prior_insert(temp_queue):
    degrade.queue_write("already inserted before replay retry", user_id="agent")
    mem = _DuplicateInsertMemory()

    summary = degrade.replay_queue(mem, user_id="agent")

    assert summary["replayed"] == 1
    assert summary["replayed_verbatim"] == 1
    assert summary["failed"] == 0
    assert summary["remaining"] == 0
