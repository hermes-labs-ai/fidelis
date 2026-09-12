"""Negative test for the `user_id` namespace boundary.

Fidelis is single-namespace by design: `user_id` is a *storage namespace*, not
an authenticated identity. The server binds one `user_id` per process at
handler-construction time (`server.make_handler`) and never accepts a `user_id`
from a request body, so a client cannot select or spoof a namespace over the
wire. There is no authentication layer, and `README.md` / `agents.md` both say
so ("fidelis is single-namespace"; multi-namespace isolation and custom
authentication are named as things the OSS path "does not yet cover").

What this file pins down is the weaker — but real, and previously untested —
property that the namespace boundary is actually *enforced* at read time:
records written under one explicit `user_id` are not retrievable under another.

Every other test in the suite runs in a single namespace ("agent"), so nothing
was covering the cross-namespace case. This test uses a real Chroma collection
(`chromadb` is a base dependency, so it runs in CI) with a stub embedder, so it
needs no Ollama and no LLM.

Both records are inserted with the *same* vector, so vector similarity cannot
be what separates them — only the `filters={"user_id": ...}` clause can. Drop
that filter from `server.py` and this test fails.
"""
from __future__ import annotations

import io
import json
import uuid
from unittest.mock import MagicMock

import chromadb
import pytest
from mem0.vector_stores.chroma import ChromaDB

from fidelis.server import make_handler

_VECTOR = [1.0, 0.0, 0.0, 0.0]


class _StubEmbedder:
    """Deterministic stand-in for the Ollama embedder."""

    def embed(self, text, memory_action=None):
        del text, memory_action
        return list(_VECTOR)


class _Memory:
    """Minimal stand-in for mem0.Memory exposing what /query actually uses."""

    def __init__(self, vector_store):
        self.embedding_model = _StubEmbedder()
        self.vector_store = vector_store


def _post_to_handler(handler_cls, path, payload_dict):
    payload = json.dumps(payload_dict).encode()

    class _Capture(io.BytesIO):
        def __init__(self):
            super().__init__()
            self.written = []

        def write(self, b):
            self.written.append(b)

        def flush(self):
            pass

    handler = handler_cls.__new__(handler_cls)
    handler.wfile = _Capture()
    handler.rfile = io.BytesIO(payload)
    handler.headers = {
        "Content-Length": str(len(payload)),
        "Content-Type": "application/json",
    }
    handler.path = path
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = "POST"
    handler.send_response = lambda *args, **kwargs: None
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda: None
    handler.do_POST()
    return json.loads(b"".join(handler.wfile.written))


@pytest.fixture
def shared_store():
    """One collection holding records from two namespaces, as on disk."""
    # EphemeralClient instances share one in-process System, so a fixed
    # collection name would leak between tests. Unique name keeps it hermetic.
    store = ChromaDB(
        collection_name=f"fidelis_namespace_test_{uuid.uuid4().hex}",
        client=chromadb.EphemeralClient(),
    )
    store.insert(
        vectors=[list(_VECTOR), list(_VECTOR)],
        payloads=[
            {"data": "alpha namespace secret", "user_id": "alpha"},
            {"data": "beta namespace secret", "user_id": "beta"},
        ],
        ids=["rec-alpha", "rec-beta"],
    )
    return store


def test_both_namespaces_share_one_collection(shared_store):
    """Control: without a filter both records come back from the same collection.

    This is what makes the isolation assertions below non-vacuous — the records
    are genuinely co-resident, and identical vectors mean similarity ranking
    cannot be doing the separating.
    """
    raw = shared_store.search(
        query="namespace secret", vectors=[list(_VECTOR)], top_k=10, filters=None
    )
    texts = {(r.payload or {}).get("data") for r in raw}
    assert texts == {"alpha namespace secret", "beta namespace secret"}


def test_records_under_one_user_id_are_not_readable_under_another(shared_store):
    """A server bound to "alpha" must never return "beta"'s record, and vice versa."""
    memory = _Memory(shared_store)

    alpha = _post_to_handler(
        make_handler(memory, {"user_id": "alpha", "vocab_map": {}}),
        "/query",
        {"text": "namespace secret", "limit": 10},
    )
    alpha_texts = [m["text"] for m in alpha["memories"]]
    assert alpha_texts == ["alpha namespace secret"]
    assert "beta namespace secret" not in alpha_texts

    beta = _post_to_handler(
        make_handler(memory, {"user_id": "beta", "vocab_map": {}}),
        "/query",
        {"text": "namespace secret", "limit": 10},
    )
    beta_texts = [m["text"] for m in beta["memories"]]
    assert beta_texts == ["beta namespace secret"]
    assert "alpha namespace secret" not in beta_texts


def test_unknown_user_id_reads_empty(shared_store):
    """A namespace with nothing written to it reads empty, not "everything"."""
    memory = _Memory(shared_store)
    result = _post_to_handler(
        make_handler(memory, {"user_id": "gamma", "vocab_map": {}}),
        "/query",
        {"text": "namespace secret", "limit": 10},
    )
    assert result["memories"] == []
