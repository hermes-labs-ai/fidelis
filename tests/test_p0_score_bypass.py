"""Regression test for the v0.0.9 P0 fix.

mem0 2.0.0's `Memory.search` runs results through `score_and_rank`, which
in 2.0.0 returned `score=1.0` for every row. The matching memory ended up
indistinguishable from unrelated memories, silently breaking /query.

The fix bypasses `Memory.search` entirely and calls `vector_store.search`
directly (cosine distance), then converts to similarity via `1 - d/2`.

This test pins the bypass with a synthetic memory whose `Memory.search`
returns the buggy uniform-1.0 results, while `vector_store.search` returns
proper distance-ordered results. The /query handler must use the latter.
"""
from __future__ import annotations

import io
import json
from unittest.mock import MagicMock

from fidelis.server import make_handler


class _BuggyScoreMemory:
    """Memory mock that mimics mem0 2.0.0's broken score_and_rank.

    .search() returns score=1.0 for every result (the bug).
    .vector_store.search() returns proper cosine distances (the workaround).
    A correct /query must NOT call .search() — it must call vector_store
    directly so it gets the real distances.
    """

    def __init__(self):
        # Track which path the handler took
        self.search_called = False
        self.vector_store = MagicMock()

        # Configure the buggy path
        def buggy_search(*args, **kwargs):
            self.search_called = True
            return {"results": [
                {"memory": "unrelated A", "score": 1.0},
                {"memory": "the actual match", "score": 1.0},
                {"memory": "unrelated B", "score": 1.0},
            ]}
        self.search = buggy_search

        # Configure the correct path
        class _Hit:
            def __init__(self, text, distance):
                self.payload = {"data": text}
                self.score = distance  # cosine distance ∈ [0, 2]

        self.vector_store.search.return_value = [
            _Hit("the actual match", 0.2),  # high similarity → score 0.9
            _Hit("unrelated A", 1.6),       # low similarity → score 0.2
            _Hit("unrelated B", 1.8),       # lowest → score 0.1
        ]
        self.vector_store.col_info.return_value.count.return_value = 42
        self.vector_store.col.count.return_value = 42

        self.embedding_model = MagicMock()
        self.embedding_model.embed.return_value = [0.0] * 8


def _post_to_handler(handler_cls, path: str, body: dict) -> dict:
    payload = json.dumps(body).encode()

    class _Capture:
        def __init__(self):
            self.written: list[bytes] = []

        def write(self, data):
            self.written.append(data)

        def flush(self):
            pass

    wfile = _Capture()

    handler = handler_cls.__new__(handler_cls)
    handler.wfile = wfile
    handler.rfile = io.BytesIO(payload)
    handler.headers = {"Content-Length": str(len(payload)), "Content-Type": "application/json"}
    handler.path = path
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = "POST"
    handler.send_response = lambda *a, **kw: None
    handler.send_header = lambda *a, **kw: None
    handler.end_headers = lambda: None
    handler.do_POST()

    return json.loads(b"".join(wfile.written))


def test_query_bypasses_broken_score_and_rank():
    """The /query handler must NOT call mem0.search (which is broken in
    mem0 2.0.0); it must call vector_store.search directly.
    """
    mem = _BuggyScoreMemory()
    cfg = {"user_id": "agent", "vocab_map": {}}
    handler_cls = make_handler(mem, cfg)

    response = _post_to_handler(handler_cls, "/query", {"text": "actual match", "limit": 3})

    # The bypass: mem.search must NOT be called
    assert mem.search_called is False, (
        "/query called mem0.Memory.search — the broken score_and_rank wrapper. "
        "It must call vector_store.search directly (the v0.0.9 P0 fix)."
    )
    # vector_store.search must have been called
    assert mem.vector_store.search.called, "/query did not call vector_store.search"

    # The matching memory must come back ranked correctly (score > the others)
    memories = response.get("memories", [])
    assert memories, f"no memories returned: {response}"
    top = memories[0]
    assert "actual match" in top["text"], f"wrong top result: {top}"

    # Scores must be properly ranked (NOT all 1.0 — that was the bug)
    scores = [m["score"] for m in memories]
    assert len(set(scores)) > 1, (
        f"all scores identical {scores} — the v0.0.8 score_and_rank=1.0 bug regressed"
    )
    # Scores must be in [0, 1] (similarity, not distance)
    assert all(0 <= s <= 1 for s in scores), f"scores out of [0,1]: {scores}"
    # The matching memory must score highest
    assert top["score"] == max(scores), "ranking inverted"


def test_query_distance_to_similarity_conversion():
    """Cosine distance d ∈ [0, 2] must convert to similarity s = 1 - d/2 ∈ [0, 1]."""
    mem = _BuggyScoreMemory()
    cfg = {"user_id": "agent", "vocab_map": {}}
    handler_cls = make_handler(mem, cfg)

    response = _post_to_handler(handler_cls, "/query", {"text": "anything", "limit": 3})

    memories = response.get("memories", [])
    # _BuggyScoreMemory returns distances 0.2, 1.6, 1.8
    # → similarities 0.9, 0.2, 0.1 (rounded to 3dp)
    expected = [0.9, 0.2, 0.1]
    actual = [m["score"] for m in memories]
    for got, want in zip(actual, expected):
        assert abs(got - want) < 0.01, f"expected ~{want}, got {got}"
