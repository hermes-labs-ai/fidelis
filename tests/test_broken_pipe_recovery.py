"""
Tests for BrokenPipe recovery (Patch 1) and RRF decompose timeout fallback (Patch 2).

All tests are pure-Python — no Ollama, no ChromaDB, no network required.

Scenarios:
  1. Client disconnect mid-write → server doesn't crash; subsequent requests served normally.
  2. Decompose timeout (mocked) → response has degraded:true + vector-only results.
  3. Concurrent requests during a slow decompose → each handled independently.
  4. /health stays ok throughout simulated disconnect events.
"""

from __future__ import annotations

import io
import json
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest

from fidelis.server import make_handler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(**overrides) -> dict:
    base = {
        "user_id": "test_agent",
        "port": 0,
        "query_threshold": 250.0,
        "recall_limit": 10,
    }
    base.update(overrides)
    return base


def _fake_memory(results=None):
    """Return a minimal memory mock that responds to .search() and .vector_store."""
    mem = MagicMock()
    search_results = results if results is not None else []
    mem.search.return_value = {"results": search_results}
    # mem0 2.x vector store: col_info() returns the collection; .count() is O(1).
    mem.vector_store.col_info.return_value.count.return_value = 42
    # Pre-2.x compat path the health endpoint also tries.
    mem.vector_store.col.count.return_value = 42

    # /recall fallback now bypasses mem0.search and calls vector_store.search
    # directly, returning objects with .payload and .score (chroma shape).
    class _Hit:
        def __init__(self, text, dist):
            self.payload = {"data": text}
            self.score = dist  # cosine distance ∈ [0, 2]

    mem.vector_store.search.return_value = [
        _Hit(r.get("memory", ""), r.get("score", 0.0)) for r in search_results
    ]
    mem.embedding_model.embed.return_value = [0.0] * 8
    return mem


def _make_request_bytes(method: str, path: str, body: dict | None = None) -> bytes:
    """Build a raw HTTP/1.1 request."""
    if body is not None:
        payload = json.dumps(body).encode()
        content_headers = (
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
        )
    else:
        payload = b""
        content_headers = ""
    request = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: 127.0.0.1\r\n"
        f"{content_headers}"
        f"\r\n"
    ).encode() + payload
    return request


# ---------------------------------------------------------------------------
# 1. Client disconnect mid-write — server absorbs BrokenPipe, keeps serving
# ---------------------------------------------------------------------------

class _BrokenWriteFile:
    """File-like object that raises BrokenPipeError on write."""

    def __init__(self):
        self.written = []
        self._raise_on_write = False

    def write(self, data):
        if self._raise_on_write:
            raise BrokenPipeError("simulated client disconnect")
        self.written.append(data)
        return len(data)

    def flush(self):
        pass


def _invoke_handler_method(handler_cls, method_name: str, path: str, body: dict | None = None) -> tuple[list[bytes], bool]:
    """
    Directly invoke a handler method (do_GET or do_POST) without a real socket.
    Returns (bytes_written, raised_exception).
    """
    wfile = _BrokenWriteFile()
    rfile_content = b""
    if body is not None:
        rfile_content = json.dumps(body).encode()

    handler = handler_cls.__new__(handler_cls)
    handler.wfile = wfile
    handler.rfile = io.BytesIO(rfile_content)
    handler.headers = {"Content-Length": str(len(rfile_content)), "Content-Type": "application/json"}
    handler.path = path
    handler.requestline = f"{method_name} {path} HTTP/1.1"
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = method_name

    # Patch send_response / send_header / end_headers (pre-write socket ops)
    handler.send_response = lambda *a, **kw: None
    handler.send_header = lambda *a, **kw: None
    handler.end_headers = lambda: None

    raised = False
    try:
        getattr(handler, f"do_{method_name}")()
    except (BrokenPipeError, ConnectionResetError):
        raised = True

    return wfile.written, raised


def test_broken_pipe_in_write_does_not_propagate():
    """BrokenPipeError raised in wfile.write must be absorbed — no exception escapes the handler."""
    mem = _fake_memory()
    cfg = _make_cfg()
    HandlerCls = make_handler(mem, cfg)

    wfile = _BrokenWriteFile()
    wfile._raise_on_write = True  # raise immediately on any write
    rfile = io.BytesIO(b"")

    handler = HandlerCls.__new__(HandlerCls)
    handler.wfile = wfile
    handler.rfile = rfile
    handler.headers = {"Content-Length": "0"}
    handler.path = "/health"
    handler.requestline = "GET /health HTTP/1.1"
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = "GET"
    handler.send_response = lambda *a, **kw: None
    handler.send_header = lambda *a, **kw: None
    handler.end_headers = lambda: None

    # Must not raise
    handler.do_GET()


def test_server_handles_subsequent_requests_after_broken_pipe():
    """After a client disconnect, the server still handles new requests normally."""
    mem = _fake_memory()
    cfg = _make_cfg()
    HandlerCls = make_handler(mem, cfg)

    # First request: broken pipe on write
    wfile_broken = _BrokenWriteFile()
    wfile_broken._raise_on_write = True

    handler1 = HandlerCls.__new__(HandlerCls)
    handler1.wfile = wfile_broken
    handler1.rfile = io.BytesIO(b"")
    handler1.headers = {"Content-Length": "0"}
    handler1.path = "/health"
    handler1.requestline = "GET /health HTTP/1.1"
    handler1.server = MagicMock()
    handler1.client_address = ("127.0.0.1", 11111)
    handler1.command = "GET"
    handler1.send_response = lambda *a, **kw: None
    handler1.send_header = lambda *a, **kw: None
    handler1.end_headers = lambda: None
    handler1.do_GET()  # must not raise

    # Second request: normal client, should get a real JSON response
    wfile_ok = _BrokenWriteFile()

    handler2 = HandlerCls.__new__(HandlerCls)
    handler2.wfile = wfile_ok
    handler2.rfile = io.BytesIO(b"")
    handler2.headers = {"Content-Length": "0"}
    handler2.path = "/health"
    handler2.requestline = "GET /health HTTP/1.1"
    handler2.server = MagicMock()
    handler2.client_address = ("127.0.0.1", 22222)
    handler2.command = "GET"
    handler2.send_response = lambda *a, **kw: None
    handler2.send_header = lambda *a, **kw: None
    handler2.end_headers = lambda: None
    handler2.do_GET()

    assert len(wfile_ok.written) > 0
    response = json.loads(b"".join(wfile_ok.written))
    assert response["status"] == "ok"


# ---------------------------------------------------------------------------
# 2. Decompose timeout → degraded:true + vector-only results
# ---------------------------------------------------------------------------

def test_decompose_timeout_returns_degraded_flag():
    """When do_recall times out, the response must contain degraded:true and fallback method."""
    fallback_results = [{"memory": "fallback memory text", "score": 100.0}]
    mem = _fake_memory(results=fallback_results)
    # Set an impossibly short timeout to force the fallback
    cfg = _make_cfg()

    with patch.dict("os.environ", {"FIDELIS_DECOMPOSE_TIMEOUT_SECS": "0.001"}):
        HandlerCls = make_handler(mem, cfg)

    body = json.dumps({"text": "what is the recall score"}).encode()
    wfile = _BrokenWriteFile()

    handler = HandlerCls.__new__(HandlerCls)
    handler.wfile = wfile
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    handler.path = "/recall"
    handler.requestline = "POST /recall HTTP/1.1"
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = "POST"
    handler.send_response = lambda *a, **kw: None
    handler.send_header = lambda *a, **kw: None
    handler.end_headers = lambda: None
    handler.do_POST()

    assert len(wfile.written) > 0
    response = json.loads(b"".join(wfile.written))
    assert response.get("degraded") is True
    assert response.get("method") == "vector-only-fallback"
    assert "memories" in response


def test_decompose_timeout_vector_only_results_returned():
    """Fallback path returns memories from the single-query vector search."""
    fallback_results = [
        {"memory": "fact about deployment", "score": 80.0},
        {"memory": "fact about configuration", "score": 120.0},
    ]
    mem = _fake_memory(results=fallback_results)
    cfg = _make_cfg()

    with patch.dict("os.environ", {"FIDELIS_DECOMPOSE_TIMEOUT_SECS": "0.001"}):
        HandlerCls = make_handler(mem, cfg)

    body = json.dumps({"text": "deployment facts"}).encode()
    wfile = _BrokenWriteFile()

    handler = HandlerCls.__new__(HandlerCls)
    handler.wfile = wfile
    handler.rfile = io.BytesIO(body)
    handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    handler.path = "/recall"
    handler.requestline = "POST /recall HTTP/1.1"
    handler.server = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.command = "POST"
    handler.send_response = lambda *a, **kw: None
    handler.send_header = lambda *a, **kw: None
    handler.end_headers = lambda: None
    handler.do_POST()

    response = json.loads(b"".join(wfile.written))
    assert len(response["memories"]) == 2
    texts = [m["text"] for m in response["memories"]]
    assert "fact about deployment" in texts
    assert "fact about configuration" in texts


# ---------------------------------------------------------------------------
# 3. Concurrent requests during slow decompose — each handled independently
# ---------------------------------------------------------------------------

def test_concurrent_requests_handled_independently():
    """
    Multiple /recall requests hitting simultaneously should each complete
    independently — no shared mutable state causes cross-contamination.
    """
    results_by_query = {
        "alpha query": [{"memory": "alpha result", "score": 50.0}],
        "beta query": [{"memory": "beta result", "score": 50.0}],
        "gamma query": [{"memory": "gamma result", "score": 50.0}],
    }

    def _smart_search(text, filters=None, top_k=10):
        for key, val in results_by_query.items():
            if key in text:
                return {"results": val}
        return {"results": []}

    mem = MagicMock()
    mem.search.side_effect = _smart_search
    mem.vector_store.col.count.return_value = 99

    cfg = _make_cfg()
    # Use very short timeout so recall_b falls back quickly without Ollama
    with patch.dict("os.environ", {"FIDELIS_DECOMPOSE_TIMEOUT_SECS": "0.001"}):
        HandlerCls = make_handler(mem, cfg)

    responses = {}
    errors = []

    def _do_request(query_key):
        body = json.dumps({"text": query_key}).encode()
        wfile = _BrokenWriteFile()
        handler = HandlerCls.__new__(HandlerCls)
        handler.wfile = wfile
        handler.rfile = io.BytesIO(body)
        handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
        handler.path = "/recall"
        handler.requestline = "POST /recall HTTP/1.1"
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 9000)
        handler.command = "POST"
        handler.send_response = lambda *a, **kw: None
        handler.send_header = lambda *a, **kw: None
        handler.end_headers = lambda: None
        try:
            handler.do_POST()
            responses[query_key] = json.loads(b"".join(wfile.written))
        except Exception as exc:
            errors.append((query_key, str(exc)))

    threads = [threading.Thread(target=_do_request, args=(k,)) for k in results_by_query]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Unexpected errors in concurrent handlers: {errors}"
    for key in results_by_query:
        assert key in responses, f"No response captured for query '{key}'"
        # Each response must be structurally valid
        assert "memories" in responses[key]
        assert "method" in responses[key]


# ---------------------------------------------------------------------------
# 4. /health stays ok throughout simulated disconnect events
# ---------------------------------------------------------------------------

def test_health_ok_after_disconnect_events():
    """/health returns ok even after multiple BrokenPipe events have occurred."""
    mem = _fake_memory()
    cfg = _make_cfg()
    HandlerCls = make_handler(mem, cfg)

    def _health_request():
        wfile = _BrokenWriteFile()
        handler = HandlerCls.__new__(HandlerCls)
        handler.wfile = wfile
        handler.rfile = io.BytesIO(b"")
        handler.headers = {"Content-Length": "0"}
        handler.path = "/health"
        handler.requestline = "GET /health HTTP/1.1"
        handler.server = MagicMock()
        handler.client_address = ("127.0.0.1", 9999)
        handler.command = "GET"
        handler.send_response = lambda *a, **kw: None
        handler.send_header = lambda *a, **kw: None
        handler.end_headers = lambda: None
        handler.do_GET()
        return wfile.written

    # Simulate a broken-pipe write before health check
    broken_wfile = _BrokenWriteFile()
    broken_wfile._raise_on_write = True
    broken_handler = HandlerCls.__new__(HandlerCls)
    broken_handler.wfile = broken_wfile
    broken_handler.rfile = io.BytesIO(b"")
    broken_handler.headers = {"Content-Length": "0"}
    broken_handler.path = "/health"
    broken_handler.requestline = "GET /health HTTP/1.1"
    broken_handler.server = MagicMock()
    broken_handler.client_address = ("127.0.0.1", 8888)
    broken_handler.command = "GET"
    broken_handler.send_response = lambda *a, **kw: None
    broken_handler.send_header = lambda *a, **kw: None
    broken_handler.end_headers = lambda: None
    broken_handler.do_GET()  # must not raise

    # Now a clean /health should still work
    written = _health_request()
    assert len(written) > 0
    response = json.loads(b"".join(written))
    assert response["status"] == "ok"
    assert response["count"] == 42
