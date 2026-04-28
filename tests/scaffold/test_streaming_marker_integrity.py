"""Blocker #5 — Streaming response / scaffold marker integrity.

Verifies that the Fidelis scaffold's bracketed markers survive:
  1. Input-side: system field arrives at the API intact (not fragmented pre-send).
  2. Output-side: if a downstream LLM echoes scaffold markers in a streamed
     assistant response, strip_scaffold / is_scaffolded work correctly on the
     concatenated output.

Uses httpx.MockTransport to fake SSE streams from Anthropic — no real credits.

MockTransport limitation (honest):
  - We cannot test actual server-side streaming chunk delivery timing.
  - We cannot test whether Anthropic's tokeniser breaks our marker mid-token
    at the server boundary (would require real streaming credits).
  - What we CAN verify: the SDK assembles SSE chunks into a final text correctly,
    and our post-stream scaffold utilities handle fragmented-but-concatenated text.
"""

from __future__ import annotations

import json

import httpx

import anthropic

from fidelis.scaffold import (
    SCAFFOLD_CLOSE,
    SCAFFOLD_OPEN,
    is_scaffolded,
    strip_scaffold,
    wrap_system_prompt,
)

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse_line(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _make_streaming_sse_body(text_chunks: list[str]) -> bytes:
    """
    Build a minimal Anthropic SSE stream that delivers text in chunks.
    Intentionally splits scaffold markers across chunk boundaries to
    prove the client-side concat + strip works.
    """
    parts: list[bytes] = []

    parts.append(
        _sse_line(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock_stream_001",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-5-haiku-20241022",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 30,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    },
                },
            },
        )
    )

    parts.append(
        _sse_line(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
    )

    for chunk in text_chunks:
        parts.append(
            _sse_line(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk},
                },
            )
        )

    parts.append(
        _sse_line(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
    )

    parts.append(
        _sse_line(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": sum(len(c) for c in text_chunks) // 4 + 1},
            },
        )
    )

    parts.append(
        _sse_line(
            "message_stop",
            {"type": "message_stop"},
        )
    )

    return b"".join(parts)


def _stream_transport(text_chunks: list[str]) -> httpx.MockTransport:
    """MockTransport that returns an SSE stream with the given text chunks."""
    sse_body = _make_streaming_sse_body(text_chunks)

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "transfer-encoding": "chunked",
            },
            content=sse_body,
        )

    return httpx.MockTransport(_handler)


def _non_stream_transport(captured: list) -> httpx.MockTransport:
    """MockTransport for non-streaming calls; captures the request body."""

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        body = {
            "id": "msg_mock_001",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "mocked"}],
            "model": "claude-3-5-haiku-20241022",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 2,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(body).encode(),
        )

    return httpx.MockTransport(_handler)


def _client_with_transport(transport: httpx.MockTransport) -> anthropic.Anthropic:
    http = httpx.Client(transport=transport)
    return anthropic.Anthropic(api_key="test-key-mock", http_client=http)


# ---------------------------------------------------------------------------
# Test 1 — Input-side: system field arrives intact at the API
# ---------------------------------------------------------------------------


class TestInputSideSystemIntegrity:
    """Blocker #5a — scaffold system text is sent to the API WITHOUT fragmentation."""

    def test_system_field_arrives_intact_for_all_qtypes(self):
        """
        For each qtype, wrap_system_prompt produces the scaffold.
        Assert the exact text lands in the HTTP request body — no truncation,
        no mid-marker splits, no encoding damage.
        """
        for qtype in [
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "knowledge-update",
            "multi-session",
            "temporal-reasoning",
        ]:
            captured: list[dict] = []
            client = _client_with_transport(_non_stream_transport(captured))
            scaffold_text = wrap_system_prompt(qtype)

            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=64,
                system=scaffold_text,
                messages=[{"role": "user", "content": "Question."}],
            )

            assert len(captured) == 1, f"No request captured for qtype={qtype!r}"
            wire_system = captured[0].get("system", "")
            assert wire_system == scaffold_text, (
                f"qtype={qtype!r}: scaffold text mutated in transit.\n"
                f"Expected len={len(scaffold_text)}, got len={len(wire_system)}"
            )

    def test_scaffold_open_marker_present_and_intact_in_wire_body(self):
        """The SCAFFOLD_OPEN marker must appear verbatim in the wire body."""
        captured: list[dict] = []
        client = _client_with_transport(_non_stream_transport(captured))
        scaffold_text = wrap_system_prompt("multi-session")

        client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            system=scaffold_text,
            messages=[{"role": "user", "content": "Q."}],
        )

        wire_system = captured[0]["system"]
        assert SCAFFOLD_OPEN in wire_system, (
            f"SCAFFOLD_OPEN marker missing from wire body. "
            f"Got: {wire_system[:120]!r}"
        )
        assert SCAFFOLD_CLOSE in wire_system, (
            "SCAFFOLD_CLOSE marker missing from wire body."
        )

    def test_scaffold_open_marker_at_wire_body_start(self):
        """Scaffold must BEGIN with the open marker — caching boundary is position 0."""
        captured: list[dict] = []
        client = _client_with_transport(_non_stream_transport(captured))
        scaffold_text = wrap_system_prompt("temporal-reasoning")

        client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=64,
            system=scaffold_text,
            messages=[{"role": "user", "content": "When?"}],
        )

        wire_system = captured[0]["system"]
        assert wire_system.startswith(SCAFFOLD_OPEN), (
            f"Wire body system does not start with SCAFFOLD_OPEN. "
            f"Start: {wire_system[:60]!r}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Output-side: streamed scaffold echo reassembly
# ---------------------------------------------------------------------------


class TestOutputSideMarkerReassembly:
    """
    Blocker #5b — if the LLM echoes scaffold markers in its reply,
    strip_scaffold / is_scaffolded work correctly after stream concatenation.

    The mock stream deliberately fragments the SCAFFOLD markers across chunk
    boundaries (e.g. "[FIDELIS-" in one chunk, "SCAFFOLD-v0.1.0]" in the next)
    to simulate what real streaming can produce at token boundaries.
    """

    def _do_stream(self, chunks: list[str]) -> str:
        """Run a streaming call with the given mock SSE chunks; return full text."""
        transport = _stream_transport(chunks)
        client = _client_with_transport(transport)

        with client.messages.stream(
            model="claude-3-5-haiku-20241022",
            max_tokens=128,
            system="plain system",
            messages=[{"role": "user", "content": "Echo the scaffold."}],
        ) as stream:
            return stream.get_final_message().content[0].text

    def test_fragmented_open_marker_reassembled_by_sdk(self):
        """
        SSE delivers '[FIDELIS-' in chunk1 and 'SCAFFOLD-v0.1.0] some answer'
        in chunk2. After SDK reassembly, the full marker is present.
        """
        chunks = [
            "Here is the scaffold: [FIDELIS-",
            "SCAFFOLD-v0.1.0] some answer text [/FIDELIS-",
            "SCAFFOLD-v0.1.0]",
        ]
        full_text = self._do_stream(chunks)

        assert SCAFFOLD_OPEN in full_text, (
            f"SCAFFOLD_OPEN not found in reassembled stream text: {full_text!r}"
        )
        assert SCAFFOLD_CLOSE in full_text

    def test_is_scaffolded_detects_echoed_marker_after_stream(self):
        """is_scaffolded() returns True on stream output that echoes scaffold markers."""
        chunks = [
            f"Prefix. {SCAFFOLD_OPEN}\n",
            "some scaffolded content\n",
            f"{SCAFFOLD_CLOSE}",
        ]
        full_text = self._do_stream(chunks)
        assert is_scaffolded(full_text), (
            f"is_scaffolded returned False on echoed-marker stream output: {full_text!r}"
        )

    def test_strip_scaffold_removes_echoed_markers_from_stream_output(self):
        """strip_scaffold() cleanly removes scaffold section from streamed echo."""
        scaffold_section = f"{SCAFFOLD_OPEN}\nsome scaffolded content\n{SCAFFOLD_CLOSE}"
        prefix = "Before scaffold. "
        suffix = " After scaffold."
        chunks = [prefix, scaffold_section, suffix]
        full_text = self._do_stream(chunks)

        stripped = strip_scaffold(full_text)
        assert SCAFFOLD_OPEN not in stripped, "Open marker survived strip_scaffold"
        assert SCAFFOLD_CLOSE not in stripped, "Close marker survived strip_scaffold"
        assert "Before scaffold." in stripped
        assert "After scaffold." in stripped

    def test_non_scaffolded_stream_unchanged_by_strip(self):
        """strip_scaffold is a no-op on streamed text that contains no scaffold markers."""
        chunks = ["Answer: ", "The user said ", "coffee is great."]
        full_text = self._do_stream(chunks)

        stripped = strip_scaffold(full_text)
        assert stripped == full_text or stripped == full_text.strip(), (
            "strip_scaffold mutated clean (non-scaffolded) stream output"
        )

    def test_scaffold_open_at_chunk_boundary_not_double_detected(self):
        """
        Deliver SCAFFOLD_OPEN split across three chunks — e.g. '[', 'FIDELIS-SCAFFOLD-',
        'v0.1.0]'. After concat, is_scaffolded and strip_scaffold treat it as ONE marker.
        """
        # Build chunks that fragment the open marker at the bracket boundaries
        open_parts = SCAFFOLD_OPEN  # "[FIDELIS-SCAFFOLD-v0.1.0]"
        chunk1 = open_parts[:1]         # "["
        chunk2 = open_parts[1:17]       # "FIDELIS-SCAFFOLD-"
        chunk3 = open_parts[17:]        # "v0.1.0]"

        close_part = SCAFFOLD_CLOSE

        chunks = [
            chunk1,
            chunk2,
            chunk3,
            "\nThis is the scaffold body.\n",
            close_part,
        ]
        full_text = self._do_stream(chunks)

        # After SDK reassembly: exactly one open marker
        assert full_text.count(SCAFFOLD_OPEN) == 1, (
            f"Expected exactly 1 SCAFFOLD_OPEN after reassembly, "
            f"got {full_text.count(SCAFFOLD_OPEN)}: {full_text!r}"
        )

        # strip_scaffold removes it cleanly
        stripped = strip_scaffold(full_text)
        assert not is_scaffolded(stripped), (
            "is_scaffolded still True after strip on single-marker stream text"
        )

    def test_idempotent_strip_on_streamed_output(self):
        """strip_scaffold(strip_scaffold(x)) == strip_scaffold(x) for stream output."""
        chunks = [
            f"{SCAFFOLD_OPEN}\nScaffold body.\n{SCAFFOLD_CLOSE}",
            " Trailing answer.",
        ]
        full_text = self._do_stream(chunks)

        once = strip_scaffold(full_text)
        twice = strip_scaffold(once)
        assert once == twice, "strip_scaffold is not idempotent on streamed output"
