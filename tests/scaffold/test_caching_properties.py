"""Structural cacheability tests for fidelis.scaffold v0.1.0.

Validates that wrap_system_prompt() output has the byte-stability properties
required for Anthropic prompt caching (and compatible providers).

NO API calls are made — cacheability is validated structurally:
  - determinism (same inputs → same bytes)
  - absence of per-call-varying tokens (timestamps, UUIDs, random hex)
  - stable prefix structure
  - open marker at character 0
  - no control characters that providers may reject in cached content
"""

import re

import pytest

from fidelis.scaffold._core import wrap_system_prompt, SCAFFOLD_OPEN, SCAFFOLD_CLOSE

ALL_QTYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "multi-session",
    "temporal-reasoning",
]

TOP_SCORES = [0.5, 0.6, 0.7, 0.8]


# ---------------------------------------------------------------------------
# a) Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_deterministic_with_top_score(self, qtype):
        """Same (qtype, top_score) → byte-identical output on two calls."""
        first = wrap_system_prompt(qtype, top_score=0.7)
        second = wrap_system_prompt(qtype, top_score=0.7)
        assert first == second, (
            f"wrap_system_prompt('{qtype}', top_score=0.7) returned different "
            f"strings on two calls — caching will never hit."
        )

    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_deterministic_without_top_score(self, qtype):
        """None top_score is also deterministic."""
        first = wrap_system_prompt(qtype, top_score=None)
        second = wrap_system_prompt(qtype, top_score=None)
        assert first == second


# ---------------------------------------------------------------------------
# b) No per-call randomness
# ---------------------------------------------------------------------------

# ISO 8601 datetime fragment: YYYY-MM-DD or YYYY/MM/DD or HH:MM
_TIMESTAMP_RE = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}:\d{2}:\d{2}")
# UUID v4: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
# Random hex strings: 16+ contiguous hex chars not part of normal words
_RANDOM_HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.I)


class TestNoPerCallRandomness:
    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_no_timestamp(self, qtype):
        out = wrap_system_prompt(qtype, top_score=0.65)
        assert not _TIMESTAMP_RE.search(out), (
            f"Timestamp found in scaffold for qtype='{qtype}': "
            f"{_TIMESTAMP_RE.search(out).group()!r}"
        )

    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_no_uuid(self, qtype):
        out = wrap_system_prompt(qtype, top_score=0.65)
        assert not _UUID_RE.search(out), (
            f"UUID found in scaffold for qtype='{qtype}': "
            f"{_UUID_RE.search(out).group()!r}"
        )

    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_no_random_hex(self, qtype):
        out = wrap_system_prompt(qtype, top_score=0.65)
        assert not _RANDOM_HEX_RE.search(out), (
            f"Long hex string found in scaffold for qtype='{qtype}': "
            f"{_RANDOM_HEX_RE.search(out).group()!r}"
        )


# ---------------------------------------------------------------------------
# c) Stable prefix per qtype across top_score values
# ---------------------------------------------------------------------------

def _confidence_line(out: str) -> str:
    """Return the retrieval-quality marker line from scaffold output."""
    for line in out.splitlines():
        if line.startswith("[retrieval-quality:"):
            return line
    return ""


class TestStablePrefix:
    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_only_confidence_line_varies_across_scores(self, qtype):
        """For a given qtype, the scaffold body is identical across all
        top_score values except for the retrieval-quality marker line."""
        outputs = {s: wrap_system_prompt(qtype, top_score=s) for s in TOP_SCORES}

        # Strip the confidence line from each output and compare the remainder
        def strip_confidence(text: str) -> str:
            return "\n".join(
                ln for ln in text.splitlines()
                if not ln.startswith("[retrieval-quality:")
            )

        stripped = {s: strip_confidence(out) for s, out in outputs.items()}
        reference = stripped[TOP_SCORES[0]]
        for score, body in stripped.items():
            assert body == reference, (
                f"qtype='{qtype}', top_score={score}: scaffold body differs "
                f"beyond the confidence line — cache prefix will be broken.\n"
                f"Reference (top_score={TOP_SCORES[0]}):\n{reference}\n\n"
                f"Got (top_score={score}):\n{body}"
            )

    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_confidence_line_is_the_only_diff(self, qtype):
        """Confirm only the confidence line differs between HIGH and MEDIUM score."""
        high = wrap_system_prompt(qtype, top_score=0.75)
        medium = wrap_system_prompt(qtype, top_score=0.55)
        high_lines = high.splitlines()
        medium_lines = medium.splitlines()
        assert len(high_lines) == len(medium_lines), (
            f"qtype='{qtype}': line count differs between HIGH and MEDIUM "
            f"score variants ({len(high_lines)} vs {len(medium_lines)})"
        )
        diffs = [
            (i, h, m)
            for i, (h, m) in enumerate(zip(high_lines, medium_lines))
            if h != m
        ]
        assert len(diffs) == 1, (
            f"qtype='{qtype}': expected exactly 1 differing line (confidence), "
            f"got {len(diffs)}: {diffs}"
        )
        diff_line_idx, h_line, m_line = diffs[0]
        assert h_line.startswith("[retrieval-quality:"), (
            f"Differing line is not a retrieval-quality marker: {h_line!r}"
        )
        assert m_line.startswith("[retrieval-quality:"), (
            f"Differing line is not a retrieval-quality marker: {m_line!r}"
        )


# ---------------------------------------------------------------------------
# d) Cacheable boundary — SCAFFOLD_OPEN at character 0
# ---------------------------------------------------------------------------

class TestCacheableBoundary:
    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_scaffold_open_at_char_zero(self, qtype):
        """System prompt must start with SCAFFOLD_OPEN so providers can cache
        from the very beginning of the prompt."""
        out = wrap_system_prompt(qtype, top_score=0.7)
        assert out.startswith(SCAFFOLD_OPEN), (
            f"qtype='{qtype}': scaffold does not start with SCAFFOLD_OPEN. "
            f"First 60 chars: {out[:60]!r}"
        )
        assert out.index(SCAFFOLD_OPEN) == 0, (
            f"qtype='{qtype}': SCAFFOLD_OPEN found but not at position 0 "
            f"(position {out.index(SCAFFOLD_OPEN)})"
        )

    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_scaffold_close_at_end(self, qtype):
        """SCAFFOLD_CLOSE must be at the end (no trailing content after it)."""
        out = wrap_system_prompt(qtype, top_score=0.7)
        assert out.endswith(SCAFFOLD_CLOSE), (
            f"qtype='{qtype}': scaffold does not end with SCAFFOLD_CLOSE. "
            f"Last 60 chars: {out[-60:]!r}"
        )


# ---------------------------------------------------------------------------
# e) No control characters
# ---------------------------------------------------------------------------

# Ranges that may be rejected by caching providers:
#   \x00-\x08  (C0 controls before tab)
#   \x0b-\x0c  (VT, FF — not tab/newline/CR)
#   \x0e-\x1f  (remaining C0 controls)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class TestNoControlCharacters:
    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_no_control_chars(self, qtype):
        out = wrap_system_prompt(qtype, top_score=0.7)
        match = _CONTROL_CHAR_RE.search(out)
        assert match is None, (
            f"qtype='{qtype}': control character \\x{ord(match.group()):02x} "
            f"found at position {match.start()} — some providers reject "
            f"control characters in cached content."
        )

    @pytest.mark.parametrize("qtype", ALL_QTYPES)
    def test_no_control_chars_none_score(self, qtype):
        out = wrap_system_prompt(qtype, top_score=None)
        match = _CONTROL_CHAR_RE.search(out)
        assert match is None, (
            f"qtype='{qtype}' (top_score=None): control character found."
        )
