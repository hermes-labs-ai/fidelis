"""Contract tests for top_score input handling in wrap_system_prompt.

Contract (Option A — clamp):
  - Values in [0, 1]       → used as-is.
  - Values outside [0, 1]  → clamped silently (no exception).
  - nan / inf / -inf       → treated as None → "[retrieval-quality: unknown]".
  - int 0 / int 1          → accepted (coerced to float 0.0 / 1.0).
  - bool True / False      → accepted (coerced to float 1.0 / 0.0).
  - Every input produces a valid scaffold (preflight passes).
"""

from __future__ import annotations

import pytest

from fidelis.scaffold import is_scaffolded, preflight, wrap_system_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scaffold(score):
    return wrap_system_prompt("single-session-user", top_score=score)


def _marker(score: object) -> str:
    """Extract the retrieval-quality marker line from the scaffold."""
    s = _scaffold(score)
    for line in s.splitlines():
        if "[retrieval-quality:" in line:
            return line.strip()
    return ""


# ---------------------------------------------------------------------------
# Parametrized: every input must produce a valid scaffold (preflight passes)
# ---------------------------------------------------------------------------

VALID_INPUTS = [
    -1.0,
    -0.1,
    0.0,
    0.5,
    1.0,
    1.01,
    1.5,
    100.0,
    float("nan"),
    float("inf"),
    float("-inf"),
    0,       # int
    1,       # int
    True,    # bool (subclass of int; coerces to 1.0)
    False,   # bool (subclass of int; coerces to 0.0)
]


@pytest.mark.parametrize("score", VALID_INPUTS)
def test_every_input_produces_valid_scaffold(score):
    s = _scaffold(score)
    assert is_scaffolded(s), f"No scaffold marker for top_score={score!r}"
    result = preflight(s)
    assert result.passed, f"Preflight failed for top_score={score!r}: {result.failures}"


# ---------------------------------------------------------------------------
# nan / inf / -inf → unknown marker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_produces_unknown(score):
    marker = _marker(score)
    assert "unknown" in marker.lower(), (
        f"Expected 'unknown' for top_score={score!r}, got: {marker!r}"
    )


# ---------------------------------------------------------------------------
# Out-of-range values → clamped → HIGH or LOW
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score", [-1.0, -0.1])
def test_negative_clamped_to_zero_produces_low(score):
    # Clamped to 0.0 → LOW band
    marker = _marker(score)
    assert "LOW" in marker, (
        f"Expected LOW for top_score={score!r} (clamped to 0.0), got: {marker!r}"
    )


@pytest.mark.parametrize("score", [1.01, 1.5, 100.0])
def test_above_one_clamped_to_one_produces_high(score):
    # Clamped to 1.0 → HIGH band
    marker = _marker(score)
    assert "HIGH" in marker, (
        f"Expected HIGH for top_score={score!r} (clamped to 1.0), got: {marker!r}"
    )


# ---------------------------------------------------------------------------
# In-range sanity checks
# ---------------------------------------------------------------------------

def test_zero_float_produces_low():
    assert "LOW" in _marker(0.0)


def test_half_float_produces_medium():
    assert "MEDIUM" in _marker(0.5)


def test_one_float_produces_high():
    assert "HIGH" in _marker(1.0)


# ---------------------------------------------------------------------------
# int 0 / int 1 are accepted (not just float)
# ---------------------------------------------------------------------------

def test_int_zero_produces_low():
    marker = _marker(0)
    assert "LOW" in marker, f"Expected LOW for int 0, got: {marker!r}"


def test_int_one_produces_high():
    marker = _marker(1)
    assert "HIGH" in marker, f"Expected HIGH for int 1, got: {marker!r}"


# ---------------------------------------------------------------------------
# bool True / False coerce correctly
# ---------------------------------------------------------------------------

def test_bool_true_produces_high():
    marker = _marker(True)
    assert "HIGH" in marker, f"Expected HIGH for True, got: {marker!r}"


def test_bool_false_produces_low():
    marker = _marker(False)
    assert "LOW" in marker, f"Expected LOW for False, got: {marker!r}"


# ---------------------------------------------------------------------------
# No exceptions raised for any input in VALID_INPUTS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score", VALID_INPUTS)
def test_no_exception_raised(score):
    try:
        _scaffold(score)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"wrap_system_prompt raised {type(exc).__name__} for top_score={score!r}: {exc}"
        ) from exc
