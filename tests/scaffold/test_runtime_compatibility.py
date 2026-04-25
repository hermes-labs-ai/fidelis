"""Runtime compatibility tests for fidelis.scaffold v0.1.0.

Covers: async-safety, streaming compatibility, no global state,
memory safety (no-exception bound), and error-path idempotency.
"""

from __future__ import annotations

import asyncio
import random
import string

import pytest

from fidelis.scaffold._core import wrap_system_prompt
from fidelis.scaffold.preflight import preflight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_QTYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "multi-session",
    "temporal-reasoning",
    "unknown-qtype-fallback",
]

_SCORES = [None, 0.0, 0.3, 0.5, 0.7, 0.99, 1.0]


# ---------------------------------------------------------------------------
# a) Async-safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wrap_system_prompt_callable_from_coroutine():
    """wrap_system_prompt works correctly inside an asyncio coroutine."""
    result = wrap_system_prompt("single-session-user", top_score=0.8)
    assert isinstance(result, str)
    assert "[FIDELIS-SCAFFOLD-" in result


@pytest.mark.asyncio
async def test_preflight_callable_from_coroutine():
    """preflight works correctly inside an asyncio coroutine."""
    prompt = wrap_system_prompt("multi-session", top_score=0.6)
    report = preflight(prompt)
    assert report.passed


@pytest.mark.asyncio
async def test_concurrent_wrap_returns_identical_output():
    """50 concurrent tasks calling wrap_system_prompt with the same inputs all return identical output."""
    qtype = "temporal-reasoning"
    score = 0.75

    async def task(_: int) -> str:
        # Pure function — no await needed; wrapping in coroutine is the test.
        return wrap_system_prompt(qtype, top_score=score)

    results = await asyncio.gather(*[task(i) for i in range(50)])

    assert len(results) == 50
    reference = results[0]
    for i, r in enumerate(results):
        assert r == reference, f"Task {i} returned different output"


@pytest.mark.asyncio
async def test_concurrent_preflight_returns_identical_output():
    """50 concurrent tasks calling preflight with the same scaffold return identical results."""
    scaffold = wrap_system_prompt("knowledge-update", top_score=0.55)

    async def task(_: int):
        return preflight(scaffold)

    reports = await asyncio.gather(*[task(i) for i in range(50)])

    assert len(reports) == 50
    reference_passed = reports[0].passed
    reference_failures = reports[0].failures
    for i, r in enumerate(reports):
        assert r.passed == reference_passed, f"Task {i} passed={r.passed}, expected {reference_passed}"
        assert r.failures == reference_failures, f"Task {i} failures differ"


# ---------------------------------------------------------------------------
# b) Streaming compatibility
# ---------------------------------------------------------------------------

def test_wrap_system_prompt_returns_str_not_generator():
    """wrap_system_prompt returns a plain str — no generator, iterator, or awaitable."""
    import inspect
    for qtype in _ALL_QTYPES:
        for score in _SCORES:
            result = wrap_system_prompt(qtype, top_score=score)
            assert isinstance(result, str), f"Expected str, got {type(result)} for qtype={qtype}"
            assert not inspect.isgenerator(result), f"Got generator for qtype={qtype}"
            assert not inspect.iscoroutine(result), f"Got coroutine for qtype={qtype}"
            assert not hasattr(result, "__anext__"), f"Got async iterator for qtype={qtype}"


def test_preflight_returns_report_not_generator():
    """preflight returns a PreflightReport, not a generator or awaitable."""
    import inspect
    from fidelis.scaffold.preflight import PreflightReport
    scaffold = wrap_system_prompt("single-session-preference", top_score=0.9)
    report = preflight(scaffold)
    assert isinstance(report, PreflightReport)
    assert not inspect.isgenerator(report)
    assert not inspect.iscoroutine(report)


# ---------------------------------------------------------------------------
# c) No global state
# ---------------------------------------------------------------------------

def test_no_cross_call_state_contamination():
    """Calling wrap_system_prompt(qt1) then wrap_system_prompt(qt2) doesn't bleed qt1 into qt2's output."""
    pairs = [
        ("single-session-user", "temporal-reasoning"),
        ("multi-session", "knowledge-update"),
        ("single-session-preference", "single-session-assistant"),
        ("temporal-reasoning", "unknown-qtype-xyz"),
    ]
    for qt1, qt2 in pairs:
        # Call qt1 first to potentially pollute any mutable state.
        _ = wrap_system_prompt(qt1, top_score=0.5)
        result_after = wrap_system_prompt(qt2, top_score=0.5)

        # Call qt2 in isolation (fresh call with no prior qt1 call in scope).
        result_isolated = wrap_system_prompt(qt2, top_score=0.5)

        assert result_after == result_isolated, (
            f"Output for qt2={qt2} differs after calling qt1={qt1} first.\n"
            f"After:    {result_after!r}\n"
            f"Isolated: {result_isolated!r}"
        )


def test_no_cross_call_state_contamination_scores():
    """Score used in call N doesn't leak into call N+1."""
    qtype = "single-session-user"
    for score_a, score_b in [(0.9, 0.1), (None, 0.5), (0.5, None), (0.0, 1.0)]:
        _ = wrap_system_prompt(qtype, top_score=score_a)
        result_after = wrap_system_prompt(qtype, top_score=score_b)
        result_isolated = wrap_system_prompt(qtype, top_score=score_b)
        assert result_after == result_isolated, (
            f"Score leak: after score_a={score_a}, call with score_b={score_b} differs from isolated call."
        )


# ---------------------------------------------------------------------------
# d) Memory safety — 10 000 calls, no exception
# ---------------------------------------------------------------------------

def test_no_exception_under_heavy_load():
    """10 000 calls with random qtypes and scores raise no exception."""
    rng = random.Random(42)
    all_qtypes = _ALL_QTYPES + [
        "".join(rng.choices(string.ascii_lowercase + "-", k=rng.randint(3, 20)))
        for _ in range(20)
    ]
    errors = []
    for i in range(10_000):
        qtype = rng.choice(all_qtypes)
        score = rng.choice([None] + [rng.uniform(0.0, 1.0)])
        try:
            result = wrap_system_prompt(qtype, top_score=score)
            assert isinstance(result, str)
        except Exception as exc:
            errors.append((i, qtype, score, exc))

    assert not errors, f"Errors on {len(errors)} calls: first={errors[0]}"


# ---------------------------------------------------------------------------
# e) Error-path: downstream LLM raises, scaffold artifact is unaffected
# ---------------------------------------------------------------------------

def test_scaffold_unaffected_by_downstream_llm_error():
    """Even if the downstream LLM call raises, calling wrap_system_prompt again returns byte-equal output."""

    class FakeLLMError(ConnectionError):
        pass

    def simulate_llm_call(system_prompt: str) -> str:
        raise FakeLLMError("Connection reset by peer")

    qtype = "single-session-user"
    score = 0.8

    # First call — construct the scaffold.
    scaffold_first = wrap_system_prompt(qtype, top_score=score)

    # Simulate the downstream LLM raising mid-conversation.
    with pytest.raises(FakeLLMError):
        simulate_llm_call(scaffold_first)

    # After the error, calling wrap_system_prompt again must return byte-equal output.
    scaffold_second = wrap_system_prompt(qtype, top_score=score)

    assert scaffold_first == scaffold_second, (
        "Scaffold output changed after simulated LLM error — implies mutable state.\n"
        f"First:  {scaffold_first!r}\n"
        f"Second: {scaffold_second!r}"
    )
    # Byte-level check.
    assert scaffold_first.encode("utf-8") == scaffold_second.encode("utf-8"), (
        "Byte-level mismatch after simulated LLM error"
    )


def test_scaffold_byte_equal_across_many_reraises():
    """Under repeated LLM failures, scaffold output stays byte-stable."""

    class TransientError(IOError):
        pass

    qtype = "temporal-reasoning"
    score = 0.65
    reference = wrap_system_prompt(qtype, top_score=score)

    for attempt in range(20):
        try:
            raise TransientError(f"attempt {attempt}")
        except TransientError:
            pass
        result = wrap_system_prompt(qtype, top_score=score)
        assert result.encode("utf-8") == reference.encode("utf-8"), (
            f"Byte mismatch on attempt {attempt}"
        )
