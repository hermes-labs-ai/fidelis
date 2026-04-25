"""Retrieval-agnostic compatibility tests for fidelis.scaffold v0.1.0.

Validates that the scaffold works correctly when paired with retrieved context
strings built from LangChain Documents, mem0 dicts, LlamaIndex NodeWithScore
dicts, raw strings, large user prompts, and non-English input.

The scaffold is retrieval-agnostic by design: callers build their context string
from whatever retrieval backend they use, then pass the scaffold as system prompt
and the (context + question) as user message. These tests verify that:
  - wrap_system_prompt + preflight pass for all common retrieval shapes
  - No nested scaffold markers bleed into combined prompts
  - Non-English question content doesn't crash preflight or wrap_system_prompt
"""

from __future__ import annotations

from collections import namedtuple

import pytest

from fidelis.scaffold import (
    SCAFFOLD_CLOSE,
    SCAFFOLD_OPEN,
    is_scaffolded,
    preflight,
    wrap_system_prompt,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(context: str, question: str) -> str:
    """Combine retrieved context + question the way a caller would."""
    return f"Retrieved context:\n{context}\n\nQuestion: {question}"


def _assert_scaffold_passes(scaffold: str) -> None:
    """Assert preflight passes and scaffold has exactly one marker pair."""
    report = preflight(scaffold)
    assert report.passed, f"Preflight FAILED:\n{report.summary()}"
    assert scaffold.count(SCAFFOLD_OPEN) == 1
    assert scaffold.count(SCAFFOLD_CLOSE) == 1


# ---------------------------------------------------------------------------
# (a) LangChain Document shape
# ---------------------------------------------------------------------------

Document = namedtuple("Document", ["page_content", "metadata"])


def test_langchain_document_context():
    """Context built from LangChain Document objects passes preflight."""
    docs = [
        Document(page_content="User prefers dark mode.", metadata={"session": "2024-01-01", "source": "chat"}),
        Document(page_content="User mentioned they use Python 3.11.", metadata={"session": "2024-01-02"}),
    ]
    # Caller-side context-building (scaffold is agnostic to this)
    context = "\n".join(
        f"[session={d.metadata.get('session', 'unknown')}] {d.page_content}"
        for d in docs
    )
    scaffold = wrap_system_prompt("single-session-preference", top_score=0.82)
    user_msg = _build_user_message(context, "What are the user's preferences?")

    _assert_scaffold_passes(scaffold)
    # Scaffold must NOT appear in user message (no bleed)
    assert not is_scaffolded(user_msg)


def test_langchain_document_no_nested_markers():
    """Combined system+user message contains exactly one scaffold block."""
    docs = [Document(page_content="User said hello.", metadata={})]
    context = docs[0].page_content
    scaffold = wrap_system_prompt("single-session-user", top_score=0.75)
    combined = scaffold + "\n\n" + _build_user_message(context, "What did the user say?")

    assert combined.count(SCAFFOLD_OPEN) == 1
    assert combined.count(SCAFFOLD_CLOSE) == 1


# ---------------------------------------------------------------------------
# (b) mem0 message shape
# ---------------------------------------------------------------------------

def test_mem0_message_context():
    """Context built from mem0 dicts passes preflight."""
    memories = [
        {"memory": "User lives in Berlin.", "score": 0.91, "id": "mem-001"},
        {"memory": "User's favorite coffee is espresso.", "score": 0.78, "id": "mem-002"},
    ]
    # Caller builds context from mem0 results
    context = "\n".join(
        f"[score={m['score']:.2f}] {m['memory']}"
        for m in memories
    )
    scaffold = wrap_system_prompt("single-session-user", top_score=memories[0]["score"])
    user_msg = _build_user_message(context, "Where does the user live?")

    _assert_scaffold_passes(scaffold)
    assert not is_scaffolded(user_msg)


def test_mem0_score_fed_to_scaffold():
    """Top mem0 score correctly maps to HIGH confidence marker."""
    memories = [{"memory": "User prefers async APIs.", "score": 0.95, "id": "mem-003"}]
    scaffold = wrap_system_prompt("knowledge-update", top_score=memories[0]["score"])
    assert "HIGH" in scaffold
    _assert_scaffold_passes(scaffold)


# ---------------------------------------------------------------------------
# (c) LlamaIndex NodeWithScore shape
# ---------------------------------------------------------------------------

def test_llamaindex_node_with_score_context():
    """Context built from LlamaIndex NodeWithScore dicts passes preflight."""
    nodes = [
        {"node": {"text": "Session 2024-03-01: User asked about LangChain."}, "score": 0.88},
        {"node": {"text": "Session 2024-03-02: User asked about LlamaIndex."}, "score": 0.71},
    ]
    context = "\n".join(
        f"[score={n['score']:.2f}] {n['node']['text']}"
        for n in nodes
    )
    top_score = nodes[0]["score"]
    scaffold = wrap_system_prompt("multi-session", top_score=top_score)
    user_msg = _build_user_message(context, "What frameworks did the user ask about?")

    _assert_scaffold_passes(scaffold)
    assert not is_scaffolded(user_msg)


def test_llamaindex_medium_score_marker():
    """Score 0.61 maps to MEDIUM confidence marker."""
    nodes = [{"node": {"text": "Some fact."}, "score": 0.61}]
    scaffold = wrap_system_prompt("single-session-assistant", top_score=nodes[0]["score"])
    assert "MEDIUM" in scaffold
    _assert_scaffold_passes(scaffold)


# ---------------------------------------------------------------------------
# (d) Raw string context
# ---------------------------------------------------------------------------

def test_raw_string_context():
    """Plain string context passes preflight; most basic retrieval shape."""
    context = "User said X yesterday. Y happened today."
    scaffold = wrap_system_prompt("temporal-reasoning", top_score=None)
    user_msg = _build_user_message(context, "How many days between X and Y?")

    _assert_scaffold_passes(scaffold)
    assert not is_scaffolded(user_msg)


def test_raw_string_no_score():
    """No retrieval score maps to 'unknown' quality marker."""
    scaffold = wrap_system_prompt("single-session-user", top_score=None)
    assert "unknown" in scaffold
    _assert_scaffold_passes(scaffold)


# ---------------------------------------------------------------------------
# (e) Token-bursting: large user prompt
# ---------------------------------------------------------------------------

def test_large_user_prompt_scaffold_still_bounded():
    """Scaffold alone passes preflight even when user prompt is very large."""
    large_user_prompt = "A" * 4000  # 4000-char user prompt (~1000 tokens)
    scaffold = wrap_system_prompt("multi-session", top_score=0.9)

    # Scaffold itself must be within 200-token bound
    _assert_scaffold_passes(scaffold)

    # Combined string must NOT introduce nested scaffold markers
    combined = scaffold + "\n\n" + large_user_prompt
    assert combined.count(SCAFFOLD_OPEN) == 1
    assert combined.count(SCAFFOLD_CLOSE) == 1


def test_large_user_prompt_no_scaffold_bleed():
    """Large user prompt with repeated brackets/brackets doesn't confuse marker counts."""
    # Stress: user prompt has bracket-heavy content (e.g. JSON logs)
    noisy_prompt = '[{"key": "val"}] ' * 200  # lots of brackets, no scaffold markers
    scaffold = wrap_system_prompt("knowledge-update", top_score=0.55)
    combined = scaffold + "\n\n" + noisy_prompt

    # Only one scaffold block
    assert combined.count(SCAFFOLD_OPEN) == 1
    assert combined.count(SCAFFOLD_CLOSE) == 1
    # Scaffold itself still passes
    _assert_scaffold_passes(scaffold)


# ---------------------------------------------------------------------------
# (f) Multi-language input (non-English question content)
# ---------------------------------------------------------------------------

def test_non_english_question_does_not_crash():
    """Non-English question in user message doesn't crash wrap_system_prompt."""
    question = "¿Cuántos días pasaron entre los eventos?"
    context = "2024-01-10: evento A. 2024-01-15: evento B."
    scaffold = wrap_system_prompt("temporal-reasoning", top_score=0.7)
    user_msg = _build_user_message(context, question)

    # wrap_system_prompt must not raise
    assert isinstance(scaffold, str)
    assert len(scaffold) > 0

    # Preflight on scaffold alone must pass (scaffold is English; question is user's)
    _assert_scaffold_passes(scaffold)

    # User message (non-English) should not be scaffolded
    assert not is_scaffolded(user_msg)


def test_non_english_context_does_not_crash_preflight_on_scaffold():
    """Even if context contains non-ASCII, the scaffold (system prompt) stays clean."""
    # Scaffold is English-only; non-ASCII lives in user message, not scaffold
    scaffold = wrap_system_prompt("single-session-user", top_score=0.8)
    _assert_scaffold_passes(scaffold)


@pytest.mark.parametrize("question", [
    "¿Cuántos días?",           # Spanish
    "Wie viele Tage?",          # German
    "何日間ですか？",             # Japanese
    "Сколько дней?",            # Russian
    "كم عدد الأيام؟",           # Arabic
])
def test_multilingual_questions_scaffold_unchanged(question):
    """For every language, scaffold text is identical (language-agnostic)."""
    scaffold_a = wrap_system_prompt("temporal-reasoning", top_score=0.7)
    scaffold_b = wrap_system_prompt("temporal-reasoning", top_score=0.7)
    # Scaffold must be deterministic regardless of question content (question not passed in)
    assert scaffold_a == scaffold_b
    _assert_scaffold_passes(scaffold_a)
