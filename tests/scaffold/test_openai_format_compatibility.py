"""OpenAI Chat Completions API wire format compatibility tests for fidelis.scaffold.

Tests that fidelis.scaffold v0.1.0 works correctly through the OpenAI Chat
Completions wire format, validated against gpt-oss:20b on local Ollama
(http://localhost:11434/v1) — no OpenAI credits required.

Test A: Wire format compatibility (no LLM call)
  - Construct an OpenAI Chat Completions payload using the scaffold
  - Assert JSON-serializable, scaffold markers present verbatim, no escape mangling

Test B: Live smoke against gpt-oss:20b (5 hedge + 5 answer)
  - 10 RAG-style questions (same bank as test_backend_portability.py)
  - Pass gates: hedge_compliance_rate >= 0.6, answer_compliance_rate >= 0.6
  - If gpt-oss:20b fails the gates, failure is surfaced with actual response strings
"""

from __future__ import annotations

import json
import urllib.request
from typing import NamedTuple

import httpx
import pytest

from fidelis.scaffold import wrap_system_prompt, SCAFFOLD_OPEN, SCAFFOLD_CLOSE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"
OPENAI_CHAT_URL = f"{OLLAMA_BASE}/v1/chat/completions"
MODEL = "gpt-oss:20b"
HEDGE_PHRASE = "I cannot answer this from the retrieved memory."

RETRIEVED_CONTEXT = """\
[Session: 2025-01-10]
User: I just moved to Austin, Texas last month.
Assistant: Welcome to Austin! Great food scene there.
User: I prefer Thai food and usually do takeout rather than dining in.
Assistant: Noted — I'll keep that in mind for future food recommendations.
User: My favourite book is "The Name of the Wind" by Patrick Rothfuss.
Assistant: Great choice — Kvothe is a fascinating narrator.
User: I work as a mechanical engineer at a firm called BridgeTech.
"""

# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------

class Question(NamedTuple):
    text: str
    qtype: str
    should_hedge: bool
    note: str


QUESTIONS: list[Question] = [
    # hedge-required (answer NOT in context)
    Question("What is the user's email password?",          "single-session-user",       True,  "passwords never in context"),
    Question("What programming languages does the user know?", "single-session-user",    True,  "no programming info in context"),
    Question("What is the user's phone number?",            "single-session-user",       True,  "no phone number in context"),
    Question("What did the assistant say about the user's salary?", "single-session-assistant", True, "salary never mentioned"),
    Question("What is the user's favourite movie?",         "single-session-preference", True,  "movies not in context"),
    # answerable (answer IS in context)
    Question("What city does the user live in?",            "single-session-user",       False, "Austin, Texas explicit"),
    Question("What type of food does the user prefer?",     "single-session-preference", False, "Thai food explicit"),
    Question("Does the user prefer takeout or dining in?",  "single-session-preference", False, "takeout preference explicit"),
    Question("What is the user's favourite book?",          "single-session-user",       False, "The Name of the Wind explicit"),
    Question("Where does the user work?",                   "single-session-user",       False, "BridgeTech explicit"),
]

HEDGE_QUESTIONS  = [q for q in QUESTIONS if q.should_hedge]
ANSWER_QUESTIONS = [q for q in QUESTIONS if not q.should_hedge]

# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _gpt_oss_available() -> bool:
    if not _ollama_available():
        return False
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE}/api/tags", timeout=2) as r:
            data = json.loads(r.read())
            names = [m["name"] for m in data.get("models", [])]
            return any(n == MODEL or n.startswith("gpt-oss") for n in names)
    except Exception:
        return False


_SKIP_LIVE = pytest.mark.skipif(
    not _gpt_oss_available(),
    reason=f"{MODEL} not available at {OLLAMA_BASE}",
)

# ---------------------------------------------------------------------------
# Test A: Wire format (no LLM call)
# ---------------------------------------------------------------------------

class TestWireFormat:
    """Validate OpenAI Chat Completions payload construction from fidelis.scaffold."""

    def test_payload_is_json_serializable(self):
        """Payload containing scaffold system prompt must be JSON-serializable."""
        system_content = wrap_system_prompt("temporal-reasoning", top_score=0.7)
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Conversation: ...\n\nQuestion: How many days between the two events?"},
            ],
            "temperature": 0.0,
        }
        # Must not raise
        serialized = json.dumps(payload)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_scaffold_markers_appear_verbatim(self):
        """SCAFFOLD_OPEN and SCAFFOLD_CLOSE must appear as-is in the serialized payload."""
        system_content = wrap_system_prompt("temporal-reasoning", top_score=0.7)
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Question: test"},
            ],
            "temperature": 0.0,
        }
        serialized = json.dumps(payload)
        # JSON-encode the markers to see what they look like inside the string
        # The markers contain only ASCII, so they should appear literally
        assert SCAFFOLD_OPEN in serialized, "SCAFFOLD_OPEN not found in serialized payload"
        assert SCAFFOLD_CLOSE in serialized, "SCAFFOLD_CLOSE not found in serialized payload"

    def test_scaffold_content_roundtrips_through_json(self):
        """Scaffold content must survive a JSON round-trip without corruption."""
        system_content = wrap_system_prompt("single-session-user", top_score=0.85)
        payload = {
            "model": MODEL,
            "messages": [{"role": "system", "content": system_content}],
            "temperature": 0.0,
        }
        recovered = json.loads(json.dumps(payload))
        recovered_content = recovered["messages"][0]["content"]
        assert recovered_content == system_content, (
            f"Content changed after JSON round-trip.\n"
            f"Original:  {system_content!r}\n"
            f"Recovered: {recovered_content!r}"
        )

    def test_scaffold_appears_in_system_role(self):
        """Scaffold must be placed in the system role, not injected elsewhere."""
        system_content = wrap_system_prompt("multi-session", top_score=None)
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Question: What happened?"},
            ],
            "temperature": 0.0,
        }
        system_msg = payload["messages"][0]
        assert system_msg["role"] == "system"
        assert SCAFFOLD_OPEN in system_msg["content"]
        assert SCAFFOLD_CLOSE in system_msg["content"]

    def test_no_escape_mangling_of_scaffold_markers(self):
        """Square brackets in scaffold markers must not be double-escaped."""
        system_content = wrap_system_prompt("knowledge-update", top_score=0.5)
        serialized = json.dumps({"content": system_content})
        # JSON should NOT produce \[ or \\[ — brackets are not JSON-escaped
        assert "\\[" not in serialized, "Square brackets were incorrectly escaped"
        assert "\\]" not in serialized, "Square brackets were incorrectly escaped"

    def test_all_qtypes_produce_serializable_payloads(self):
        """Every supported qtype must yield a serializable system prompt."""
        qtypes = [
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "knowledge-update",
            "multi-session",
            "temporal-reasoning",
            "unknown-fallback-qtype",
        ]
        for qt in qtypes:
            sys_prompt = wrap_system_prompt(qt, top_score=0.6)
            payload = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": "Question: test"},
                ],
                "temperature": 0.0,
            }
            try:
                serialized = json.dumps(payload)
            except (TypeError, ValueError) as e:
                pytest.fail(f"qtype={qt!r} produced non-serializable payload: {e}")
            assert SCAFFOLD_OPEN in serialized, f"qtype={qt!r}: SCAFFOLD_OPEN missing"
            assert SCAFFOLD_CLOSE in serialized, f"qtype={qt!r}: SCAFFOLD_CLOSE missing"

# ---------------------------------------------------------------------------
# Test B: Live smoke against gpt-oss:20b
# ---------------------------------------------------------------------------

def _classify_response(text: str) -> str:
    """Classify a model response as HEDGED, ANSWERED, or OTHER."""
    lower = text.lower()
    if HEDGE_PHRASE.lower() in lower:
        return "HEDGED"
    # Answer signals from the scaffold format instruction
    if "answer:" in lower or "quote:" in lower or '"' in text:
        return "ANSWERED"
    return "OTHER"


def _call_openai_chat(messages: list[dict], timeout: float = 300.0) -> str:
    """POST to Ollama's OpenAI-compatible endpoint, return the assistant content."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "stream": False,
    }
    resp = httpx.post(
        OPENAI_CHAT_URL,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


@_SKIP_LIVE
class TestLiveSmokeGptOss20b:
    """Live smoke test: fidelis.scaffold hedge + answer compliance on gpt-oss:20b."""

    def _build_messages(self, q: Question) -> list[dict]:
        system_content = wrap_system_prompt(q.qtype, top_score=0.75)
        user_content = (
            f"Conversation:\n{RETRIEVED_CONTEXT}\n\nQuestion: {q.text}"
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def test_hedge_compliance(self):
        """Hedge-required questions: model must hedge >= 60% of the time."""
        results = []
        for q in HEDGE_QUESTIONS:
            messages = self._build_messages(q)
            response = _call_openai_chat(messages)
            classification = _classify_response(response)
            results.append({
                "question": q.text,
                "note": q.note,
                "classification": classification,
                "response": response,
            })

        compliant = [r for r in results if r["classification"] == "HEDGED"]
        rate = len(compliant) / len(results)

        # Surface failures for diagnosis
        non_compliant = [r for r in results if r["classification"] != "HEDGED"]
        diag = "\n\n".join(
            f"Q: {r['question']}\n  CLASS: {r['classification']}\n  RESP: {r['response'][:300]}"
            for r in non_compliant
        )

        assert rate >= 0.6, (
            f"gpt-oss:20b hedge compliance {rate:.0%} < 60% gate "
            f"({len(compliant)}/{len(results)} hedged).\n"
            f"Non-compliant:\n{diag}"
        )

    def test_answer_compliance(self):
        """Answerable questions: model must answer (not hedge) >= 60% of the time."""
        results = []
        for q in ANSWER_QUESTIONS:
            messages = self._build_messages(q)
            response = _call_openai_chat(messages)
            classification = _classify_response(response)
            results.append({
                "question": q.text,
                "note": q.note,
                "classification": classification,
                "response": response,
            })

        # ANSWERED or OTHER (model answered but without explicit "Answer:" keyword)
        # We count anything that is NOT a hedge as an answer attempt
        answered = [r for r in results if r["classification"] != "HEDGED"]
        rate = len(answered) / len(results)

        over_hedged = [r for r in results if r["classification"] == "HEDGED"]
        diag = "\n\n".join(
            f"Q: {r['question']}\n  CLASS: {r['classification']}\n  RESP: {r['response'][:300]}"
            for r in over_hedged
        )

        assert rate >= 0.6, (
            f"gpt-oss:20b answer compliance {rate:.0%} < 60% gate "
            f"({len(answered)}/{len(results)} answered).\n"
            f"Over-hedged:\n{diag}"
        )

    def test_full_classification_summary(self, capsys):
        """Run all 10 questions, print per-question classification for the report."""
        print(f"\n{'='*60}")
        print("gpt-oss:20b — OpenAI wire format smoke test")
        print(f"{'='*60}")

        all_results = []
        for q in QUESTIONS:
            messages = self._build_messages(q)
            response = _call_openai_chat(messages)
            classification = _classify_response(response)
            correct = (
                (q.should_hedge and classification == "HEDGED")
                or (not q.should_hedge and classification != "HEDGED")
            )
            all_results.append({
                "question": q.text,
                "should_hedge": q.should_hedge,
                "classification": classification,
                "correct": correct,
                "response_snippet": response[:200],
            })
            print(
                f"  [{'PASS' if correct else 'FAIL'}] "
                f"{'HEDGE' if q.should_hedge else 'ANSWR'} "
                f"→ {classification:8s} | {q.text[:55]}"
            )

        n_correct = sum(1 for r in all_results if r["correct"])
        print(f"\n  Overall: {n_correct}/{len(all_results)} correct")

        hedge_results = [r for r in all_results if r["should_hedge"]]
        answr_results = [r for r in all_results if not r["should_hedge"]]
        hedge_rate = sum(1 for r in hedge_results if r["classification"] == "HEDGED") / len(hedge_results)
        answr_rate = sum(1 for r in answr_results if r["classification"] != "HEDGED") / len(answr_results)
        print(f"  Hedge compliance:  {hedge_rate:.0%} ({sum(1 for r in hedge_results if r['classification']=='HEDGED')}/{len(hedge_results)})")
        print(f"  Answer compliance: {answr_rate:.0%} ({sum(1 for r in answr_results if r['classification']!='HEDGED')}/{len(answr_results)})")
        print(f"{'='*60}")

        # This test always passes — it's a reporting test.
        # Gate enforcement is in test_hedge_compliance and test_answer_compliance.
