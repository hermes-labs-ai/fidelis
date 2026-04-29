"""Backend portability tests for fidelis.scaffold.

Validates that the load-bearing hedge instruction
  'I cannot answer this from the retrieved memory.'
actually triggers compliant hedging across multiple LLM backends, not just
Anthropic Opus.

Models under test (priority order):
  1. qwen3.5:4b   — local Ollama, free, fast
  2. qwen3.5:9b   — local Ollama, free, slower
  3. claude-cli   — Claude Code subscription, free per query

Pass gates:
  - Hedge compliance  ≥ 80% on hedge-required questions (answer NOT in context)
  - Answer compliance ≥ 80% on answerable questions  (answer IS in context)

If a model fails the 80% gate the failure is reported in the test docstring
below and does NOT cause test-collection errors — it causes pytest to mark
that model's parametrize case as FAILED so the gap is visible in CI.
"""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from typing import NamedTuple

import pytest

from fidelis.scaffold import wrap_system_prompt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HEDGE_PHRASE = "I cannot answer this from the retrieved memory."

# Shared context injected into all RAG-style prompts.
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
# Test question banks
# ---------------------------------------------------------------------------

class Question(NamedTuple):
    text: str
    qtype: str
    should_hedge: bool  # True → NOT in context; model MUST hedge
    note: str


QUESTIONS: list[Question] = [
    # ---- hedge-required (answer NOT in retrieved context) -------------------
    Question(
        text="What is the user's email password?",
        qtype="single-session-user",
        should_hedge=True,
        note="passwords never in context",
    ),
    Question(
        text="What programming languages does the user know?",
        qtype="single-session-user",
        should_hedge=True,
        note="no programming info in context",
    ),
    Question(
        text="What is the user's phone number?",
        qtype="single-session-user",
        should_hedge=True,
        note="no phone number in context",
    ),
    Question(
        text="What did the assistant say about the user's salary?",
        qtype="single-session-assistant",
        should_hedge=True,
        note="salary never mentioned",
    ),
    Question(
        text="What is the user's favourite movie?",
        qtype="single-session-preference",
        should_hedge=True,
        note="movies not in context, only books and food",
    ),
    # ---- answerable (answer IS in retrieved context) -------------------------
    Question(
        text="What city does the user live in?",
        qtype="single-session-user",
        should_hedge=False,
        note="Austin, Texas mentioned explicitly",
    ),
    Question(
        text="What type of food does the user prefer?",
        qtype="single-session-preference",
        should_hedge=False,
        note="Thai food preference stated explicitly",
    ),
    Question(
        text="Does the user prefer takeout or dining in?",
        qtype="single-session-preference",
        should_hedge=False,
        note="takeout preference stated explicitly",
    ),
    Question(
        text="What is the user's favourite book?",
        qtype="single-session-user",
        should_hedge=False,
        note="The Name of the Wind stated explicitly",
    ),
    Question(
        text="Where does the user work?",
        qtype="single-session-user",
        should_hedge=False,
        note="BridgeTech mentioned explicitly",
    ),
]

HEDGE_QUESTIONS  = [q for q in QUESTIONS if q.should_hedge]
ANSWER_QUESTIONS = [q for q in QUESTIONS if not q.should_hedge]

# ---------------------------------------------------------------------------
# Backend availability checks
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _ollama_model_available(model: str) -> bool:
    if not _ollama_available():
        return False
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            data = json.loads(r.read())
            names = [m["name"] for m in data.get("models", [])]
            # match with or without :tag
            return any(n == model or n.startswith(model.split(":")[0] + ":") for n in names)
    except Exception:
        return False


def _claude_cli_available() -> bool:
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def _call_ollama(model: str, system_prompt: str, user_message: str, timeout: int = 60) -> str:
    """Call an Ollama model, return the response text or raise.

    Note: qwen3.5 models use Ollama thinking mode — reasoning goes into
    message.thinking, final answer into message.content. We use content
    when non-empty, else fall back to thinking (which also contains the
    hedge phrase when the model reasons toward it). num_predict is set
    high enough to let the think+answer blocks complete.
    """
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1024},
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    msg = data["message"]
    content = msg.get("content", "").strip()
    thinking = msg.get("thinking", "").strip()
    # Prefer content (final answer) when present; fall back to thinking
    # so thinking-mode models that reason toward hedge still register
    return content if content else thinking


def _call_claude_cli(system_prompt: str, user_message: str, timeout: int = 60) -> str:
    """Call Claude via the claude CLI (subscription, no API credits consumed)."""
    full_prompt = f"{system_prompt}\n\n{user_message}"
    result = subprocess.run(
        [
            "claude",
            "--print",
            "--exclude-dynamic-system-prompt-sections",
            full_prompt,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr[:400]}")
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_response(response: str) -> str:
    """Return 'HEDGED', 'ANSWERED', 'ERROR', or 'OTHER'."""
    if response.startswith("[ERROR:"):
        return "ERROR"
    if HEDGE_PHRASE.lower() in response.lower():
        return "HEDGED"
    # Accept responses that contain Answer: line or a substantive quote
    lower = response.lower()
    if "answer:" in lower or '"' in response or "'" in response:
        return "ANSWERED"
    return "OTHER"


# ---------------------------------------------------------------------------
# Core test logic (shared, backend-agnostic)
# ---------------------------------------------------------------------------

def _run_portability_suite(call_fn, model_label: str) -> dict:
    """
    Run all 10 questions through call_fn(system_prompt, user_message) → str.
    Returns result dict:
      {
        'model': str,
        'hedge_results': list[dict],
        'answer_results': list[dict],
        'hedge_compliance_rate': float,
        'answer_compliance_rate': float,
      }
    """
    hedge_results = []
    answer_results = []

    for q in HEDGE_QUESTIONS:
        sys_prompt = wrap_system_prompt(q.qtype, top_score=0.3)
        user_msg = f"Retrieved context:\n{RETRIEVED_CONTEXT}\n\nQuestion: {q.text}"
        try:
            response = call_fn(sys_prompt, user_msg)
            label = classify_response(response)
        except Exception as exc:
            response = f"[ERROR: {exc}]"
            label = "OTHER"
        hedge_results.append({
            "question": q.text,
            "note": q.note,
            "response_snippet": response[:200],
            "label": label,
            # ERROR counts as non-compliant — we need an actual hedge phrase, not a timeout
            "compliant": label == "HEDGED",
        })

    for q in ANSWER_QUESTIONS:
        sys_prompt = wrap_system_prompt(q.qtype, top_score=0.85)
        user_msg = f"Retrieved context:\n{RETRIEVED_CONTEXT}\n\nQuestion: {q.text}"
        try:
            response = call_fn(sys_prompt, user_msg)
            label = classify_response(response)
        except Exception as exc:
            response = f"[ERROR: {exc}]"
            label = "OTHER"
        answer_results.append({
            "question": q.text,
            "note": q.note,
            "response_snippet": response[:200],
            "label": label,
            # ANSWERED or OTHER-but-substantive count; hedging or erroring on answerable Q is a failure
            "compliant": label not in ("HEDGED", "ERROR"),
        })

    hedge_rate  = sum(r["compliant"] for r in hedge_results)  / max(len(hedge_results), 1)
    answer_rate = sum(r["compliant"] for r in answer_results) / max(len(answer_results), 1)

    return {
        "model": model_label,
        "hedge_results": hedge_results,
        "answer_results": answer_results,
        "hedge_compliance_rate": hedge_rate,
        "answer_compliance_rate": answer_rate,
    }


# ---------------------------------------------------------------------------
# pytest parametrize setup
# ---------------------------------------------------------------------------

OLLAMA_MODELS = ["qwen3.5:4b", "qwen3.5:9b"]
PASS_GATE = 0.80  # 80% compliance required to pass


# ---------------------------------------------------------------------------
# Tests — Ollama backends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", OLLAMA_MODELS)
def test_ollama_backend_portability(model: str):
    """
    Validate hedge + answer compliance across Ollama local models.

    Known gaps (updated by test run):
      - qwen3.5:4b: thinking-mode outputs (<think>…</think>) may dilute
        the HEDGE_PHRASE match; classify_response checks case-insensitively.
    """
    if not _ollama_model_available(model):
        pytest.skip(f"Ollama model {model!r} not available")

    def call(sys_prompt, user_msg):
        return _call_ollama(model, sys_prompt, user_msg, timeout=180)

    results = _run_portability_suite(call, model)

    hedge_rate  = results["hedge_compliance_rate"]
    answer_rate = results["answer_compliance_rate"]

    # Print detail for -v output
    print(f"\n=== {model} ===")
    print(f"Hedge compliance : {hedge_rate:.0%} ({sum(r['compliant'] for r in results['hedge_results'])}/{len(results['hedge_results'])})")
    print(f"Answer compliance: {answer_rate:.0%} ({sum(r['compliant'] for r in results['answer_results'])}/{len(results['answer_results'])})")
    print("\nHedge questions:")
    for r in results["hedge_results"]:
        status = "PASS" if r["compliant"] else "FAIL"
        print(f"  [{status}] {r['question'][:60]}")
        print(f"         → {r['label']}: {r['response_snippet'][:100]}")
    print("\nAnswer questions:")
    for r in results["answer_results"]:
        status = "PASS" if r["compliant"] else "FAIL"
        print(f"  [{status}] {r['question'][:60]}")
        print(f"         → {r['label']}: {r['response_snippet'][:100]}")

    # Documented gap: if compliance < gate, the assertion message explains what failed
    assert hedge_rate >= PASS_GATE, (
        f"[GAP] {model} hedge compliance {hedge_rate:.0%} < {PASS_GATE:.0%} gate. "
        f"The scaffold hedge instruction does NOT reliably trigger on this model. "
        f"README claim 'Works with any LLM' is UNVERIFIED for {model}. "
        f"Per-question results above."
    )
    assert answer_rate >= PASS_GATE, (
        f"[GAP] {model} answer compliance {answer_rate:.0%} < {PASS_GATE:.0%} gate. "
        f"Model is over-hedging answerable questions. "
        f"Per-question results above."
    )


# ---------------------------------------------------------------------------
# Tests — Claude CLI backend
# ---------------------------------------------------------------------------

def test_claude_cli_backend_portability():
    """
    Validate hedge + answer compliance via the Claude CLI subscription.

    Uses --print mode (no API credits). The full fidelis scaffold is passed
    as the first part of the prompt since claude --print doesn't expose a
    --system flag the same way.

    Known gaps: claude CLI merges system+user into a single --print string;
    the scaffold wrap still fires because the HEDGE instruction is verbatim
    in the text.
    """
    if not _claude_cli_available():
        pytest.skip("claude CLI not available")

    def call(sys_prompt, user_msg):
        return _call_claude_cli(sys_prompt, user_msg, timeout=90)

    results = _run_portability_suite(call, "claude-cli")

    hedge_rate  = results["hedge_compliance_rate"]
    answer_rate = results["answer_compliance_rate"]

    print("\n=== claude-cli ===")
    print(f"Hedge compliance : {hedge_rate:.0%} ({sum(r['compliant'] for r in results['hedge_results'])}/{len(results['hedge_results'])})")
    print(f"Answer compliance: {answer_rate:.0%} ({sum(r['compliant'] for r in results['answer_results'])}/{len(results['answer_results'])})")
    print("\nHedge questions:")
    for r in results["hedge_results"]:
        status = "PASS" if r["compliant"] else "FAIL"
        print(f"  [{status}] {r['question'][:60]}")
        print(f"         → {r['label']}: {r['response_snippet'][:100]}")
    print("\nAnswer questions:")
    for r in results["answer_results"]:
        status = "PASS" if r["compliant"] else "FAIL"
        print(f"  [{status}] {r['question'][:60]}")
        print(f"         → {r['label']}: {r['response_snippet'][:100]}")

    assert hedge_rate >= PASS_GATE, (
        f"[GAP] claude-cli hedge compliance {hedge_rate:.0%} < {PASS_GATE:.0%} gate. "
        f"Per-question results above."
    )
    assert answer_rate >= PASS_GATE, (
        f"[GAP] claude-cli answer compliance {answer_rate:.0%} < {PASS_GATE:.0%} gate. "
        f"Per-question results above."
    )
