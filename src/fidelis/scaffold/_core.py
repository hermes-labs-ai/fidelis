"""Fidelis Scaffold v0.1.0 — drift-safe QA wrapper for retrieved memory.

The scaffold modifies the LLM's procedure for answering retrieval-augmented questions
WITHOUT modifying the LLM itself. It is designed to:

1. Lift accuracy on questions where extraction-only fails (TR, MS, Pref).
2. Preserve hedge ability — "I don't know" stays available when retrieval is thin.
3. Bound multi-turn linguistic drift via versioned, removable scaffold wrap.
4. Pass static validation (lintlang, scaffold-lint, custom preflight).
5. Be idempotent: wrap(wrap(x)) == wrap(x).
6. Stay below 200 input tokens at all qtypes.

Versioning: scaffold tokens are bracketed with [FIDELIS-SCAFFOLD-v0.1.0] markers so
downstream tools (driftwatch, agent-convergence-scorer) can locate and remove
scaffold contributions when measuring drift on subsequent turns.
"""

from __future__ import annotations

import math as _math

# Version banner — used as scaffold-presence marker for drift measurement.
SCAFFOLD_VERSION = "v0.1.0"
SCAFFOLD_OPEN = f"[FIDELIS-SCAFFOLD-{SCAFFOLD_VERSION}]"
SCAFFOLD_CLOSE = f"[/FIDELIS-SCAFFOLD-{SCAFFOLD_VERSION}]"

# Hedge invitation — overrides the v3 prompts' "never answer UNKNOWN" rule.
# Calibrated hedging is the load-bearing safety property: when retrieval is
# thin, the model should say so rather than confabulate.
_HEDGE = (
    "If the retrieved conversation does NOT contain the answer, respond exactly: "
    "'I cannot answer this from the retrieved memory.' This is calibrated; "
    "do not penalize yourself for hedging when evidence is absent."
)

# Confidence signal — inline meta-information about retrieval quality.
# Helps the LLM calibrate its confidence to actual retrieval quality.


def _sanitize_top_score(top_score: float | None) -> float | None:
    """Sanitize a raw retrieval score to [0, 1] or None.

    Values outside [0, 1] are clamped; nan/inf/-inf are treated as None (unknown).
    Accepts int 0/1 and bool (True/False) in addition to float.
    """
    if top_score is None:
        return None
    try:
        v = float(top_score)
    except (TypeError, ValueError):
        return None
    if not _math.isfinite(v):
        # nan, inf, -inf → unknown
        return None
    return max(0.0, min(1.0, v))


def _confidence_marker(top_score: float | None) -> str:
    score = _sanitize_top_score(top_score)
    if score is None:
        return "[retrieval-quality: unknown]"
    if score >= 0.7:
        return "[retrieval-quality: HIGH (top-1 similarity ≥ 0.7)]"
    if score >= 0.5:
        return "[retrieval-quality: MEDIUM (top-1 similarity 0.5-0.7)]"
    return "[retrieval-quality: LOW (top-1 similarity < 0.5) — hedge if unclear]"


# qtype-specific procedural instructions. Compact (each ~30-50 tokens).
_QTYPE_PROC = {
    "single-session-user": (
        "Procedure: Quote the user's exact statement. Answer concisely citing the quote. "
        "If the user did not state this, hedge."
    ),
    "single-session-assistant": (
        "Procedure: Quote the assistant's exact statement. Answer using only the quote. "
        "If the assistant did not state this, hedge."
    ),
    "single-session-preference": (
        "Procedure: Aggregate ALL the user's stated preferences from the conversation. "
        "Recommend based ONLY on those preferences. Cite each one. If preferences are absent, hedge."
    ),
    "knowledge-update": (
        "Procedure: List every quote about the subject with its session date. "
        "Sort by date. The MOST RECENT quote is the current answer. Older versions are superseded."
    ),
    "multi-session": (
        "Procedure: For each session containing partial information, quote the relevant passage. "
        "Combine ALL partial answers from ALL sessions. Do NOT stop after one session. "
        "Aggregate (count/list/sum) across sessions to form the complete answer."
    ),
    "temporal-reasoning": (
        "Procedure: Find the session date headers (YYYY/MM/DD format) for both events in the question. "
        "Compute the calendar difference between THOSE TWO DATES — never use 'today' or training cutoff. "
        "Show: 'From DATE_A to DATE_B = N days'. Off-by-one is forgiven."
    ),
}


def wrap_system_prompt(qtype: str, top_score: float | None = None) -> str:
    """Return a complete Fidelis-scaffold system prompt for a given qtype.

    Args:
        qtype: one of single-session-user, single-session-assistant,
            single-session-preference, knowledge-update, multi-session,
            temporal-reasoning. Unknown qtypes get a generic factual prompt.
        top_score: optional top-1 retrieval similarity score (0..1) for
            inline confidence signaling. None = no signal.

    Returns:
        A string ready to use as the LLM's system prompt. Always begins
        with SCAFFOLD_OPEN and ends with SCAFFOLD_CLOSE for downstream
        scaffold-detection / drift measurement.
    """
    qtype = qtype.lower().strip()
    proc = _QTYPE_PROC.get(qtype) or (
        "Procedure: Quote the relevant passage from the retrieved conversation. "
        "Answer using only the quote. If the answer is absent, hedge."
    )
    confidence = _confidence_marker(top_score)
    body = (
        f"{SCAFFOLD_OPEN}\n"
        f"You are answering a question using retrieved conversation memory.\n"
        f"{confidence}\n"
        f"{proc}\n"
        f"{_HEDGE}\n"
        f"Format: quote(s) first, then a single line starting with 'Answer:'.\n"
        f"{SCAFFOLD_CLOSE}"
    )
    return body


def is_scaffolded(text: str) -> bool:
    """Detect whether text already contains a Fidelis scaffold (any version)."""
    return "[FIDELIS-SCAFFOLD-" in text


def strip_scaffold(text: str) -> str:
    """Remove all Fidelis scaffold sections from text. Idempotent for un-scaffolded text."""
    import re
    pat = re.compile(
        r"\[FIDELIS-SCAFFOLD-[^\]]+\].*?\[/FIDELIS-SCAFFOLD-[^\]]+\]",
        re.DOTALL,
    )
    return pat.sub("", text).strip()


def wrap_idempotent(qtype: str, top_score: float | None = None, prior: str = "") -> str:
    """Idempotent wrap: if prior already has any scaffold, strip it first then re-wrap.
    Guarantees scaffold(scaffold(x)) == scaffold(x) at the prompt-construction layer."""
    if prior and is_scaffolded(prior):
        prior = strip_scaffold(prior)
    new = wrap_system_prompt(qtype, top_score=top_score)
    if prior:
        return new + "\n\n" + prior
    return new
