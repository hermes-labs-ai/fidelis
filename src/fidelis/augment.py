"""fidelis.augment — the one-line caller helper.

For users who want "give me memory + scaffold + answer" in one call instead of
wiring three separate functions. Pure wrapper; uses the running fidelis-server
for retrieval and the user's own LLM client for generation.

Example:

    from fidelis.augment import augment
    from anthropic import Anthropic

    client = Anthropic()
    response = augment(
        question="What did I say about Sarah?",
        qtype="single-session-user",
        llm_call=lambda system, user: client.messages.create(
            model="claude-opus-4-7",  # use any current Claude Messages API model
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=512,
        ).content[0].text,
    )
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

from fidelis.scaffold import wrap_system_prompt


def _server_url() -> str:
    port = os.environ.get("FIDELIS_PORT", os.environ.get("COGITO_PORT", "19420"))
    return f"http://127.0.0.1:{port}"


def _recall(query: str, limit: int = 5) -> tuple[str, float | None]:
    """Retrieve memories from fidelis-server. Returns (formatted_context, top_score)."""
    body = json.dumps({"text": query, "limit": limit}).encode()
    req = urllib.request.Request(
        f"{_server_url()}/recall",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"fidelis-server unreachable at {_server_url()}. "
            f"Run `fidelis init` to install + start the service. ({e})"
        ) from e

    memories = data.get("memories", [])
    if not memories:
        return "(no memories retrieved)", None

    lines = []
    top_score: float | None = None
    for m in memories:
        text = m.get("text", "").strip()
        score = m.get("score")
        if score is not None and top_score is None:
            try:
                top_score = float(score)
                # mem0/cogito returns "lower is better" similarity; normalize roughly to [0, 1]
                if top_score > 1.0:
                    # heuristic: distance scores in [0, ~500] → invert to [0, 1]
                    top_score = max(0.0, min(1.0, 1.0 - (top_score / 500.0)))
            except (ValueError, TypeError):
                top_score = None
        lines.append(text)

    return "\n\n---\n\n".join(lines), top_score


def augment(
    question: str,
    qtype: str = "single-session-user",
    *,
    llm_call: Callable[[str, str], str],
    limit: int = 5,
) -> str:
    """Retrieve memory + wrap with scaffold + invoke caller's LLM.

    Args:
        question: the user's natural-language question
        qtype: one of single-session-user / single-session-assistant /
            single-session-preference / knowledge-update / multi-session /
            temporal-reasoning. Caller chooses based on their question type.
            Default: single-session-user.
        llm_call: a callable taking (system_prompt, user_message) and returning
            the LLM's response text. The user supplies this so fidelis stays
            agnostic to LLM SDK choice.
        limit: max memories to retrieve (default 5)

    Returns:
        The LLM's response text.

    Raises:
        RuntimeError: if fidelis-server is unreachable.
    """
    context, top_score = _recall(question, limit=limit)
    system = wrap_system_prompt(qtype, top_score=top_score)
    user_message = f"Conversation memory:\n{context}\n\nQuestion: {question}"
    return llm_call(system, user_message)
