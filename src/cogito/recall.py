"""
cogito recall — two-stage retrieval with integer-pointer fidelity filter.

Stage 1: broad vector search — top-N by rank, NO hard L2 threshold.
         Threshold would exclude the right memory before Stage 2 ever sees it.
         Stage 2 exists to handle noise, so Stage 1 should maximise recall.

Stage 2: cheap LLM receives numbered candidate list, outputs integer indices ONLY.
         Never outputs memory text — structurally cannot corrupt or hallucinate
         into the content returned to the caller.

Stage 3: server picks verbatim candidate text by those indices and returns it.

Callable as a library:
    from cogito.recall import recall
    memories, method = recall(memory, "query", user_id="agent", cfg=cfg)

Or hit the HTTP endpoint:
    POST /recall  {"text": "...", "limit": 50}
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


def recall(
    memory: Any,
    query: str,
    user_id: str,
    cfg: dict[str, Any],
    limit: int | None = None,
) -> tuple[list[dict], str]:
    """
    Run two-stage recall. Returns (memories, method).

    memories: list of {"text": str, "score": float}
    method:   "filter" | "fallback_*"
    """
    limit = limit or cfg.get("recall_limit", 50)

    # Stage 1 — top-N by rank, no threshold cut.
    # The filter handles precision; cutting here only creates recall ceiling.
    raw = memory.search(query, user_id=user_id, limit=min(limit, 100))
    candidates = [
        {"text": r.get("memory", ""), "score": round(r.get("score", 9999), 3)}
        for r in raw.get("results", [])
        if r.get("memory")  # skip empty strings only
    ]

    if not candidates:
        return [], "no_candidates"

    # Stage 2 — integer-pointer filter
    selected, method = _filter(query, candidates, cfg)
    return selected, method


def _filter(
    query: str,
    candidates: list[dict],
    cfg: dict[str, Any],
) -> tuple[list[dict], str]:
    """
    Ask the filter model which candidates are relevant.
    Outputs only integer indices — the model never generates memory text.
    """
    endpoint, token = _resolve_filter_endpoint(cfg)
    if not endpoint:
        return candidates, "fallback_no_endpoint"

    model = cfg.get("filter_model", "anthropic/claude-haiku-4-5")
    timeout = cfg.get("filter_timeout_ms", 12000) / 1000

    lines = [
        f"[{i+1}] {c['text'][:150].replace(chr(10), ' ')}"
        for i, c in enumerate(candidates)
    ]
    candidates_block = "\n".join(lines)

    system = (
        "You are a relevance filter for a memory retrieval system. "
        "Decide which numbered memories are relevant to the query. "
        "Output ONLY a JSON array of integers — no explanation, no other text. "
        "Examples: [1, 4, 7]   or   []"
    )
    user = (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{candidates_block}\n\n"
        "Return a JSON array of the relevant memory numbers."
    )

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 150,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{endpoint}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        raw_output = result["choices"][0]["message"]["content"].strip()
        return _parse_indices(raw_output, candidates)
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", str(e))
        return candidates, f"fallback_unreachable:{reason}"
    except Exception as e:
        return candidates, f"fallback_error:{type(e).__name__}"


def _parse_indices(raw: str, candidates: list[dict]) -> tuple[list[dict], str]:
    """
    Extract a JSON integer array from the model's output.
    Accept only valid ints in range — anything else is silently dropped.
    Falls back to returning all candidates if parsing fails.
    """
    # Strip thinking tokens (<think>...</think>) that some models emit
    if "<think>" in raw:
        end_think = raw.rfind("</think>")
        raw = raw[end_think + 8:].strip() if end_think >= 0 else raw

    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return candidates, "fallback_parse_error"

    try:
        indices = json.loads(raw[start:end])
    except json.JSONDecodeError:
        return candidates, "fallback_parse_error"

    if not isinstance(indices, list):
        return candidates, "fallback_parse_error"

    seen: set[int] = set()
    selected = []
    for idx in indices:
        if isinstance(idx, int) and 1 <= idx <= len(candidates) and idx not in seen:
            seen.add(idx)
            selected.append(candidates[idx - 1])

    return selected, "filter"


def _resolve_filter_endpoint(cfg: dict[str, Any]) -> tuple[str, str]:
    """
    Return (base_url, token) for the filter LLM.

    Priority:
      1. COGITO_FILTER_ENDPOINT + COGITO_FILTER_TOKEN — any OpenAI-compat endpoint
      2. ANTHROPIC_API_KEY — direct api.anthropic.com
    """
    endpoint = cfg.get("filter_endpoint", "")
    token = cfg.get("filter_token", "")
    if endpoint and token:
        return endpoint.rstrip("/"), token

    api_key = cfg.get("anthropic_api_key", "")
    if api_key:
        return "https://api.anthropic.com", api_key

    return "", ""
