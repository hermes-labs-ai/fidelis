"""
cogito recall — two-stage retrieval with integer-pointer fidelity filter.

Stage 1: broad vector search against the memory store
Stage 2: cheap LLM receives numbered candidate list, outputs integer indices ONLY
Stage 3: server picks verbatim text from candidates using those indices

The cheap LLM never outputs memory text — only integers — so it cannot
corrupt, summarize, or hallucinate into the content returned to the caller.
This is a structural guarantee, not a prompt engineering hope.

Callable as a library:
    from cogito.recall import recall
    memories = recall(memory, query="...", cfg=cfg)

Or hit the HTTP endpoint:
    POST /recall  {"text": "...", "limit": 50, "threshold": 400}
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
    threshold: float | None = None,
) -> tuple[list[dict], str]:
    """
    Run two-stage recall. Returns (memories, method).

    memories: list of {"text": str, "score": float}
    method:   "filter" | "fallback_*" (fallback = no filter applied)
    """
    limit = limit or cfg.get("recall_limit", 50)
    threshold = threshold or cfg.get("recall_threshold", 400.0)

    # Stage 1 — broad search
    raw = memory.search(query, user_id=user_id, limit=min(limit, 100))
    candidates = [
        {"text": r.get("memory", ""), "score": round(r.get("score", 9999), 3)}
        for r in raw.get("results", [])
        if r.get("memory") and r.get("score", 9999) < threshold
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
    Returns (selected, method).
    """
    endpoint, token = _resolve_filter_endpoint(cfg)
    if not endpoint:
        return candidates, "fallback_no_endpoint"

    model = cfg.get("filter_model", "anthropic/claude-haiku-4-5")
    timeout = cfg.get("filter_timeout_ms", 12000) / 1000

    lines = [f"[{i+1}] {c['text'][:150].replace(chr(10), ' ')}" for i, c in enumerate(candidates)]
    candidates_block = "\n".join(lines)

    system = (
        "You are a relevance filter for a memory retrieval system. "
        "Your only job: decide which numbered memories are relevant to the query. "
        "Output ONLY a JSON array of integers — nothing else, no explanation. "
        "Examples: [1, 4, 7]   or   []"
    )
    user = (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{candidates_block}\n\n"
        "Return only a JSON array of the relevant memory numbers."
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
        return candidates, f"fallback_unreachable:{e.reason if hasattr(e, 'reason') else e}"
    except Exception as e:
        return candidates, f"fallback_error:{type(e).__name__}"


def _parse_indices(raw: str, candidates: list[dict]) -> tuple[list[dict], str]:
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

    Tries in order:
      1. COGITO_FILTER_ENDPOINT + COGITO_FILTER_TOKEN  (explicit, any OpenAI-compat endpoint)
      2. ANTHROPIC_API_KEY  → direct api.anthropic.com
    """
    endpoint = cfg.get("filter_endpoint", "")
    token = cfg.get("filter_token", "")

    if endpoint and token:
        return endpoint.rstrip("/"), token

    api_key = cfg.get("anthropic_api_key", "")
    if api_key:
        return "https://api.anthropic.com", api_key

    return "", ""
