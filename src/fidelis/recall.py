"""
cogito recall — two-stage retrieval with integer-pointer fidelity filter.

Stage 1: recall_b (zero-LLM multi-query RRF) — broad candidate pool, no threshold cut.
         Uses sub-query decomposition + vocab expansion to maximise recall.
         Replacing single memory.search() here removes the recall ceiling caused
         by a single query's embedding being far from the stored fact's embedding.

Stage 2: cheap LLM receives numbered candidate list, outputs integer indices ONLY.
         Never outputs memory text — structurally cannot corrupt or hallucinate
         into the content returned to the caller.

Stage 3: server picks verbatim candidate text by those indices and returns it.

Callable as a library:
    from fidelis.recall import recall
    memories, method = recall(memory, "query", user_id="agent", cfg=cfg)

Or hit the HTTP endpoint:
    POST /recall  {"text": "...", "limit": 50}
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from fidelis.recall_b import recall_b as _stage1


def recall(
    memory: Any,
    query: str,
    user_id: str,
    cfg: dict[str, Any],
    limit: int | None = None,
    since: str | None = None,
) -> tuple[list[dict], str]:
    """
    Run two-stage recall. Returns (memories, method).

    memories: list of {"text": str, "score": float, "created_at": str (optional)}
    method:   "filter" | "fallback_*" | "<stage1_method>|<fallback>" (may include "|since_filter" if since is applied)
    since:    ISO 8601 date string to filter memories created after this date
    """
    limit = limit or cfg.get("recall_limit", 50)

    # Stage 1 — broad multi-query search via recall_b (zero-LLM).
    # recall_b issues multiple sub-queries and merges with RRF.
    # No threshold cut — Stage 2 handles precision.
    # Using recall_b instead of single memory.search() removes the single-query
    # recall ceiling: the right memory can fail one query vector but surface in another.
    candidates, stage1_method = _stage1(
        memory, query, user_id=user_id, cfg=cfg, limit=min(limit, 100)
    )

    if not candidates:
        return [], "no_candidates"

    # Stage 2 — integer-pointer filter
    selected, filter_method = _filter(query, candidates, cfg)

    # Stage 3 — optional timestamp filter
    if since and selected:
        selected, since_applied = _filter_by_since(selected, since)
        if since_applied:
            filter_method = f"{filter_method}|since_filter" if filter_method != "filter" else "since_filter"

    # Method tag: just "filter" when clean, compound when fallback
    method = f"{stage1_method}|{filter_method}" if filter_method != "filter" else "filter"
    return selected, method


def _filter(
    query: str,
    candidates: list[dict],
    cfg: dict[str, Any],
) -> tuple[list[dict], str]:
    """
    Ask the filter model which candidates are relevant.
    Outputs only integer indices — the model never generates memory text.

    Supports two call paths:
      - OpenAI-compat (/v1/chat/completions): cloud APIs, non-thinking local models
      - Native Ollama (/api/chat with think:false): qwen3 thinking models
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
        "Output ONLY a JSON array of integers, ordered from most to least relevant. "
        "If the query is off-topic or none of the candidates are relevant, output []. "
        "No explanation, no other text. "
        "Examples: [1, 4, 7]   or   []"
    )
    user = (
        f"Query: {query}\n\n"
        f"Candidate memories:\n{candidates_block}\n\n"
        "Return a JSON array of the relevant memory numbers."
    )

    # Qwen3 thinking models (qwen3, qwen3.5) need the native Ollama API to
    # disable thinking mode — /v1/chat/completions returns empty content for these.
    is_ollama_local = "localhost:11434" in endpoint or "127.0.0.1:11434" in endpoint
    is_thinking_model = model.startswith("qwen3") or model.startswith("qwen3.5")

    if is_ollama_local and is_thinking_model:
        return _filter_ollama_native(query, model, system, user, candidates, endpoint, timeout)

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


def _filter_ollama_native(
    query: str,
    model: str,
    system: str,
    user: str,
    candidates: list[dict],
    endpoint: str,
    timeout: float,
) -> tuple[list[dict], str]:
    """
    Filter using native Ollama /api/chat with think:false.
    Required for qwen3/qwen3.5 thinking models which return empty content
    via the OpenAI-compat endpoint.
    """
    # Derive base URL (strip /v1 if present, point at ollama native)
    base = endpoint.split("/v1")[0].rstrip("/")
    # Ollama native endpoint is always localhost:11434 or the configured host
    if "11434" not in base:
        base = "http://localhost:11434"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "think": False,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        raw_output = result["message"]["content"].strip()
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


def _filter_by_since(
    memories: list[dict],
    since: str,
) -> tuple[list[dict], bool]:
    """
    Filter memories to only include those created on or after 'since' date.
    since: ISO 8601 date string (e.g., "2026-04-01" or "2026-04-01T12:00:00Z").

    Returns (filtered_memories, was_filter_applied).
    was_filter_applied=False if no memories have created_at field or if date parsing fails.
    """

    try:
        since_dt = _parse_iso_date(since)
    except ValueError:
        return memories, False

    filtered = []
    for m in memories:
        created_at = m.get("created_at")
        if not created_at:
            continue
        try:
            mem_dt = _parse_iso_date(created_at)
            if mem_dt >= since_dt:
                filtered.append(m)
        except ValueError:
            continue

    return filtered if filtered else memories, len(filtered) > 0


def _parse_iso_date(date_str: str) -> object:
    """Parse ISO 8601 date string to datetime object."""
    from datetime import datetime

    date_str = date_str.strip()
    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}")


def _resolve_filter_endpoint(cfg: dict[str, Any]) -> tuple[str, str]:
    """
    Return (base_url, token) for the filter LLM.

    Requires COGITO_FILTER_ENDPOINT + COGITO_FILTER_TOKEN (any OpenAI-compat endpoint).
    For Ollama, set endpoint to http://localhost:11434 with an empty token.
    """
    endpoint = cfg.get("filter_endpoint", "")
    token = cfg.get("filter_token", "")
    if endpoint:
        return endpoint.rstrip("/"), token

    return "", ""
