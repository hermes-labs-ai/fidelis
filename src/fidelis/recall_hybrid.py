"""
recall_hybrid — BM25 + dense + RRF retrieval with optional tiered LLM escalation.

Default tier is ``zero_llm`` (83.2% R@1 on LongMemEval_S, $0/query, ~90 ms,
fully local). The LLM tiers (``filter`` and ``flagship``) are benchmark-tuned
and opt-in: they port the architecture that reached 96.4% R@1 at flagship
tier (runP-v35, 2026-04-18, 470 questions) but currently escalate on ~80%
of queries versus the 10% the threshold was designed for — see
``docs/RELEASE-SCOPE.md`` and ``docs/THRESHOLD-AUDIT.md``. Use the LLM
tiers for benchmark replication or hard-query lookups; do not base a
production cost model on them yet.

Adaptation to production data shape
-----------------------------------
The benchmark operated over LongMemEval sessions (multi-turn dialogs with
turn-level chunking and session-date metadata). Production cogito stores
atomic memory texts — there are no session boundaries, no turns, no
haystack_dates. The components that port cleanly:

  1. BM25 + nomic dense + Reciprocal Rank Fusion over candidates.
  2. ``search_query:`` / ``search_document:`` prefix wrapping for nomic-embed-text.
  3. Regex query classifier → 2-way router (``skip`` / ``llm`` / default).
  4. Two-tier LLM escalation: cheap filter → flagship rerank on uncertainty.

Components that do NOT port (and are omitted):

  * Turn-level chunking (memories are already atomic units).
  * Session-date temporal scaffold (no session grouping in the store).
  * hardset qid list (LongMemEval-specific).

Tiers (opt-in, graceful degradation)
------------------------------------
* ``tier="zero_llm"`` — BM25+dense+RRF only. No LLM calls.
* ``tier="filter"``   — Above + cheap filter LLM (integer pointers, existing path).
* ``tier="flagship"`` — Above + flagship rerank on low-confidence cases.

If the flagship endpoint / API key is missing, the function transparently
falls back to the filter tier. If the filter endpoint is missing, it falls
back to zero_llm. Nothing crashes when keys are absent.

See :func:`recall_hybrid` for the public entry point.
"""

from __future__ import annotations

import json as _json
import os
import urllib.error
import urllib.request
from typing import Any

from fidelis.recall_b import (
    _build_subqueries,
    _cosine_sim,
    MAX_SUBQUERIES,
)

# ---------------------------------------------------------------------------
# Constants (mirrored from longmemeval_combined_pipeline_flagship.py)
# ---------------------------------------------------------------------------
_RRF_K = 60
# Cosine weight blends RRF rank with direct cosine similarity. 0.7 matched
# the LongMemEval benchmark. On production ``mem0`` stores the best setting
# depends on corpus density — override via cfg["hybrid_cosine_weight"] or
# ``COGITO_HYBRID_COSINE_WEIGHT``.
_COSINE_WEIGHT = 0.7
_QUERY_PREFIX = "search_query: "
_DOC_PREFIX = "search_document: "
# calibrated: 0.1 gap threshold from LongMemEval_S score distributions; fires ~15%
# of queries (escalates to flagship when top-1 and top-2 are too close to trust).
_GAP_THRESHOLD = 0.1  # top1 − top2 score gap; above = confident, below = escalate
_EMBED_MODEL = "nomic-embed-text"
_DEFAULT_FILTER_TOP_K = 5
_FILTER_SNIPPET_CHARS = 500
_FLAGSHIP_SNIPPET_CHARS = 2000

# Query classifier patterns (verbatim from flagship pipeline)
_SKIP_PATTERNS = [
    "you told me", "you suggested", "you recommended",
    "you mentioned", "you said", "you explained",
    "our previous conversation", "our last chat",
    "remind me what you",
]
_TEMPORAL_PATTERNS = [
    "which happened first", "which did i do first",
    "how many days", "how many weeks", "how many months",
    "before or after", "what order", "what was the date",
    "which event", "which trip",
    "order of the", "from earliest", "from first",
    "most recently", "a week ago", "two weeks ago",
    "a month ago", "last saturday", "last sunday",
    "last weekend", "last monday", "last tuesday",
    "graduated first", "started first", "finished first",
    "did i do first", "did i attend first",
]
_COUNTING_PATTERNS = [
    "how many", "total number", "how much total",
    "in total", "altogether", "combined",
]

_FILTER_SYSTEM = (
    "Execute this procedure:\n"
    "```\n"
    "def rank(query, candidates):\n"
    "  for each candidate:\n"
    "    keyword_match = count(query_keywords IN candidate_text)\n"
    "    fact_match = candidate contains specific answer to query (bool)\n"
    "    score = keyword_match * 2 + fact_match * 10\n"
    "  return sorted(candidates, by=score, descending)\n"
    "```\n"
    "Input: query and numbered candidates.\n"
    "Output: ONLY a JSON array of candidate numbers sorted by score.\n"
    "Example: [3, 1, 5, 2, 4]"
)


# ---------------------------------------------------------------------------
# Query classifier
# ---------------------------------------------------------------------------
def classify_query(query: str) -> str:
    """Classify a query for routing.

    Returns one of:
      * ``"skip"``    — assistant-reference queries ("you told me …").
                        LLM reranking empirically hurts — keep Stage 1 order.
      * ``"llm"``     — temporal or counting queries. LLM filter always helps.
      * ``"default"`` — every other query. Use Stage 1 order unless the
                        confidence gap is small (then escalate).
    """
    q = query.lower()
    if any(p in q for p in _SKIP_PATTERNS):
        return "skip"
    if any(p in q for p in _TEMPORAL_PATTERNS):
        return "llm"
    if any(p in q for p in _COUNTING_PATTERNS):
        return "llm"
    return "default"


# ---------------------------------------------------------------------------
# BM25 helpers (bm25s is optional; fall back to pure-dense if missing)
# ---------------------------------------------------------------------------
def _bm25_available() -> bool:
    """Return True if the ``bm25s`` package is importable."""
    try:
        import bm25s  # noqa: F401
        return True
    except Exception:
        return False


def _bm25_index(corpus_texts: list[str]):
    """Build a BM25 index for ``corpus_texts``. Returns the index or None."""
    try:
        import bm25s
        os.environ.setdefault("BM25S_SHOW_PROGRESS", "0")
        os.environ.setdefault("TQDM_DISABLE", "1")
        tokens = bm25s.tokenize(corpus_texts, show_progress=False)
        index = bm25s.BM25()
        index.index(tokens, show_progress=False)
        return index
    except Exception:
        return None


def _bm25_search(index, query: str, k: int) -> list[tuple[int, float]]:
    """Run a BM25 query. Returns list of (doc_idx, score)."""
    try:
        import bm25s
        q_tokens = bm25s.tokenize([query], show_progress=False)
        docs, scores = index.retrieve(q_tokens, k=k, show_progress=False)
        return [(int(i), float(s)) for i, s in zip(docs[0], scores[0])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Prefixed embeddings (nomic convention)
# ---------------------------------------------------------------------------
def _embed_prefixed(texts: list[str], cfg: dict[str, Any]) -> list[list[float]] | None:
    """Call Ollama /api/embed with texts already wrapped in their prefix."""
    url = cfg.get("ollama_url", "http://localhost:11434")
    model = cfg.get("embed_model", _EMBED_MODEL)
    sanitized = [t[:2000] if t.strip() else "empty" for t in texts]
    try:
        body = _json.dumps({"model": model, "input": sanitized}).encode()
        req = urllib.request.Request(
            f"{url}/api/embed", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        vecs = data.get("embeddings", [])
        return vecs if len(vecs) == len(texts) else None
    except Exception:
        return None


def _embed_docs(texts: list[str], cfg: dict[str, Any]) -> list[list[float]] | None:
    return _embed_prefixed([_DOC_PREFIX + t for t in texts], cfg)


def _embed_queries(texts: list[str], cfg: dict[str, Any]) -> list[list[float]] | None:
    return _embed_prefixed([_QUERY_PREFIX + t for t in texts], cfg)


# ---------------------------------------------------------------------------
# Stage 1: BM25 + dense + RRF hybrid retrieval
# ---------------------------------------------------------------------------
def _hybrid_stage1(
    memory: Any,
    query: str,
    user_id: str,
    cfg: dict[str, Any],
    pool_size: int,
) -> tuple[list[dict], list[float], str]:
    """Stage 1 hybrid retrieval over the existing mem0 store.

    Strategy:
      1. Pull a broad candidate pool via mem0.search() across sub-queries (reuses
         recall_b's decomposition logic so vocab_map still applies).
      2. Build an in-memory BM25 index over the candidate texts.
      3. For each sub-query, take dense-cosine top-k and BM25 top-k on the
         candidate pool. Fuse with RRF across all runs.
      4. Blend RRF with cosine similarity against the prefixed original query.

    Returns (ranked_candidates, ranked_scores, method).
    """
    vocab_map: dict[str, list[str]] = cfg.get("vocab_map") or {}
    subqueries, expanded = _build_subqueries(query, vocab_map if vocab_map else None)

    # --- Build candidate pool from mem0 (same as recall_b) ---
    per_query_limit = min(pool_size, 20)
    seen_texts: set[str] = set()
    pool: list[dict] = []
    # Per-sub-query ranked lists, for RRF downstream (by text)
    runs_by_text: list[list[str]] = []
    for sq in subqueries:
        raw = memory.search(sq, user_id=user_id, limit=per_query_limit)
        run_texts: list[str] = []
        for r in raw.get("results", []):
            text = r.get("memory", "")
            if not text:
                continue
            run_texts.append(text)
            if text not in seen_texts:
                seen_texts.add(text)
                pool.append({"text": text, "score": round(r.get("score", 9999), 3)})
        if run_texts:
            runs_by_text.append(run_texts)

    if not pool:
        return [], [], "no_candidates"

    # Cap pool (mem0 already de-duplicated, but bound worst case)
    pool = pool[:pool_size]
    pool_texts = [c["text"] for c in pool]

    # --- Prefixed embeddings ---
    doc_vecs = _embed_docs(pool_texts, cfg)
    q_vecs = _embed_queries([query] + subqueries[:MAX_SUBQUERIES], cfg)

    # If embeddings failed, fall back to mem0 order (graceful)
    if doc_vecs is None or q_vecs is None:
        method = f"hybrid_fallback_no_embed_{len(runs_by_text)}"
        return pool, [1.0] * len(pool), method

    query_vec = q_vecs[0]
    sq_vecs = q_vecs[1:]

    # --- BM25 index on the candidate pool ---
    bm25 = _bm25_index(pool_texts) if _bm25_available() else None

    # --- Collect RRF runs ---
    runs: list[list[int]] = []

    # Run 1 per sub-query: dense cosine top-k
    k = min(20, len(pool))
    for sv in sq_vecs:
        scored = [(i, _cosine_sim(sv, dv)) for i, dv in enumerate(doc_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        runs.append([i for i, _ in scored[:k]])

    # Run 1 per sub-query: BM25 top-k
    if bm25 is not None:
        for sq in subqueries:
            bm25_hits = _bm25_search(bm25, sq, k)
            runs.append([i for i, _ in bm25_hits])

    # Run: mem0's own ranking (per sub-query, mapped back to pool indices)
    text_to_idx = {t: i for i, t in enumerate(pool_texts)}
    for run_texts in runs_by_text:
        runs.append([text_to_idx[t] for t in run_texts if t in text_to_idx])

    # --- RRF merge ---
    rrf: dict[int, float] = {}
    for run in runs:
        for rank, idx in enumerate(run, 1):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (_RRF_K + rank)

    # Blend with cosine to original query
    cosine_scores = {i: _cosine_sim(query_vec, doc_vecs[i]) for i in range(len(pool))}
    rrf_max = max(rrf.values()) if rrf else 1.0
    cosine_weight = float(cfg.get("hybrid_cosine_weight", _COSINE_WEIGHT))

    blended: list[tuple[int, float]] = []
    for i in range(len(pool)):
        rrf_n = rrf.get(i, 0.0) / rrf_max if rrf_max > 0 else 0.0
        cos = cosine_scores.get(i, 0.0)
        blended.append((i, (1.0 - cosine_weight) * rrf_n + cosine_weight * cos))

    blended.sort(key=lambda x: x[1], reverse=True)
    ranked = [pool[i] for i, _ in blended]
    scores = [s for _, s in blended]

    # Annotate scores with cosine (more interpretable than blended)
    for item, idx_and_score in zip(ranked, blended):
        item["score"] = round(cosine_scores.get(idx_and_score[0], 0.0), 4)

    suffix = "_v" if expanded else ""
    bm_tag = "_bm25" if bm25 is not None else "_nobm25"
    method = f"hybrid_{len(runs)}{bm_tag}{suffix}"
    return ranked, scores, method


# ---------------------------------------------------------------------------
# Stage 2: cheap filter LLM (reuses cogito.recall._filter for consistency)
# ---------------------------------------------------------------------------
def _parse_indices_1based(raw: str, n_candidates: int) -> list[int] | None:
    """Parse an LLM output to a list of 0-based indices (drop out-of-range)."""
    if "<think>" in raw:
        end = raw.rfind("</think>")
        if end >= 0:
            after = raw[end + 8:].strip()
            if after:
                raw = after
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return None
    try:
        arr = _json.loads(raw[start:end])
    except _json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    result: list[int] = []
    for item in arr:
        try:
            idx = int(item)
        except (ValueError, TypeError):
            continue
        if 1 <= idx <= n_candidates and (idx - 1) not in result:
            result.append(idx - 1)
    return result if result else None


def _filter_rerank(
    query: str,
    candidates: list[dict],
    cfg: dict[str, Any],
    top_k: int,
    snippet_chars: int,
) -> tuple[list[dict], str]:
    """Cheap filter: rerank top_k candidates via the OpenAI-compatible filter endpoint.

    Mirrors the flagship pipeline's qwen-turbo rerank but routes through
    cogito's existing filter config (``COGITO_FILTER_ENDPOINT`` + token +
    model). Returns (reranked, method). On any failure, returns candidates
    unchanged with a ``fallback_*`` method tag.
    """
    endpoint = (cfg.get("filter_endpoint") or "").rstrip("/")
    token = cfg.get("filter_token", "")
    model = cfg.get("filter_model", "")
    if not endpoint or not model:
        return candidates, "filter_no_endpoint"

    top = candidates[:top_k]
    if len(top) <= 1:
        return candidates, "filter_skip"

    lines = [
        f"[{i+1}] {c['text'][:snippet_chars].replace(chr(10), ' ')}"
        for i, c in enumerate(top)
    ]
    user_msg = (
        f"Query: {query}\n\n"
        f"Candidate memories:\n" + "\n".join(lines) + "\n\n"
        f"Rank these {len(top)} candidates by relevance. Output JSON array."
    )

    body = _json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _FILTER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": 100,
    }).encode()
    timeout = cfg.get("filter_timeout_ms", 12000) / 1000

    try:
        req = urllib.request.Request(
            f"{endpoint}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}" if token else "",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read())
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", str(e))
        return candidates, f"filter_unreachable:{reason}"
    except Exception as e:
        return candidates, f"filter_error:{type(e).__name__}"

    parsed = _parse_indices_1based(raw, len(top))
    if parsed is None:
        return candidates, "filter_parse_fail"

    reranked_top = [top[i] for i in parsed]
    seen = set(parsed)
    for i in range(len(top)):
        if i not in seen:
            reranked_top.append(top[i])

    return reranked_top + candidates[top_k:], "filter"


# ---------------------------------------------------------------------------
# Stage 3: flagship rerank (opt-in, cloud model with larger context)
# ---------------------------------------------------------------------------
def _flagship_rerank(
    query: str,
    candidates: list[dict],
    cfg: dict[str, Any],
    top_k: int,
) -> tuple[list[dict], str]:
    """Flagship rerank: larger snippets, stronger model.

    Reads credentials from (in priority order):
      1. ``COGITO_FLAGSHIP_ENDPOINT`` + ``COGITO_FLAGSHIP_TOKEN`` + ``COGITO_FLAGSHIP_MODEL``
      2. ``DASHSCOPE_API_KEY`` (uses DashScope intl endpoint with qwen-max)

    On any missing credential or network failure, returns candidates
    unchanged with a ``flagship_*`` method tag — the caller can then fall
    back to the filter tier or zero-LLM order.
    """
    endpoint = cfg.get("flagship_endpoint") or os.environ.get("COGITO_FLAGSHIP_ENDPOINT", "")
    token = cfg.get("flagship_token") or os.environ.get("COGITO_FLAGSHIP_TOKEN", "")
    model = cfg.get("flagship_model") or os.environ.get("COGITO_FLAGSHIP_MODEL", "")

    if not endpoint and os.environ.get("DASHSCOPE_API_KEY"):
        endpoint = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        token = os.environ["DASHSCOPE_API_KEY"]
        model = model or "qwen-max"

    if not endpoint or not model:
        return candidates, "flagship_no_endpoint"

    top = candidates[:top_k]
    if len(top) <= 1:
        return candidates, "flagship_skip"

    lines = [
        f"[{i+1}] {c['text'][:_FLAGSHIP_SNIPPET_CHARS].replace(chr(10), ' ')}"
        for i, c in enumerate(top)
    ]
    user_msg = (
        f"Query: {query}\n\n"
        f"Candidate memories:\n" + "\n".join(lines) + "\n\n"
        f"Rank these {len(top)} candidates by relevance. Output JSON array."
    )

    body = _json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": _FILTER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": 100,
    }).encode()
    timeout = cfg.get("flagship_timeout_ms", 30000) / 1000

    try:
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}" if token else "",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = _json.loads(resp.read())
        raw = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", str(e))
        return candidates, f"flagship_unreachable:{reason}"
    except Exception as e:
        return candidates, f"flagship_error:{type(e).__name__}"

    parsed = _parse_indices_1based(raw, len(top))
    if parsed is None:
        return candidates, "flagship_parse_fail"

    reranked_top = [top[i] for i in parsed]
    seen = set(parsed)
    for i in range(len(top)):
        if i not in seen:
            reranked_top.append(top[i])

    return reranked_top + candidates[top_k:], "flagship"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def recall_hybrid(
    memory: Any,
    query: str,
    user_id: str,
    cfg: dict[str, Any],
    limit: int | None = None,
    tier: str = "zero_llm",
    top_k: int | None = None,
) -> tuple[list[dict], str]:
    """BM25 + dense + RRF hybrid recall with optional tiered LLM escalation.

    Default tier is ``zero_llm`` (83.2% R@1 on LongMemEval_S at $0/query).
    The ``filter`` and ``flagship`` tiers are benchmark-tuned and experimental.
    It routes by query type, fuses BM25 and dense
    retrieval with RRF, and optionally reranks the top candidates through
    a cheap filter LLM or a flagship cloud model.

    Parameters
    ----------
    memory:
        mem0 ``Memory`` instance.
    query:
        User query string.
    user_id:
        Memory namespace.
    cfg:
        cogito config dict (from ``cogito.config.load()``).
    limit:
        Max candidates to return (defaults to ``cfg['recall_limit']``).
    tier:
        One of:
          * ``"zero_llm"``  — BM25+dense+RRF only, no LLM.
          * ``"filter"``    — + cheap filter on the top candidates for temporal
                              / counting queries only (default). Default-route
                              queries stay on the Stage-1 ranking because
                              reranking at small snippet sizes tends to hurt
                              rank-1 on domain queries (empirically measured).
          * ``"flagship"``  — + flagship rerank on low-confidence cases when
                              a flagship endpoint is configured.
    top_k:
        Number of top candidates shown to the reranker (default 5). The rest
        of the pool keeps its Stage 1 order after the reranked block.

    Returns
    -------
    (memories, method)
        ``memories``: list of ``{"text": str, "score": float}`` ordered best-first.
        ``method``:   pipeline tag (e.g. ``"hybrid_12_bm25|filter"``,
                      ``"hybrid_6_bm25_v|flagship"``,
                      ``"hybrid_8_nobm25|skip_route"``).

    Example
    -------
    >>> from fidelis.config import load, mem0_config
    >>> from fidelis.recall_hybrid import recall_hybrid
    >>> from mem0 import Memory
    >>> cfg = load()
    >>> mem = Memory.from_config(mem0_config(cfg))
    >>> hits, method = recall_hybrid(
    ...     mem, "auth architecture decisions",
    ...     user_id=cfg["user_id"], cfg=cfg, tier="filter",
    ... )
    >>> for h in hits[:3]:
    ...     print(h["text"][:80])
    """
    limit = limit or cfg.get("recall_limit", 50)
    top_k = top_k or _DEFAULT_FILTER_TOP_K

    # --- Stage 1: hybrid retrieval ---
    ranked, scores, s1_method = _hybrid_stage1(
        memory, query, user_id, cfg, pool_size=min(limit, 100),
    )
    if not ranked:
        return [], s1_method

    # --- Router (matches the flagship benchmark pipeline) ---
    route = classify_query(query)

    if tier == "zero_llm" or route == "skip":
        # Zero-LLM or skip route: Stage 1 order is authoritative. The skip
        # route empirically hurts when reranked (assistant-reference queries).
        method = f"{s1_method}|{'skip_route' if route == 'skip' else 'zero_llm'}"
        return ranked[:limit], method

    # --- Confidence gap (top1 vs top2 after cosine blend) ---
    top1 = scores[0] if scores else 0.0
    top2 = scores[1] if len(scores) > 1 else 0.0
    confident = (top1 - top2) > _GAP_THRESHOLD

    # Default route: keep Stage 1 order at ``filter`` tier (mirrors the
    # benchmark pipeline's "default_s1" policy — reranking default queries
    # at small snippet sizes tends to hurt rank-1 on domain queries).
    # Only the ``llm`` route (temporal / counting) gets the cheap filter.
    if route == "default" and tier == "filter":
        method = f"{s1_method}|default_s1"
        return ranked[:limit], method

    # --- Stage 2: cheap filter (runs on route=='llm', or tier=='flagship') ---
    filtered, f_method = _filter_rerank(
        query, ranked, cfg, top_k=top_k, snippet_chars=_FILTER_SNIPPET_CHARS,
    )

    # --- Stage 3: flagship escalation (only at tier=='flagship') ---
    # Escalate when the cheap filter failed or stage-1 confidence is low.
    needs_flagship = (
        tier == "flagship"
        and (not confident or f_method != "filter")
    )

    if needs_flagship:
        flagship_result, fl_method = _flagship_rerank(query, filtered, cfg, top_k=top_k)
        if fl_method == "flagship":
            method = f"{s1_method}|{f_method}|{fl_method}"
            return flagship_result[:limit], method
        # Flagship failed → keep filter output (or Stage 1 if filter failed too)
        method = f"{s1_method}|{f_method}|{fl_method}"
        return filtered[:limit], method

    method = f"{s1_method}|{f_method}"
    return filtered[:limit], method


# Re-export the core helpers the tests rely on.
__all__ = [
    "recall_hybrid",
    "classify_query",
    "_parse_indices_1based",
]
