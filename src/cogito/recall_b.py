"""
recall_b — zero-LLM structural query decomposition + multi-query vector fusion.

Strategy:
  1. Strip question scaffolding (what, why, how, describe, etc.) from query.
  2. Extract key tokens by removing stop words.
  3. Expand tokens using vocab_map (plain-English → technical terms bridge).
  4. Generate sub-queries: original → stripped phrase → expansion terms → bigrams → trigrams → tokens.
  5. Run up to MAX_SUBQUERIES vector searches independently.
  6. Merge with Reciprocal Rank Fusion (RRF); deduplicate by text.
  7. Return top-N by merged score.

No LLM at runtime. vocab_map written once by `cogito calibrate`. Target latency: <150ms.
"""

from __future__ import annotations

import re
from typing import Any


# Words that don't contribute to retrieval; stripped before sub-query generation.
_STOP = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall",
    "what", "which", "who", "whom", "whose", "how", "why", "when", "where",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they",
    "this", "that", "these", "those", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "and", "or", "but", "if", "while", "because",
    "so", "not", "no", "nor", "yet", "than", "then", "about",
    # Question scaffolding words — add semantic noise without adding signal
    "cause", "caused", "describe", "explain", "tell", "give", "show",
    "find", "get", "make", "use", "used", "using", "their", "its",
    "did", "was", "were", "been", "being", "can", "cannot",
})

# Minimum token length to include as an individual sub-query.
_MIN_TOKEN_LEN = 4

# Max number of sub-queries to issue against the vector store.
MAX_SUBQUERIES = 8

# RRF constant — k=60 is empirically stable across retrieval tasks.
_RRF_K = 60


def _tokenize(text: str) -> list[str]:
    """Lower-case, remove punctuation, split on whitespace."""
    return re.sub(r"[^\w\s-]", " ", text.lower()).split()


def _key_tokens(text: str) -> list[str]:
    """Return tokens that aren't stop words, preserving order."""
    return [t for t in _tokenize(text) if t not in _STOP and len(t) >= 2]


def _expand_with_vocab_map(
    tokens: list[str],
    query: str,
    vocab_map: dict[str, list[str]],
) -> list[str]:
    """
    Return additional search terms from vocab_map.

    Checks individual tokens, adjacent bigrams from key tokens, and
    substrings of the query against vocab_map keys (case-insensitive).
    Returns deduplicated expansion terms (each term separately, not joined).
    """
    expansion: list[str] = []
    seen: set[str] = set()
    query_lower = query.lower()

    def _add_terms(terms: list[str]) -> None:
        for t in terms:
            t = str(t).strip()
            if t and t not in seen:
                seen.add(t)
                expansion.append(t)

    # Check individual tokens
    for tok in tokens:
        if tok in vocab_map:
            _add_terms(vocab_map[tok])

    # Check bigrams of key tokens
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i+1]}"
        if bigram in vocab_map:
            _add_terms(vocab_map[bigram])

    # Check any vocab_map key that appears as a substring in the query
    for key, terms in vocab_map.items():
        if key in query_lower and key not in {t for t in tokens}:
            _add_terms(terms)

    return expansion


def _build_subqueries(
    query: str,
    vocab_map: dict[str, list[str]] | None = None,
) -> tuple[list[str], bool]:
    """
    Generate up to MAX_SUBQUERIES search strings from query.

    Priority order:
      1. Original query (always first — preserves full semantic vector)
      2. Stripped key-token phrase (stop words removed)
      3. Vocabulary expansion terms (if vocab_map provided and matches found)
      4. Adjacent bigrams of key tokens
      5. All adjacent trigrams (for longer queries)
      6. Individual key tokens (length >= MIN_TOKEN_LEN)

    Returns (subqueries, expanded) where expanded=True if vocab expansion was used.
    """
    seen: set[str] = set()
    result: list[str] = []

    def _add(q: str) -> None:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            result.append(q)

    _add(query)

    tokens = _key_tokens(query)
    if not tokens:
        return result, False

    # Stripped phrase
    _add(" ".join(tokens))

    # Vocab expansion terms (inserted before bigrams for priority)
    expanded = False
    if vocab_map:
        expansion_terms = _expand_with_vocab_map(tokens, query, vocab_map)
        for term in expansion_terms:
            _add(term)
        expanded = bool(expansion_terms)

    # All adjacent bigrams
    for i in range(len(tokens) - 1):
        _add(f"{tokens[i]} {tokens[i+1]}")

    # All adjacent trigrams (if query is long enough)
    for i in range(len(tokens) - 2):
        _add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

    # Individual significant tokens
    for t in tokens:
        if len(t) >= _MIN_TOKEN_LEN:
            _add(t)

    return result[:MAX_SUBQUERIES], expanded


def _rrf_merge(
    runs: list[list[dict]],
    limit: int,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Each run is a list of {"text": str, "score": float} ordered best-first.
    Returns a unified list ordered by RRF score, deduplicated by text.
    """
    scores: dict[str, float] = {}
    canonical: dict[str, dict] = {}  # text → first-seen result dict

    for run in runs:
        for rank, item in enumerate(run, 1):
            text = item.get("text", "")
            if not text:
                continue
            scores[text] = scores.get(text, 0.0) + 1.0 / (_RRF_K + rank)
            if text not in canonical:
                canonical[text] = item

    merged = sorted(canonical.values(), key=lambda x: scores[x["text"]], reverse=True)
    return merged[:limit]


def recall_b(
    memory: Any,
    query: str,
    user_id: str,
    cfg: dict[str, Any],
    limit: int | None = None,
) -> tuple[list[dict], str]:
    """
    Zero-LLM recall.  Returns (memories, method).

    memories: list of {"text": str, "score": float}
    method:   "decompose_N" or "decompose_N_v" (v = vocab expansion applied)
    """
    limit = limit or cfg.get("recall_limit", 50)
    per_query_limit = min(limit, 20)  # per-sub-query cap; RRF widens coverage
    vocab_map: dict[str, list[str]] = cfg.get("vocab_map", {})

    subqueries, expanded = _build_subqueries(query, vocab_map if vocab_map else None)
    runs: list[list[dict]] = []

    for sq in subqueries:
        raw = memory.search(sq, user_id=user_id, limit=per_query_limit)
        candidates = [
            {"text": r.get("memory", ""), "score": round(r.get("score", 9999), 3)}
            for r in raw.get("results", [])
            if r.get("memory")
        ]
        if candidates:
            runs.append(candidates)

    if not runs:
        return [], "no_candidates"

    merged = _rrf_merge(runs, limit=limit)
    suffix = "_v" if expanded else ""
    method = f"decompose_{len(runs)}{suffix}"
    return merged, method
