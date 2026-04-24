"""
Unit tests for cogito.recall_hybrid — no network, no Ollama, no ChromaDB.

Tests cover the pure-Python components of the hybrid recall pipeline:
  * classify_query router patterns
  * 1-based index parser (accepts valid ints, drops out-of-range)
  * BM25 availability / graceful degradation
  * End-to-end recall_hybrid with a fake ``Memory`` object and
    monkey-patched embedding helper (validates routing + fallback paths
    without any external services).
"""

from __future__ import annotations

import pytest

# Note: cogito.__init__ re-exports recall_hybrid as a function for convenience;
# use the full module path here so we can reach the internal helpers.
from fidelis import recall_hybrid as rh


# ---------------------------------------------------------------------------
# classify_query
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("you told me the auth tokens expire in 3600s", "skip"),
    ("you mentioned a bug in the extraction prompt", "skip"),
    ("remind me what you said earlier", "skip"),
    ("how many days between the two incidents", "llm"),
    ("what was the date of the deploy", "llm"),
    ("most recently shipped feature", "llm"),
    ("how many bugs were tracked in total", "llm"),
    ("what is the recall score for snapshot layer", "default"),
    ("describe the architecture of cogito-ergo", "default"),
])
def test_classify_query(query, expected):
    assert rh.classify_query(query) == expected


# ---------------------------------------------------------------------------
# _parse_indices_1based — no network
# ---------------------------------------------------------------------------

def test_parse_indices_basic():
    assert rh._parse_indices_1based("[1, 3, 2]", 3) == [0, 2, 1]


def test_parse_indices_drops_out_of_range():
    # 0 is not a valid 1-based index; 5 is out of range for 3 candidates.
    assert rh._parse_indices_1based("[0, 1, 5, 2]", 3) == [0, 1]


def test_parse_indices_bad_json_returns_none():
    assert rh._parse_indices_1based("nonsense", 3) is None


def test_parse_indices_empty_returns_none():
    assert rh._parse_indices_1based("[]", 3) is None


def test_parse_indices_strips_think_tokens():
    raw = "<think>let me think</think>[1, 2]"
    assert rh._parse_indices_1based(raw, 3) == [0, 1]


def test_parse_indices_deduplicates():
    assert rh._parse_indices_1based("[1, 1, 2, 2, 3]", 3) == [0, 1, 2]


# ---------------------------------------------------------------------------
# BM25 availability
# ---------------------------------------------------------------------------

def test_bm25_available_returns_bool():
    # Either True or False is fine — we just need graceful behavior.
    assert isinstance(rh._bm25_available(), bool)


def test_bm25_search_handles_missing_index():
    # Passing None for the index should return [] — never raise.
    assert rh._bm25_search(None, "query", 5) == []


# ---------------------------------------------------------------------------
# Fake Memory + monkey-patched embed for end-to-end routing tests
# ---------------------------------------------------------------------------

class FakeMemory:
    """Minimal mem0-compatible stub for unit tests."""

    def __init__(self, memories: list[str]):
        self._memories = memories

    def search(self, query: str, user_id: str, limit: int) -> dict:
        # Deterministic: return all memories with rank by substring match count,
        # capped to ``limit``.
        q_lower = query.lower()
        scored = []
        for m in self._memories:
            score = sum(1 for tok in q_lower.split() if tok in m.lower())
            scored.append((m, score))
        scored.sort(key=lambda x: -x[1])
        return {
            "results": [
                {"memory": m, "score": 100.0 - score}  # lower score = better for mem0
                for m, score in scored[:limit]
            ]
        }


@pytest.fixture
def fake_memory():
    return FakeMemory([
        "auth tokens expire after 3600 seconds",
        "the cogito recall pipeline uses integer pointer filter",
        "mem0 stores memories in chromadb collection cogito_memory",
        "snapshot layer contributes +15% hit@any vs recall-only",
        "nomic-embed-text is the default embedding model",
    ])


@pytest.fixture
def stub_embed(monkeypatch):
    """Patch _embed_docs / _embed_queries to return deterministic unit vectors.

    Each text hashes into a tiny 4-dim vector so cosine similarity is
    well-defined without requiring Ollama.
    """
    def _vec(t: str) -> list[float]:
        h = abs(hash(t))
        return [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(4)]

    def fake_docs(texts, cfg):
        return [_vec(t) for t in texts]

    def fake_queries(texts, cfg):
        return [_vec(t) for t in texts]

    monkeypatch.setattr(rh, "_embed_docs", fake_docs)
    monkeypatch.setattr(rh, "_embed_queries", fake_queries)


# ---------------------------------------------------------------------------
# End-to-end recall_hybrid tests (zero-LLM tier — no network)
# ---------------------------------------------------------------------------

def test_recall_hybrid_zero_llm_returns_results(fake_memory, stub_embed):
    cfg = {"recall_limit": 10, "ollama_url": "http://localhost:11434"}
    results, method = rh.recall_hybrid(
        fake_memory, "cogito recall pipeline integer pointer",
        user_id="agent", cfg=cfg, tier="zero_llm",
    )
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("text" in r and "score" in r for r in results)
    assert "zero_llm" in method or "hybrid" in method


def test_recall_hybrid_skip_route_bypasses_llm(fake_memory, stub_embed):
    cfg = {
        "recall_limit": 10,
        "filter_endpoint": "http://unused.example.com",
        "filter_token": "x",
        "filter_model": "test",
        "ollama_url": "http://localhost:11434",
    }
    # "you told me" triggers the skip router — no LLM call even at filter tier.
    results, method = rh.recall_hybrid(
        fake_memory, "you told me about the auth tokens",
        user_id="agent", cfg=cfg, tier="filter",
    )
    assert "skip_route" in method
    assert len(results) > 0


def test_recall_hybrid_filter_degrades_without_endpoint(fake_memory, stub_embed):
    # No filter_endpoint → graceful fallback to Stage 1 order.
    cfg = {"recall_limit": 10, "ollama_url": "http://localhost:11434"}
    results, method = rh.recall_hybrid(
        fake_memory, "describe snapshot layer behavior",
        user_id="agent", cfg=cfg, tier="filter",
    )
    # Should return Stage 1 results with a degraded method tag — never raise.
    assert len(results) > 0
    # Either default_confident, filter_no_endpoint, or similar fallback.
    assert method  # non-empty


def test_recall_hybrid_flagship_degrades_without_endpoint(fake_memory, stub_embed, monkeypatch):
    # Make sure DashScope key is unset so we test the pure fallback path.
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("COGITO_FLAGSHIP_ENDPOINT", raising=False)
    monkeypatch.delenv("COGITO_FLAGSHIP_TOKEN", raising=False)
    monkeypatch.delenv("COGITO_FLAGSHIP_MODEL", raising=False)

    cfg = {"recall_limit": 10, "ollama_url": "http://localhost:11434"}
    results, method = rh.recall_hybrid(
        fake_memory, "how many bugs were tracked in total",
        user_id="agent", cfg=cfg, tier="flagship",
    )
    # Flagship unavailable → falls back. Must not raise.
    assert len(results) > 0
    assert "flagship_no_endpoint" in method or "filter_no_endpoint" in method


def test_recall_hybrid_empty_store(stub_embed):
    empty_mem = FakeMemory([])
    cfg = {"recall_limit": 10, "ollama_url": "http://localhost:11434"}
    results, method = rh.recall_hybrid(
        empty_mem, "any query",
        user_id="agent", cfg=cfg, tier="zero_llm",
    )
    assert results == []
    assert method == "no_candidates"
