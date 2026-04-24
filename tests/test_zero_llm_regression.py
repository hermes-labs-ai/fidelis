"""Zero-LLM tier regression tests.

The zero-LLM tier is the v0.3.1 default and the release-surface moat:
83.2% R@1 on LongMemEval_S at $0/query, 90ms, fully local. These tests pin
invariants of that path so future changes don't silently regress default
behavior.

Scope pinned:
  - Default-tier call (no `tier` arg) MUST NOT invoke filter or flagship.
  - tier="zero_llm" MUST succeed with no filter_endpoint / filter_token
    configured.
  - Explicit tier="zero_llm" and default call MUST return identical output.
  - Empty corpus MUST return [] not raise.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from fidelis import recall_hybrid as rh


class FakeMemory:
    def __init__(self, memories: list[str]):
        self._memories = memories

    def search(self, query: str, user_id: str, limit: int) -> dict:
        q_lower = query.lower()
        scored = []
        for m in self._memories:
            score = sum(1 for tok in q_lower.split() if tok in m.lower())
            scored.append((m, score))
        scored.sort(key=lambda x: -x[1])
        return {
            "results": [
                {"memory": m, "score": 100.0 - score}
                for m, score in scored[:limit]
            ]
        }


@pytest.fixture
def fake_memory():
    return FakeMemory([
        "user told me they like espresso in the morning",
        "the meeting was scheduled for tuesday at three",
        "shipping address is in brooklyn",
        "the project deadline is may fifteenth",
        "nomic embed text is the default embedding model",
    ])


@pytest.fixture
def stub_embed(monkeypatch):
    def _vec(t: str) -> list[float]:
        h = abs(hash(t))
        return [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(4)]

    monkeypatch.setattr(rh, "_embed_docs", lambda texts, cfg: [_vec(t) for t in texts])
    monkeypatch.setattr(rh, "_embed_queries", lambda texts, cfg: [_vec(t) for t in texts])


def test_zero_llm_is_the_default_tier(fake_memory, stub_embed):
    """Default-tier call must use the zero-LLM path; no filter/flagship invocation."""
    cfg = {"recall_limit": 5, "ollama_url": "http://localhost:11434"}
    with patch.object(rh, "_filter_rerank") as filter_spy, \
         patch.object(rh, "_flagship_rerank") as flagship_spy:
        memories, method = rh.recall_hybrid(
            fake_memory, "brooklyn shipping", user_id="agent", cfg=cfg,
        )
    assert filter_spy.call_count == 0, f"zero-LLM default invoked filter rerank; method={method}"
    assert flagship_spy.call_count == 0, f"zero-LLM default invoked flagship rerank; method={method}"
    assert isinstance(memories, list)
    assert "flagship" not in method


def test_zero_llm_ignores_missing_llm_config(fake_memory, stub_embed):
    """No filter_endpoint / filter_token → zero-LLM path must succeed unchanged."""
    cfg = {"recall_limit": 5, "ollama_url": "http://localhost:11434"}
    memories, method = rh.recall_hybrid(
        fake_memory, "espresso morning", user_id="agent", cfg=cfg, tier="zero_llm",
    )
    assert isinstance(memories, list)
    assert "flagship" not in method


def test_zero_llm_matches_default_behavior(fake_memory, stub_embed):
    """Explicit tier='zero_llm' and default-tier calls must produce identical ordering."""
    cfg = {"recall_limit": 5, "ollama_url": "http://localhost:11434"}
    explicit_mem, explicit_method = rh.recall_hybrid(
        fake_memory, "brooklyn", user_id="agent", cfg=cfg, tier="zero_llm",
    )
    default_mem, default_method = rh.recall_hybrid(
        fake_memory, "brooklyn", user_id="agent", cfg=cfg,
    )
    assert [m.get("text") or m.get("memory") for m in explicit_mem] == \
           [m.get("text") or m.get("memory") for m in default_mem]
    assert explicit_method == default_method


def test_zero_llm_empty_store_returns_empty_list(stub_embed):
    """Empty corpus produces [] not a crash."""
    cfg = {"recall_limit": 5, "ollama_url": "http://localhost:11434"}
    memories, method = rh.recall_hybrid(
        FakeMemory([]), "anything", user_id="agent", cfg=cfg, tier="zero_llm",
    )
    assert memories == []
    assert isinstance(method, str)
