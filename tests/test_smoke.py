"""
Smoke tests — no Ollama, no ChromaDB, no network required.
"""

import pytest
from fidelis.config import load
from fidelis.recall import _parse_indices, _filter_by_since, _parse_iso_date
from fidelis.recall_b import _build_subqueries, _rrf_merge


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_load_defaults():
    cfg = load(config_path="/nonexistent/path/.cogito.json")
    assert cfg["port"] == 19420
    assert cfg["user_id"] == "agent"


def test_config_load_returns_dict():
    cfg = load(config_path="/nonexistent/path/.cogito.json")
    assert isinstance(cfg, dict)
    assert "recall_limit" in cfg
    assert "llm_model" in cfg


# ---------------------------------------------------------------------------
# _parse_indices
# ---------------------------------------------------------------------------

CANDIDATES = [
    {"text": "alpha", "score": 1.0},
    {"text": "beta", "score": 2.0},
    {"text": "gamma", "score": 3.0},
]


def test_parse_indices_basic():
    result, method = _parse_indices("[1, 3]", CANDIDATES)
    assert method == "filter"
    assert len(result) == 2
    assert result[0]["text"] == "alpha"
    assert result[1]["text"] == "gamma"


def test_parse_indices_empty_array():
    result, method = _parse_indices("[]", CANDIDATES)
    assert method == "filter"
    assert result == []


def test_parse_indices_out_of_range():
    result, method = _parse_indices("[0, 4, 99]", CANDIDATES)
    # 0 and 4+ are out of range for 3 candidates — should be dropped
    assert method == "filter"
    assert result == []


def test_parse_indices_bad_json_fallback():
    result, method = _parse_indices("not json at all", CANDIDATES)
    assert method == "fallback_parse_error"
    assert result is CANDIDATES


def test_parse_indices_strips_think_tokens():
    raw = "<think>I should return 1 and 2</think>[1, 2]"
    result, method = _parse_indices(raw, CANDIDATES)
    assert method == "filter"
    assert len(result) == 2


def test_parse_indices_deduplicates():
    result, method = _parse_indices("[1, 1, 2]", CANDIDATES)
    assert method == "filter"
    assert len(result) == 2


# ---------------------------------------------------------------------------
# _build_subqueries
# ---------------------------------------------------------------------------

def test_build_subqueries_original_always_first():
    query = "What is the recall score for the snapshot layer?"
    subqueries, _ = _build_subqueries(query)
    assert subqueries[0] == query


def test_build_subqueries_no_stop_words_in_stripped():
    query = "how does the snapshot layer improve recall"
    subqueries, _ = _build_subqueries(query)
    # Second item should be the stripped phrase — no stop words
    stripped = subqueries[1]
    stop_words = {"how", "does", "the"}
    for word in stop_words:
        assert word not in stripped.split(), f"stop word '{word}' found in stripped phrase"


def test_build_subqueries_max_count():
    from fidelis.recall_b import MAX_SUBQUERIES
    query = "tell me about the infrastructure changes to the server configuration"
    subqueries, _ = _build_subqueries(query)
    assert len(subqueries) <= MAX_SUBQUERIES


def test_build_subqueries_empty_query():
    subqueries, expanded = _build_subqueries("")
    # Empty query: original is empty string, gets added then tokens are empty
    assert isinstance(subqueries, list)
    assert expanded is False


def test_build_subqueries_with_vocab_map():
    vocab_map = {"recall": ["retrieval", "memory search"]}
    query = "recall score after snapshot"
    subqueries, expanded = _build_subqueries(query, vocab_map)
    assert expanded is True
    # Expansion terms should appear in subqueries
    all_sq = " ".join(subqueries)
    assert "retrieval" in all_sq or "memory search" in all_sq


def test_build_subqueries_no_vocab_expansion_when_no_match():
    vocab_map = {"unrelated_term": ["something"]}
    query = "recall score after snapshot"
    subqueries, expanded = _build_subqueries(query, vocab_map)
    assert expanded is False


# ---------------------------------------------------------------------------
# _rrf_merge
# ---------------------------------------------------------------------------

def test_rrf_merge_single_run():
    runs = [[
        {"text": "alpha", "score": 1.0},
        {"text": "beta", "score": 2.0},
    ]]
    result = _rrf_merge(runs, limit=10)
    assert len(result) == 2
    # rank 1 should score higher than rank 2
    assert result[0]["text"] == "alpha"
    assert result[1]["text"] == "beta"


def test_rrf_merge_deduplicates_across_runs():
    runs = [
        [{"text": "alpha", "score": 1.0}, {"text": "beta", "score": 2.0}],
        [{"text": "beta", "score": 1.0}, {"text": "gamma", "score": 2.0}],
    ]
    result = _rrf_merge(runs, limit=10)
    texts = [r["text"] for r in result]
    assert len(texts) == 3
    assert len(set(texts)) == 3  # no duplicates


def test_rrf_merge_boosted_by_multiple_runs():
    # "beta" appears at rank 1 in both runs, "alpha" at rank 1 in one only
    runs = [
        [{"text": "alpha", "score": 1.0}, {"text": "beta", "score": 2.0}],
        [{"text": "beta", "score": 1.0}, {"text": "gamma", "score": 2.0}],
    ]
    result = _rrf_merge(runs, limit=10)
    # beta appears in both runs (rank 2 + rank 1) → higher RRF than alpha (rank 1 only)
    assert result[0]["text"] == "beta"


def test_rrf_merge_respects_limit():
    runs = [[
        {"text": f"item_{i}", "score": float(i)}
        for i in range(20)
    ]]
    result = _rrf_merge(runs, limit=5)
    assert len(result) == 5


def test_rrf_merge_empty_runs():
    result = _rrf_merge([], limit=10)
    assert result == []


def test_rrf_merge_skips_empty_text():
    runs = [[
        {"text": "", "score": 1.0},
        {"text": "valid", "score": 2.0},
    ]]
    result = _rrf_merge(runs, limit=10)
    assert len(result) == 1
    assert result[0]["text"] == "valid"


# ---------------------------------------------------------------------------
# _parse_iso_date
# ---------------------------------------------------------------------------

def test_parse_iso_date_iso_format():
    dt = _parse_iso_date("2026-04-22T10:30:00Z")
    assert dt.year == 2026
    assert dt.month == 4
    assert dt.day == 22


def test_parse_iso_date_date_only():
    dt = _parse_iso_date("2026-04-22")
    assert dt.year == 2026
    assert dt.month == 4
    assert dt.day == 22


def test_parse_iso_date_invalid():
    with pytest.raises(ValueError):
        _parse_iso_date("not-a-date")


# ---------------------------------------------------------------------------
# _filter_by_since
# ---------------------------------------------------------------------------

MEMORIES_WITH_DATES = [
    {"text": "old fact", "score": 1.0, "created_at": "2026-04-20"},
    {"text": "recent fact", "score": 2.0, "created_at": "2026-04-22"},
    {"text": "future fact", "score": 3.0, "created_at": "2026-04-23"},
]


def test_filter_by_since_basic():
    result, applied = _filter_by_since(MEMORIES_WITH_DATES, "2026-04-21")
    assert applied is True
    assert len(result) == 2
    assert result[0]["text"] == "recent fact"
    assert result[1]["text"] == "future fact"


def test_filter_by_since_exact_date():
    result, applied = _filter_by_since(MEMORIES_WITH_DATES, "2026-04-22")
    assert applied is True
    assert len(result) == 2
    assert result[0]["text"] == "recent fact"


def test_filter_by_since_no_matches():
    result, applied = _filter_by_since(MEMORIES_WITH_DATES, "2026-04-25")
    assert applied is False
    assert result == MEMORIES_WITH_DATES


def test_filter_by_since_invalid_date():
    result, applied = _filter_by_since(MEMORIES_WITH_DATES, "invalid-date")
    assert applied is False
    assert result == MEMORIES_WITH_DATES


def test_filter_by_since_missing_created_at():
    memories = [
        {"text": "has date", "score": 1.0, "created_at": "2026-04-22"},
        {"text": "no date", "score": 2.0},
    ]
    result, applied = _filter_by_since(memories, "2026-04-21")
    assert applied is True
    assert len(result) == 1
    assert result[0]["text"] == "has date"
