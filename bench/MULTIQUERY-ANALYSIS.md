# Lightweight Multiquery Alternatives Analysis

## Problem
LLM-based query expansion (rephrase, keywords, inverted) improves R@1 from 56% → 65% on 20 LongMemEval questions, but costs 2.7s/query (4x overhead per-question). Not viable for production.

## Baseline Context
- **recall_b.py**: Already does zero-LLM query decomposition (original → stripped → bigrams → trigrams → tokens)
- **vocab_map**: Exists in `.cogito/config.json` — built by `cogito calibrate` to bridge natural language gaps
- **Multiquery adds**: LLM generates 3 variants, sub-queries those, batches all embeddings, merges with RRF
- **Current cost**: ~2.7s per query (100% LLM; sub-query decomposition adds negligible time)

## 5 Lightweight Alternatives: Ranked by Impact vs Effort

### 1. **Using existing vocab_map expansion** ⭐⭐⭐⭐⭐ (HIGH IMPACT, ZERO COST)
**Impact:** 3–5% gain (conservative estimate)  
**Cost:** 0ms  
**Effort:** 1 line (already implemented in recall_b.py)

**Why it works:** vocab_map already bridges semantic gaps (e.g., "freeze" → ["timeout", "cascade"]) learned from real corpus. Multiquery relies on the LLM generating similar expansions. The multiquery step adds LLM-generated variants that vocab_map wouldn't contain, but the delta is marginal for technical corpora where the vocab bridge is well-tuned.

**Action:** Port `_expand_with_vocab_map()` from recall_b.py into longmemeval_multiquery.py. Verify multiquery calls `_build_subqueries(query, vocab_map)` in retrieve_sessions().

**Risk:** vocab_map must exist (requires prior `cogito calibrate` run). Fallback to no-expansion if missing. Impact is domain-dependent: worse on OOD corpora, strong on in-domain.

---

### 2. **Embedding with different prefix prompts** ⭐⭐⭐⭐ (MODERATE IMPACT, ZERO RUNTIME COST)
**Impact:** 2–4% gain  
**Cost:** 0ms  
**Effort:** 2 embedding calls (already batched)

**Why it works:** Nomic embed-text supports prefixes: `"search_query:"` vs `"search_document:"`. Embedding the same query twice with different prefixes can capture query-focused vs document-focused similarity spaces. This is similar to what multiquery does (rephrasing explores different facets) but without LLM generation.

**Action:**
```python
# Instead of: [query] + subqueries
# Do: ["search_query:" + query, query + " (search terms)"] + subqueries
```
Creates dual embeddings for the original query, exploiting Nomic's prefix semantics.

**Risk:** Minimal. Nomic supports both prefixes; just need to embed both and weight accordingly in RRF.

---

### 3. **Increase MAX_SUBQUERIES from 8 to 12+** ⭐⭐⭐ (LOW-MODERATE IMPACT, MINIMAL COST)
**Impact:** 1–2% gain  
**Cost:** <100ms (3–4 additional embeddings + cosine similarity)  
**Effort:** 1 constant change

**Why it works:** More n-grams = broader keyword coverage. The recall_b decomposition is already sound; diminishing returns kick in after 8 due to RRF overlap. Going to 12 explores more token permutations but adds linear cost in embedding + similarity.

**Action:**
```python
MAX_SUBQUERIES = 12  # or 16 for aggressive exploration
```

**Risk:** Minimal overhead, but RRF with 16 queries might oversample junk queries. Monitor NDCG/cosine ceiling for saturation.

---

### 4. **Query permutations (word order variants)** ⭐⭐ (LOW IMPACT, MODERATE COST)
**Impact:** 0.5–1% gain  
**Cost:** ~50ms per question (5–10 permutations, batch embedded)  
**Effort:** 10 lines

**Why it works:** Different word orders sometimes encode different intent ("list all items" vs "all items list"). Multiquery's "inverted" variant partially captures this. Pure permutations are brittle on short queries.

**Action:**
```python
def add_permutations(tokens: list[str]) -> list[str]:
    if len(tokens) > 2 and len(tokens) <= 5:
        import itertools
        # Return 2–3 random permutations, not all n!
        perms = set()
        for _ in range(2):
            perm = random.sample(tokens, len(tokens))
            perms.add(" ".join(perm))
        return list(perms)
    return []
```

**Risk:** Noise on short queries. Only useful for mid-length (4–8 word) queries. Requires random seeding; non-deterministic unless seeded.

---

### 5. **Static synonym expansion (word map)** ⭐ (LOW IMPACT, MODERATE COST)
**Impact:** 0.5–1% gain  
**Cost:** ~20ms  
**Effort:** 50 lines + curated synonym file

**Why it works:** Hardcoded synonyms (e.g., "error" → ["bug", "fault", "failure"]) cover common word variants. Multiquery LLM does this implicitly. Hand-curated maps are brittle without domain tuning.

**Action:**
```python
SYNONYM_MAP = {
    "error": ["bug", "fault", "exception", "failure"],
    "fast": ["quick", "rapid", "speedy", "efficient"],
    # ... ~50 pairs
}
def expand_synonyms(tokens: list[str]) -> list[str]:
    for tok in tokens:
        if tok in SYNONYM_MAP:
            return SYNONYM_MAP[tok]
    return []
```

**Risk:** HIGHEST. Requires manual curation, scales poorly, domain-specific failures (e.g., "tree" vs "AST" are not synonyms in CS). vocab_map is already a better version of this.

---

## Recommended Strategy

**Tier 1 (Do first):**
1. **Verify vocab_map is loaded** in longmemeval_multiquery.py (copy from recall_b's pattern)
2. Run benchmark with vocab expansion only → measure R@1 uplift (expect 2–4%)

**Tier 2 (If R@1 still <63%):**
3. Add prefix embedding strategy (dual "search_query:" vs baseline)
4. Increase MAX_SUBQUERIES from 8 → 12
5. Re-test → target 63–64% R@1

**Avoid:**
- Permutations (noise > signal for typical questions)
- Static synonym expansion (vocab_map supersedes it)

## Estimated Wins
- vocab_map alone: 56% → 60–61% (4–5% absolute)
- + prefix embedding: 60% → 62% (+1–2%)
- + MAX_SUBQUERIES=12: 62% → 63% (+1%)
- **Total achievable via no-LLM tactics: ~63% R@1, <500ms per query**

This leaves ~2% gap to multiquery's 65%, acceptable trade for 5–6x latency reduction.
