# LongMemEval Benchmark Methodology

## 1. Dataset

**LongMemEval S (Small) Split**: 470 total questions from session-based haystack retrieval tasks. Filters remove abstention questions (those with "_abs" suffix), leaving 443 evaluable questions.

**Task Definition**: For each question, given a set of haystack sessions (conversation history), retrieve the sessions most likely to contain the answer. Scoring is per-question, where multiple answer sessions may exist.

**Metrics**:
- **Recall@k (R@k)**: Binary metric—does at least one correct session appear in the top-k?
- **Recall-Any**: Reports R@k for "at least one answer session recalled"
- **Recall-All**: Reports R@k for "all answer sessions recalled" (stricter, multi-answer compliance)
- **NDCG@k**: Normalized Discounted Cumulative Gain, a ranking quality measure that penalizes when correct items appear lower in the ranking

Evaluated at k = {1, 3, 5, 10}.

---

## 2. Baseline: recall_b (Zero-LLM Multi-Query + RRF + Cosine Rerank)

**Architecture**:
1. **Query Decomposition** (deterministic, no LLM): Each query is broken into up to 8 sub-queries:
   - Original query
   - Key tokens only (stopwords removed)
   - Bigrams and trigrams of key tokens
   - Individual 4+ character terms
2. **Embedding**: All sub-queries and corpus sessions embedded via `nomic-embed-text` (Ollama, locally hosted)
3. **Fusion**: Each sub-query ranked independently against all sessions via cosine similarity (top-20 per query)
4. **RRF Merge**: Reciprocal Rank Fusion combines rankings: `score(doc) = sum(1/(K+rank))` for all queries that ranked the doc
   - `_RRF_K = 60`: RRF parameter
5. **Cosine Rerank**: Second-pass blend against original query embedding:
   - Final Score = `(1 - cosine_weight) * rrf_normalized + cosine_weight * cosine_sim`
   - `_COSINE_WEIGHT = 0.7`: Weighting parameter (favors semantic match over fusion consensus)

**Max Subqueries**: 8 (capped to avoid embedding load)

**Corpus Truncation**: 6000 characters per session text (to fit token context)

**Results** (443 questions):
- **R@1 Any**: 56.2%
- **R@5 Any**: 81.3%
- **R@10 Any**: 92.3%
- **R@1 NDCG**: 0.562
- **Avg Latency**: 75ms per query
- **Total Time**: 33.1s

---

## 3. Test 1: Multiquery — LLM Query Expansion (gemma3:4b)

**What Changed**: Added LLM-based semantic expansion to baseline sub-queries.

**Architecture**:
1. Original baseline sub-queries (8 maximum)
2. + LLM expansion (gemma3:4b via Ollama):
   - Prompt: "Generate exactly 3 rephrasings: (1) rephrase with different words, (2) key terms only, (3) focus on answer type."
   - Generates 3 query variants, each then sub-query decomposed (4 sub-queries per variant)
   - Total queries capped at 16
3. All queries embedded, ranked, and merged with **identical RRF + cosine rerank** as baseline

**Hypothesis**: Semantic rephrasing by LLM captures synonyms and answer-type inference, improving R@1 through richer query diversity.

**Latency Impact**: LLM expansion adds ~2.7s average (2708ms vs 75ms baseline) due to gemma3:4b inference.

**Results** (20 questions, early-stage limited run):
- **R@1 Any**: 65.0% (+8.8 percentage points vs baseline)
- **R@5 Any**: 75.0% (-6.3 points; limited by small sample)
- **R@1 NDCG**: 0.650
- **Avg Latency**: 2708ms per query
- **Note**: Early-stage test on subset; full 443-question run not completed due to latency

---

## 4. Test 2: Rerank — Two-Pass Cross-Encoder Reranking (bge-reranker-v2-m3)

**What Changed**: Added second-pass reranking of top-20 candidates using a cross-encoder model.

**Architecture**:
1. Phase 1: Embed all corpus sessions with `nomic-embed-text` (baseline-compatible)
2. Phase 2: Pre-embed all corpus sessions and queries with `qllama/bge-reranker-v2-m3` in separate embedding space
3. Phase 3: Retrieval + reranking:
   - First pass: baseline RRF + cosine rerank (nomic embeddings)
   - Second pass: rerank top-20 from first pass using bge-reranker cosine similarity
   - Blend: `score = (1 - rerank_weight) * initial_score + rerank_weight * reranker_score`
   - `RERANK_WEIGHT = 0.5`: Equal weighting between initial and reranker scores

**Note**: bge-reranker used as bi-encoder here (cosine similarity on embeddings), not as true cross-encoder (query-doc pair input). This is a simplification.

**Hypothesis**: Cross-encoder (or cross-encoder-like bi-encoder) specializes in query-doc relevance judgment beyond sparse embedding distance, improving R@1 via better top-1 discrimination.

**Results** (20 questions, same limited run):
- **R@1 Any**: 20.0% (-36 points vs baseline; **REGRESSION**)
- **R@5 Any**: 60.0% (-21.3 points)
- **R@1 NDCG**: 0.200
- **Avg Latency**: 106ms per query
- **Issue**: bge-reranker markedly underperformed. Likely causes:
  - Model mismatch (bge-reranker trained on short-form QA, not long session text)
  - Reranker_weight = 0.5 may over-trust reranker scores relative to initial ranking
  - Bi-encoder usage instead of true cross-encoder (model not intended for cosine similarity)

---

## 5. Test 3: Tuned Baseline (Hypothesis: Parameter Optimization)

**Proposed Improvements to Test** (not yet run):
- Increase `max_subqueries` from 8 → 12: Richer query coverage without explosion
- Reduce `cosine_weight` from 0.7 → 0.55: Rebalance toward RRF consensus (more fusion-friendly)
- Reduce truncation from 6000 → 2000 characters: Faster embedding, less noise from long sessions

**Hypothesis**: More sub-queries improve coverage; reweighting RRF higher may benefit low-diversity sessions; truncation noise reduction improves embedding quality.

**Projected Impact**: 
- ~2-4% R@1 improvement
- Latency reduction from truncation offset by 50% more sub-queries
- Total latency ≈ 80–100ms (vs baseline 75ms)

---

## 6. Known Limitations

1. **Truncation Loss**: 6000-character (or proposed 2000-char) limit loses context from long multi-turn sessions. Queries requiring early-conversation context may fail retrieval.

2. **LLM Expansion Latency**: gemma3:4b adds 2.6+ seconds per query, making real-time deployment impractical without caching or batching.

3. **bge-reranker Misuse**: Model used as bi-encoder (cosine on embeddings) rather than true cross-encoder (dense ranking layer). This likely caused regression in Test 2. Proper cross-encoder usage would require per-query-doc pair inference (exponential cost).

4. **RRF Stagnation**: RRF with `_RRF_K=60` assumes ranked lists have value. With high cosine_weight (0.7), fusion contribution may saturate; diminishing returns on additional sub-queries.

5. **Test 1 & 2 Sample Size**: Both expansion and reranking tests ran on only 20 questions (4.5% of full split). Results are illustrative, not statistically robust.

6. **Embedding Model Bias**: nomic-embed-text trained on short-form text; session contexts are long and conversational (up to 6000 chars). Semantic drift likely at truncation boundaries.

---

## 7. Reproducibility

**Environment**:
- OS: macOS 12+
- Ollama 0.20.6 (local inference server)
- Models:
  - `nomic-embed-text` (768-dim, 8192-token context)
  - `gemma3:4b` (4B parameter LLM, inference only)
  - `qllama/bge-reranker-v2-m3` (bi-encoder, alternative to cross-encoder)

**Data Source**: LongMemEval S split (curated session-based QA, available via https://github.com/long-mem-eval/longmemeval)

**Running Benchmarks**:
```bash
# Baseline
python bench/longmemeval_retrieval.py --split s --limit 0

# Test 1: Multiquery
python bench/longmemeval_multiquery.py --split s --limit 20

# Test 2: Rerank
python bench/longmemeval_rerank.py --split s --limit 20
```

**Key Parameters** (hardcoded in scripts):
- `OLLAMA_URL = "http://localhost:11434"`
- `EMBED_MODEL = "nomic-embed-text"`
- `MAX_SUBQUERIES = 8`
- `_RRF_K = 60`
- `_COSINE_WEIGHT = 0.7`
- `RERANK_WEIGHT = 0.5` (Test 2 only)

---

## 8. Next Steps

1. **Full Test 1 Run**: Re-run multiquery on full 443 questions with latency profiling to justify LLM cost
2. **Cross-Encoder Integration**: Replace bi-encoder reranking with true cross-encoder (e.g., mxbai-rerank) or fix bge usage (pair-wise scoring)
3. **Parameter Sweep**: Grid search on cosine_weight, RRF_K, truncation length for tuned baseline
4. **Ablation**: Isolate RRF vs. cosine rerank contribution to validate 0.7 weighting
5. **Error Analysis**: Inspect failure cases (R@10=0) for patterns (question type, session length, answer position)
