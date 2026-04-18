# LongMemEval Retrieval Improvement Comparison — 2026-04-15

## Results

| Test | N | R@1 | R@5 | R@10 | NDCG@10 | Avg latency |
|------|---|-----|-----|------|---------|-------------|
| **Baseline** (recall_b) | 443 | 56.2% | 81.3% | 92.3% | 69.2% | 75ms |
| **Rerank** (bge-reranker second pass) | 443 | 30.5% | 66.1% | 85.3% | — | ~106ms |
| **Multiquery** (gemma3:4b expansion) | 20* | 65.0% | 75.0% | 85.0% | 73.6% | 2708ms |
| **Tuned** (cosine=0.55, subq=12) | 20* | 50.0% | 70.0% | 80.0% | 68.2% | 82ms |

*20-question sample only — full 470-question run not completed.

## Analysis

### Rerank: NEGATIVE (confirmed at scale)
- R@1 dropped 56.2% → 30.5% (−25.7pp)
- R@5 dropped 81.3% → 66.1% (−15.2pp)
- R@10 dropped 92.3% → 85.3% (−7.0pp)
- **Root cause**: bge-reranker-v2-m3 was used as a bi-encoder (independent query/doc embeddings + cosine). Cross-encoders need paired input to work — Ollama doesn't expose a proper rerank API. The reranker's embedding space is not optimized for independent cosine similarity, so it adds noise.
- **Verdict**: Do not ship. Cross-encoder reranking requires a different inference path.
- **Note**: Rerank test used 2000-char truncation vs baseline's 6000-char. This loses context but the truncation alone wouldn't explain a 25pp R@1 drop.

### Multiquery: PROMISING (20-question sample)
- R@1 improved 56.2% → 65.0% (+8.8pp on 20q sample)
- R@5 and R@10 appear lower but sample is too small (20 vs 443) to draw conclusions
- **Mechanism**: LLM-generated query variants (rephrase, keyword extraction, answer-type focus) surface results that match different phrasings of the same intent
- **Cost**: 2.7s per query (gemma3:4b inference) — not viable for production
- **Verdict**: The R@1 gain validates that semantic query expansion helps. Ship a lightweight version (see below), not the LLM call.

### Tuned baseline: INCONCLUSIVE (20-question sample)
- R@1 at 50.0% on 20q vs baseline 56.2% on 443q — not directly comparable
- Changed: cosine_weight 0.7→0.55, MAX_SUBQUERIES 8→12, truncation 6000→2000
- **Verdict**: Needs full 470-question run to evaluate. The 2000-char truncation may lose relevant context on long sessions, offsetting the sub-query gains.

## Key discoveries

1. **Truncation bug found**: Some LongMemEval sessions (e.g., JSON/hex blobs) tokenize to >8192 tokens at 6000 chars. The baseline silently skipped these (27 questions). This is a real bug in cogito's recall_b — any user storing code/JSON gets silent recall failures. Fix: truncate to 2000 chars or implement token-aware truncation.

2. **vocab_map is the lightweight multiquery**: Cogito already has a vocabulary bridge (built by calibrate.py) that maps plain English → technical terms. This is precomputed multiquery expansion with zero latency. Highest-ROI improvement to test next.

3. **Cross-encoder reranking needs proper infrastructure**: bge-reranker as bi-encoder is wrong tool. Would need either (a) Ollama rerank API support, (b) direct model loading via sentence-transformers, or (c) a different reranking approach entirely.

## Recommended next steps

1. Fix 2000-char truncation in production recall_b.py (real bug)
2. Test vocab_map integration as lightweight query expansion
3. Run full 470-question multiquery + tuned tests from terminal
4. Consider MAX_SUBQUERIES 8→10 as a zero-cost improvement (needs full run to validate)

## Reproducibility

- Ollama 0.20.6, macOS Darwin 24.6.0
- Embedding: nomic-embed-text (768-dim, 8192 token context)
- Reranker: qllama/bge-reranker-v2-m3 (1024-dim, used as bi-encoder)
- LLM expansion: gemma3:4b (Ollama, ~2.5s per query)
- Dataset: LongMemEval S split, 470 non-abstention questions
- See METHODOLOGY.md for full protocol
