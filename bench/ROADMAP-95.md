# cogito-ergo Retrieval R&D Roadmap: 83% → 95%+ R@1

## Current State (2026-04-16)
- **Zero-LLM combined**: R@1=83.2%, R@5=98.3%, R@10=99.1% (296ms)
- **+ qwen-turbo filter**: R@1=83.4% (+0.2pp net, but +6.9pp mid-run before regression)
- **Benchmark**: LongMemEval S, 470 questions, zero failures

## Diagnosis: Why We're Stuck at 83%

64% of questions need MULTIPLE answer sessions. The LLM filter promotes one, buries others.
38% are temporal. Dates exist in metadata but are never used.
The dip zone (q230-310) is 96% temporal-reasoning, 80% multi-answer.

| Question Type | Count | % | Multi-answer? | Current weakness |
|---------------|-------|---|---------------|-----------------|
| temporal-reasoning | 127 | 27% | 84% multi | No date signal anywhere |
| multi-session | 121 | 26% | 100% multi | LLM hurts by promoting 1 |
| knowledge-update | 72 | 15% | 100% multi | LLM prefers stale over updated |
| single-session-user | 64 | 14% | 0% multi | LLM helps here |
| single-session-assistant | 56 | 12% | 0% multi | LLM helps here |
| single-session-preference | 30 | 6% | 0% multi | Already high accuracy |

## Phase 1: Date Injection (no LLM, targets temporal 38%)
**Hypothesis**: Prepending session dates to chunk text before embedding gives BM25 and dense search temporal signal they currently lack.
**What**: Modify chunk text from "I upgraded my RAM" → "2023/05/20: I upgraded my RAM"
**Expected gain**: +3-5pp R@1 on temporal questions = +1-2pp overall
**Effort**: 5 lines of code change in combined script
**Risk**: Low — additive signal, can't hurt non-temporal queries
**Acceptance**: R@1 > 84.5% on full 470q

## Phase 2: Router (targets multi-answer 64% + single-answer 36%)
**Hypothesis**: LLM helps on single-answer questions, hurts on multi-answer. Route accordingly.
**What**: 
- Detect query type at runtime via regex (temporal keywords, counting keywords)
- Score gap check (rank 1 vs rank 2 blended score)
- If multi-answer pattern OR high confidence → skip LLM, keep Stage 1 order
- If single-answer pattern AND low confidence → call LLM filter
**Expected gain**: Preserve the +6.9pp the LLM gives on single-answer cases, prevent regression on multi-answer
**Effort**: ~50 lines (regex classifier + score gap threshold)
**Risk**: Medium — threshold tuning needed, but fallback is always Stage 1
**Acceptance**: R@1 > 87% on full 470q

## Phase 3: Purpose-Built Scaffolds (targets routed LLM cases)
**Hypothesis**: Different question types need different scaffolds. One generic code-scaffold underperforms.
**What**:
- Factual recall → code-as-scaffold (proven 94% on 2b classification)
- Temporal single-answer → scaffold with injected dates + temporal reasoning prompt
- Knowledge-update → contrastive-NOT scaffold ("the NEWER session is more relevant than the older one")
- Preference recall → Korean evidential commitment scaffold (forces epistemic declaration)
**Expected gain**: +2-4pp on routed questions = +1-2pp overall
**Effort**: Prompt engineering per category, ~100 lines
**Risk**: Low — scaffolds are additive, router handles fallback
**Acceptance**: R@1 > 89% on full 470q

## Phase 4: Verify Mode (alternative to rank mode)
**Hypothesis**: Binary "does this answer the query? yes/no" is easier for small models than "rank 5 candidates"
**What**: Already built. --verify flag. Checks candidates sequentially until YES.
**Expected gain**: Unknown — needs testing. Could beat rank mode on ambiguous cases.
**Effort**: Zero — script exists
**Risk**: Uses up to 5x more API calls per question
**Acceptance**: R@1 > rank mode on same question set

## Phase 5: Flagship Model on Hard Tail (targets remaining ~5-8% gap)
**Hypothesis**: qwen-max/opus with full session text (not 500-char snippets) resolves the hardest cases.
**What**:
- Router identifies hard cases (low confidence after Phase 2-3)
- Send full session text (up to 2000 chars) for top-5 candidates to flagship model
- Purpose-built temporal scaffold with pre-computed date ordering
**Expected gain**: +3-5pp on hard cases = +2-3pp overall
**Effort**: Swap model + increase snippet length for routed cases
**Risk**: Cost per query increases ($0.01-0.05 for flagship calls)
**Acceptance**: R@1 > 92% on full 470q

## Phase 6: Temporal Scaffold — The LPCI Play (targets the ceiling)
**Hypothesis**: Pre-computing temporal structure ("Session A precedes Session B by 26 days") and injecting it as language scaffold gives the model state it can't derive.
**What**:
- For temporal questions: compute pairwise date ordering of top-5 candidates
- Inject as structured text: "[1] May 20 (earliest) ... [3] June 15 (26 days later)"
- LLM reads temporal state from scaffold rather than inferring from text
**Expected gain**: +2-3pp on temporal questions = +1pp overall
**Effort**: ~30 lines of date parsing + prompt injection
**Risk**: Low — only activates on temporal queries
**Acceptance**: R@1 > 94% on full 470q

## Phase 7: Ensemble + Final Push (if needed)
- Borda count across BM25 rank + cosine rank + LLM rank
- Multi-scaffold voting (run 3 scaffolds, majority wins)
- Anti-scaffold pass (nonsense scaffold on 4b for FP-heavy cases)
**Only if**: Phases 1-6 don't reach 95%

## Execution Order
Phase 1 → Phase 2 → run benchmark → Phase 3 → run benchmark → Phase 4 → compare with rank mode → pick best → Phase 5 → Phase 6 → final benchmark

## Not Doing (diminishing returns)
- Better embedding model swap (nomic is fine, R@5=98.3%)
- vocab_map integration (good for production, marginal for benchmark)
- Scaffold-as-index (LLM preprocessing too slow)
- Cross-encoder reranking (already proven harmful)
- Cosine weight / RRF_K tuning (already tried, minimal impact)

## Production Integration (after benchmark target hit)
1. Apply date injection to production recall_b.py
2. Build router into /recall endpoint
3. Per-category scaffold selection in filter step
4. Flagship escalation as configurable option
5. Rerun 31-case eval + real corpus eval to verify generalization
