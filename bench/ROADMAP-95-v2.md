# cogito-ergo Retrieval R&D — v2 (iterative, with oversight)

Supersedes ROADMAP-95.md. Same phases, disciplined loop, explicit checkpoints.

## Current baseline (2026-04-16)
- Zero-LLM combined: **R@1=83.2%**, R@5=98.3%, R@10=99.1%
- + qwen-turbo filter: R@1=83.4% (net +0.2pp; was +6.9pp mid-run before regression on hard tail)
- 470q LongMemEval_S, zero failures
- Raw: `bench/results-combined-pipeline-2026-04-15.json`

## Diagnosis (from `analyze_pipeline.py`)
- 64% of questions are multi-answer → LLM rerank promotes 1, buries others
- 38% are temporal → dates exist in `haystack_dates` but are never used
- Dip zone (q230-310): 96% temporal, 80% multi-answer

## The Loop (applies to every phase)
1. **Hypothesize** — write expected gain + falsification criterion in `bench/phase-N/HYPOTHESIS.md`
2. **Minimal change** — smallest possible diff; commit before running
3. **Measure** — full 470q run that emits per-question JSON (not just aggregates)
4. **Diagnose** — `diff_runs.py baseline.json phaseN.json` → wins/losses by qtype + multi-answer flag
5. **Generalize** — rerun on `eval_cases.json` (31-case) + 20-question cogito prod slice
6. **Checkpoint with Roli** — GO / TWEAK / REVERT decision before next phase
7. **Log** — copy phase dir to `~/Documents/projects/research-corpus/agent-infra/raw/cogito-phase-N-YYYYMMDD/`

## Kill criteria (global)
- Any phase that loses >2pp on the 31-case eval → revert regardless of 470q gain (benchmark overfitting guard)
- Any phase where a qtype drops >3pp → revert or route around it
- Never layer phase N+1 on top of a phase N that didn't cleanly beat its acceptance criterion

---

## Phase 0 — Instrumentation (NO OPTIMIZATION)
**Goal**: make the measurement rig trustworthy before changing anything.

- **0a**: Modify `longmemeval_combined_pipeline.py` so each run emits per-question JSON with fields:
  `{qid, qtype, question, gold_session_ids, s1_top5_ids, s1_top5_scores, s2_top5_ids, s2_top5_scores, s1_hit_at_1, s2_hit_at_1, s1_hit_at_5, s2_hit_at_5, filter_called, filter_ms, query_date, candidate_dates}`
  Output: `bench/runs/<run_id>/per_question.json` + existing aggregate results.
- **0b**: Rerun current pipeline once to produce `bench/runs/baseline-v2/per_question.json`. This is THE baseline.
- **0c**: Write `bench/diff_runs.py` — takes two per-question JSONs, prints:
  - Wins (S1 miss → S2 hit): count by qtype, list qids
  - Losses (S1 hit → S2 miss): count by qtype, list qids
  - Both-hit / both-miss counts
  - Per-qtype R@1 delta
- **0d**: Verify `eval_cases.json` 31-case harness still runs. Record baseline.
- **0e**: Sample a 20q slice from cogito prod memories → `bench/prod_slice.json`. Record baseline retrieval accuracy.

**Acceptance**: all three baselines recorded, diff tool works against existing data, no code changed that affects retrieval math.

**Checkpoint**: post the diff tool output against `results-combined-pipeline-2026-04-15.json` vs itself (should be zero delta) + the three baseline numbers. Roli confirms rig is trustworthy.

---

## Phase 1 — Date injection (deterministic, no LLM)
**Hypothesis**: Prepending `YYYY/MM/DD: ` to each chunk before BM25 + embedding gives both retrievers temporal signal they currently lack.
**Falsifier**: Temporal-reasoning R@1 does not move, OR non-temporal R@1 drops.
**Change**: one-line prepend in the chunking step of `longmemeval_combined_pipeline.py`.
**Run**: full 470q + eval_cases + prod slice.
**Acceptance**: Overall R@1 ≥ 84.0%, temporal-reasoning R@1 ≥ +3pp, no qtype drops >2pp.
**Kill**: revert if eval_cases drops >2pp (overfit signal).

---

## Phase 2 — Router (gate the LLM, don't change it)
**Hypothesis**: LLM filter helps on single-answer (~36% of questions), hurts on multi-answer (~64%). A regex-based classifier + score-gap threshold routes around the damage.
**Falsifier**: Routed single-answer cases still don't see LLM gain, OR multi-answer cases still regress.
**Change**: classifier function + gate in the filter step. No LLM prompt changes.
**Acceptance**: Overall R@1 ≥ 85%, multi-answer R@1 no worse than S1, single-answer R@1 ≥ S1+4pp.
**Kill**: revert if routed single-answer doesn't gain — means router signal is wrong, not the scaffold.

---

## Phase 3 — Diagnose-then-scaffold (no code yet — diagnosis gate)
After Phase 2, look at the per-q diff. Only THEN design scaffolds. Do not pre-commit to the contrastive-NOT / Korean evidential / code triad — that prescription assumes categories the router may have already solved or shifted.

Deliverable: `bench/phase-3/SCAFFOLD-DESIGN.md` with per-category scaffold choice justified by per-q data from Phase 2.
**Checkpoint before coding**: Roli reviews the design doc.

---

## Phase 4 — LPCI temporal scaffold (PROMOTED from phase 6)
**Hypothesis**: Pre-computing temporal structure and injecting it as language scaffold lets the stateless LLM answer temporal queries it can't answer from raw text alone. This is the LPCI thesis applied to retrieval.
**Falsifier**: Temporal-reasoning R@1 doesn't improve when model gets pre-computed date structure — means the thesis doesn't apply to this task.
**Change**: for temporal queries, build injection block from `haystack_dates`:
```
[1] 2023/05/20 (earliest)
[2] 2023/06/15 (+26 days)
[3] 2023/08/02 (+74 days, most recent)
```
Prepend to filter prompt for routed temporal cases.
**Acceptance**: Temporal-reasoning R@1 ≥ Phase 2 + 4pp.
**Why promoted**: it's the moat, it's cheap, and it needs to be falsified BEFORE flagship spend — not after.

---

## Phase 5 — Verify mode A/B
Existing `longmemeval_combined_verify.py`. Run with Phase 1-4 applied. Compare per-q vs rank mode. Pick winner per qtype or route between them.

---

## Phase 6 — Flagship on routed hard tail only
qwen-max (or opus) with full session text (up to 2000 chars) on the subset Phase 2-4 still miss. Gate by confidence threshold. Track cost per query.
**Acceptance**: overall R@1 ≥ 92%.

---

## Phase 7 — Ensemble (only if <95% after Phase 6)
Borda / multi-scaffold voting. Don't touch unless earlier phases plateau.

---

## Execution split
- **Main session (the agent that wrote ROADMAP-95.md)**: executes phases, writes per-q JSON, runs benchmarks, writes diagnosis
- **Fresh-context reviewer (separate Claude Code session, cron'd)**: reads artifacts, audits kill-criteria calls, flags drift, checks generalization
- **Roli**: final checkpoint GO / TWEAK / REVERT between phases

## File conventions
- Runs: `bench/runs/<run_id>/per_question.json` + `aggregate.json`
- Phase work: `bench/phase-N/` (HYPOTHESIS.md, diff output, DECISION.md)
- Reviews: `bench/reviews/phase-N-review-<timestamp>.md`
- Review queue: `bench/REVIEW-QUEUE.md` — main session appends; reviewer reads

## What we are NOT doing
- Better embedding model swap (nomic fine, R@5=98.3%)
- vocab_map integration (production concern, not benchmark)
- Scaffold-as-index (LLM preprocessing too slow)
- Cross-encoder reranking (already proven harmful)
- RRF_K / cosine weight tuning (already tried, marginal)
