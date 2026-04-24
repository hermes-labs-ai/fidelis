# cogito-ergo — Status as of 2026-04-24

One-page crystallization. If you only read one file, read this.

## Bottom line

cogito-ergo has **two retrieval paths** living in the same server. Path A is the
production path (atomic-fact recall). Path B is the session-retrieval pipeline,
benchmarked at **96.4% R@1 on LongMemEval_S** (runP-v35, 2026-04-18). On
cogito's own internal eval, Path B scores 54% — 21 points below Path A's 75%.
The benchmark win and the production workload are not the same task. This is
documented, not hidden.

## Numbers

| System | Benchmark | R@1 | Notes |
|---|---|---|---|
| `/recall` (atomic, Path A) | 31-case eval (cogito's own) | 75% | stochastic 60-90% per seed |
| `/recall` (atomic, Path A) | LongMemEval_S | not run | workload mismatch |
| `/recall_hybrid` (session, Path B) | LongMemEval_S | **96.4%** (453/470, runP-v35) | runtime-only escalation, no hardset |
| `/recall_hybrid` (session, Path B) | 31-case eval (cogito's own) | 54% | 21pt regression vs Path A |

**Combined (snapshot + recall):** 85% R@1, 96% hit@any on 31-case eval.

## The two architectures

**Path A — `/recall` (atomic):**
- Stores ~50-200 char facts
- Dense (nomic) retrieval + optional LLM filter reranker
- Tuned for paraphrase queries over short facts
- Code: `src/cogito/recall.py`

**Path B — `/recall_hybrid` (session):**
- Stores 2000+ char multi-turn sessions
- BM25 + nomic dense + RRF fusion
- Turn-level chunking, nomic prefixes (`search_query:` / `search_document:`)
- Regex router classifies query type → llm / skip / default
- Tiered LLM: runtime escalation threshold (`top1_score < 0.8 OR gap < 0.07`)
- Code: `src/cogito/recall_hybrid.py`
- Benchmark: `bench/longmemeval_combined_pipeline_v35.py` (runP-v35)

**Why Path B regresses on cogito's eval:** BM25 adds noise on 50-200 char facts;
the filter's 150-char snippet truncation mangles session content. Two separate bugs,
fixable by per-workload indexes (see `docs/DISPATCHER_DESIGN.md`).

## LongMemEval_S per-category breakdown (runP-v35)

| Category | n | R@1 |
|---|---|---|
| single-session-user | 64 | 100.0% |
| multi-session | 121 | 99.2% |
| knowledge-update | 72 | 98.6% |
| single-session-assistant | 56 | 98.2% |
| temporal-reasoning | 127 | 92.1% |
| single-session-preference | 30 | 86.7% |

17 total misses. Temporal-reasoning accounts for 59% (10/17).

## Known bugs (NOT fixed — documented)

1. **Verify-guard never activated.** Guard logic coded in pipeline; activation field
   absent from per-question output. The 96.4% is from temporal boost + runtime
   escalation only. (Source: WRITEUP-LONGMEMEVAL-20260423.md)

2. **Workload divergence.** Path B 96.4% on LongMemEval, 54% on cogito's 31-case
   eval. Architectural: BM25 hurts short facts, snippet truncation hurts sessions.
   Separate indexes required.

3. **Escalation rate: 80.2% actual vs 10% intended.** Threshold was calibrated on
   65 SSU/multi-session questions; preference questions have different confidence
   distributions. In production this is 8x the planned cost.

4. ~~**Preference-question QA: 0.0%.** Resolved.~~ E2 QA run (2026-04-24): Pref = 66.7%
   with dedicated preference prompt in v3_routing.py. Original 0% was from terse prompts.

## What's been tried (don't redo)

| Attempt | Outcome |
|---|---|
| Combined retrieval (BM25 + dense + RRF + chunks + prefixes, zero LLM) | 83.2% R@1 |
| Regex router + qwen-turbo filter | 88.9% |
| Learned router (LR + RF, 774 features) | 83% CV — worse than hand-regex |
| Chunk-level date injection | 0pp net |
| LPCI temporal scaffold in filter prompt | -1.2pp. Reverted |
| Assistant-text in snippets | Regressed. Reverted |
| qwen-max flagship on hardset | +3.6pp → 93.4% (v33) |
| Temporal boost + runtime escalation (runP, v35) | **96.4%** |
| Terse qtype-specific QA prompts | -10pp vs quote-first CoT. Never use |
| gemma3:27b local QA reader | Worse than qwen-max on most qtypes |
| v3 MS synthesis prompt + gpt-4o-mini (E1) | -11.9pp on MS. Rejected for mini. |
| GPT-4o reader for MS/TR/KU + v3_routing prompts (E2) | **75.5% overall** (355/470) +11.5pp vs E0 blended |
| gpt-4o-mini + v3 prompts on TR (E2a ablation) | 47.2% TR (vs 66.9% with GPT-4o). GPT-4o is primary driver (+19.7pp). |

## QA Results (2026-04-24)

| Config | QA accuracy | Cost |
|--------|------------|------|
| gpt-4o-mini optimal K (E0 blended) | 64.0% (301/470) | $0 |
| GPT-4o MS/TR/KU + v3 prompts (E2) | **75.5%** (355/470) | $9.77 |
| E5 final (locked, single-process) | TBD | TBD |

Per-qtype E2: SSU=90.6%, MS=76.0%, Pref=66.7%, TR=66.9%, KU=66.7%, SSA=92.9%.

## Docs to read in order

1. This file
2. `WRITEUP-LONGMEMEVAL-20260423.md` — full benchmark writeup
3. `docs/DISPATCHER_DESIGN.md` — per-type index plan
4. `docs/GENERALIZABILITY_SKEPTIC.md` — 5 falsifier tests
5. `bench/RESULTS-SUMMARY.md` — full metrics table

## Next experiments

1. **E5 final measurement** — in progress 2026-04-24. Single clean process, locked SHA.
2. Per-workload index split — hypothesis: Path A improves to 85%+ on 31-case; Path B flat.
3. LOCOMO and MemoryBench — out-of-distribution validation.
4. E3 contingency: --v3-temporal-prompt flag to fix TR "today reference" error (+3-5pp TR).
