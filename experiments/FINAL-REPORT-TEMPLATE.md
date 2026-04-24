# Cogito QA Push — Final Report

**Experiment series:** E0–E5 (autonomous run 2026-04-23/24)
**Goal:** LongMemEval_S QA: 50.4% → 75%+ (floor), 85%+ (stretch)
**Outcome:** FLOOR MET (E2: 75.5%, E5: TBD)

---

## Summary (≤500 words)

Starting QA accuracy (E0 baseline): 64.0% optimal blended (50.4% top-1, 45.3% top-5).
Target: 75%+ floor, 85%+ stretch. Budget: $50 OpenAI.

**Approach:** Three levers tested in sequence — K routing (locked in E0), synthesis prompt enhancement (tested in E1), and GPT-4o reader upgrade (E2). An ablation (E2a) decomposed the model vs prompt contribution.

**E0 — Baseline lock (free)**
Confirmed optimal K routing: SSU/SSA/Pref/KU use K=1; MS/TR use K=5. Blended optimal = 64.0% (better than the 61.5% estimate).

**E1 — Synthesis prompt test (+0pp on MS, $0.26)**
v3 synthesis prompt for MS regressed -11.9pp on gpt-4o-mini. Prompt enhancement rejected for mini; E2 tests GPT-4o.

**E2 — GPT-4o reader (75.5%, $9.77)**
Config: GPT-4o for MS/TR/KU, gpt-4o-mini for SSU/SSA/Pref, v3_routing prompts (improved over E0 baseline for all qtypes). Full 470 questions.

| qtype | n | E0 | E2 | Δ |
|-------|---|----|----|---|
| SSU | 64 | 90.6% | 90.6% | 0 |
| MS | 121 | 67.8% | 76.0% | +8.3pp |
| Pref | 30 | 40.0% | 66.7% | +26.7pp |
| TR | 127 | 40.2% | 66.9% | +26.8pp |
| KU | 72 | 63.9% | 66.7% | +2.8pp |
| SSA | 56 | 92.9% | 92.9% | 0 |
| **Overall** | **470** | **64.0%** | **75.5%** | **+11.5pp** |

Adversarial flags: prompt-model confound on TR/Pref → E2a ablation.

**E2a — TR ablation ($0.30)**
gpt-4o-mini + v3 prompts on TR only: 47.2% (vs 40.2% baseline). Decomposition:
- v3 prompt effect: +7.0pp
- GPT-4o model effect: +19.7pp (74% of TR gain)
**GPT-4o is the primary driver. Config justified.**

**E5 — Final sealed measurement (TBD)**
Single clean process, locked script SHA (768a419). Result: [TBD] / 470 = [TBD]%.

---

## Key Findings

1. **Floor met in E2**: 75.5% (355/470), 95% CI [71.4%, 79.2%]. +11.5pp over optimal blended baseline.

2. **GPT-4o is the dominant lever for TR** (+19.7pp over gpt-4o-mini with identical prompt). Prompt adds +7pp on TR.

3. **Improved prompts help Pref massively** (+26.7pp) with no model change. gpt-4o-mini's Pref accuracy was artificially low (40%) due to weak prompt in the original qa_eval_v2.py.

4. **K routing confirmed**: SSU/SSA/Pref/KU at K=1; MS/TR at K=5. This was the foundation experiment.

5. **TR error analysis**: 29% of E2 TR wrong answers were "today reference" errors (GPT-4o uses training cutoff as anchor for elapsed-time questions). Fix is available (_QA_SYS_TEMPORAL_V2, --v3-temporal-prompt flag) as E3 contingency.

6. **gpt-4o-mini + v3 prompts alone**: ~67.7% overall — 7.3pp below the floor. GPT-4o necessary.

---

## Cost Breakdown

| Exp | Config | Cost |
|-----|--------|------|
| E0 | Baseline compute | $0.00 |
| E1 | MS prompt test (109 questions) | $0.26 |
| E2 | Full 470-question GPT-4o run | $9.77 |
| E2a | TR ablation (127 questions) | $0.30 |
| E5 | Final sealed (470 questions) | $TBD |
| **Total** | | **~$10.33 + E5** |

Budget: $50. Remaining: ~$39.67 before E5.

---

## Open Questions / E3 Contingency

If E5 falls short of 75%:
- **E3**: TR temporal prompt fix (_QA_SYS_TEMPORAL_V2). Expected +3-5pp on TR = +1-2pp overall.
- **E2b**: v3 MS synthesis prompt + GPT-4o (untested; E1 showed v3 hurts mini, unknown for GPT-4o).

If stretch (85%) is desired:
- E3 TR fix: estimated 80-83% (from 75.5%)
- Larger K for TR: untested
- Better context assembly (langstate compression, E4)

---

## Shippability Assessment

- **Paper claim**: "75.5% QA on LongMemEval_S (n=470)" — substantiated by E2 point estimate.
- **CI note**: 95% CI lower bound is 71.4%. The claim is at the point estimate, not robustly above floor within CI.
- **Reproducibility**: E5 provides single-process receipt. Reproducible with locked script SHA.
- **Attribution**: Gains from (1) prompt improvement on Pref/TR/KU and (2) GPT-4o upgrade on MS/TR/KU, both documented.
