# E2a — TR Ablation: gpt-4o-mini + v3 Temporal Prompt
**Date:** 2026-04-24
**Cost:** $0.2965 (1,762,782 in / 53,444 out tokens, gpt-4o-mini)
**Reproduce:** `OPENAI_API_KEY=sk-... python3 bench/qa_eval_v3_routing.py --run-id runP-v35 --experiment-id E2a --qtypes temporal-reasoning --data-dir /path/to/LongMemEval/data --out-dir experiments/E2a-tr-ablation`
**Random seed:** temperature=0 (deterministic)
**Backend:** gpt-4o-mini (reader + grader)
**Prompt version:** v3 temporal (date arithmetic) — DEFAULT in qa_eval_v3_routing.py

## Purpose

Decompose the E2 TR gain (+26.7pp) into:
- Component A: v3 temporal prompt effect (same model mini)
- Component B: GPT-4o model upgrade effect (beyond v3 prompt)

## Result

| Config | TR accuracy | n |
|---|---|---|
| E0: v2 prompt + gpt-4o-mini | 40.2% | 127 |
| **E2a: v3 prompt + gpt-4o-mini** | **47.2%** | **127** |
| E2: v3 prompt + gpt-4o | 66.9% | 127 |

**NOTE: receipt.md was auto-generated with wrong number (51.2%) due to race condition. Correct value is 47.2% per receipt.json and data JSON.**

**Decomposition:**
- v3 prompt effect (mini vs mini): **+7.0pp** (40.2% → 47.2%)
- GPT-4o model effect (on top of v3 prompt): **+19.7pp** (47.2% → 66.9%)
- Total E2 gain: **+26.7pp** = 7.0pp (prompt) + 19.7pp (model)

Source: `experiments/E2a-tr-ablation/qa_eval_v3_E2a_runP-v35_receipt.json` (authoritative)

## Key Finding

**GPT-4o is the primary driver of the TR gain (+19.7pp = 74% of total gain). The v3 temporal prompt contributes modestly (+7.0pp = 26% of total gain).** This unambiguously resolves the prompt-model confound in E2's adversarial flag.

## Resolution of E2 Adversarial Flags 1+2

- Flag 1 (methodology change): **RESOLVED**. GPT-4o accounts for 74% of TR gain; prompt accounts for 26%. The E2 result is primarily model-driven.
- Flag 2 (Pref confound): **CONFIRMED SEPARATELY**. Pref gain (+26.7pp) is entirely from the v3 prompt (same mini model). TR's prompt contribution (+7pp) is distinct and much smaller.

## Impact on Hypothetical gpt-4o-mini + v3 prompts Blended

If all qtypes used gpt-4o-mini + v3 prompts (no GPT-4o):
- SSU: 90.6% × 64 = 58
- SSA: 92.9% × 56 = 52
- Pref: 66.7% × 30 = 20
- MS: ~67.8% × 121 = 82 (no GPT-4o uplift)
- TR: 47.2% × 127 = 60
- KU: ~63.9% × 72 = 46
**Total: ~318/470 = 67.7% — below 75% floor by 7.3pp**

**Conclusion: GPT-4o is NECESSARY to hit the 75% floor. v3 prompts + mini reaches ~68%.**

## Decision

**PROCEED TO E5 FINAL MEASUREMENT with E2 config (GPT-4o for MS/TR/KU).**

Flags 1+2 are resolved. E2's 75.5% result is primarily model-driven (+19.7pp from GPT-4o on TR alone) with meaningful but smaller prompt contributions. The configuration is justified for the paper.
