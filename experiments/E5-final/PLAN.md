# E5 — Final Sealed Measurement

**Status:** RUNNING
**Date:** 2026-04-24
**Purpose:** Single clean process run of the E2-proven config. No resume, no fragmentation.

## Config (locked)

- **Run ID:** runP-v35
- **GPT-4o reader:** MS (K=5), TR (K=5), KU (K=1)
- **gpt-4o-mini reader:** SSU (K=1), SSA (K=1), Pref (K=1)
- **Prompts:** v3_routing.py defaults (v2 MS, v3 temporal, v3 Pref, v3 KU)
- **Grader:** gpt-4o-mini
- **max_tokens:** 512
- **rate_limit_delay:** 0.4s
- **Script SHA:** 768a419252a7a9cbcd9542d1eaaeac3a874597a3 (bench/qa_eval_v3_routing.py at commit 6c30bb1)

## Why E5 vs E2

E2 (75.5%) was:
- Fragmented across 3 sessions with --resume
- Had multiple concurrent processes race
- reproduce_command was wrong (showed only last sub-run)

E5 is:
- Single clean process, no --resume
- Locked commit SHA
- Correct reproduce_command

## Expected Result

~75% ± 3% (matching E2's 75.5% within noise of n=470)

## Decision Gate

- If E5 ≥ 75%: result confirmed. Write final report.
- If E5 is 72-75%: within E2's CI. Report as "75%" with E2 as primary evidence, E5 as confirmation attempt.
- If E5 < 72%: something is wrong. Investigate before reporting.
