# Recorded benchmark case: what hit@5 does and does not say

## Job
Read cases from `bench/hardset.json` (LongMemEval-S questions with gold session IDs and the
recorded top-five retrieved IDs) and recompute what the recorded `s1_hit_at_5` flag means.

## Files
- `input.json`: three unmodified entries from `bench/hardset.json`: `d3ab962e` (2 gold
  sessions, both retrieved), `gpt4_e061b84f` (3 gold, only 1 retrieved) and `gpt4_af6db32f`
  (1 gold, not retrieved).
- `expected.json`: per case, the gold IDs found in `s1_top5_ids`, hit@5, fraction of gold
  found, and agreement with the recorded flag.

## How it is verified (no server)
JSON parsing and set arithmetic only. The test also checks that the three entries are
identical to the ones in `bench/hardset.json` and that all 37 recorded flags agree with the
recorded ID lists. `bench/eval.py` and the LongMemEval runners are not run; they need a
server, embeddings, and the dataset.

## What you should see
`hit@5` is true when ANY gold session is in the top five. `gpt4_e061b84f` counts as a hit while
finding one of three needed sessions, so hit@5 can overstate how much of a multi-session
answer was retrieved. The third case is a plain miss.

## Provenance
Real entries from `bench/hardset.json`. The dataset is LongMemEval-S; see
`bench/DEFAULT-RETRIEVAL-METHODOLOGY.md`.

## Limits
- The 37-question hardset was chosen from failures of an earlier pipeline run on the same
  evaluation set (`bench/BENCHMARK_INTEGRITY_AUDIT.md`, section 5; `bench/VALIDATION_PACK.md`).
  It is test-set-informed, so it is not an unbiased sample and not a headline number.
- The recorded IDs come from historical (April 2026) pipeline runs, not from the 0.3.0rc1
  default retrieval path. `AGENTS.md` asks that historical benchmark headlines not be reused as
  evidence for that path.
- Retrieval of a session is not answer correctness; nothing here measures generated answers.
- `bench/eval_cases.json` uses a different, keyword-based format (`expected` keywords, scored
  by `bench/eval.py` against a live server) and is not used here.
