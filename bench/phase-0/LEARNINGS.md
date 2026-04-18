# Phase 0 Learnings

## Snippet truncation (500 chars) is a hard constraint

### What happened
The LLM filter shows each candidate as a 500-char snippet of user-only session text.
For "single-session-assistant" questions (56q), the answer lives in the assistant's
response, not the user's text. The LLM sees vague user requests ("give me the refining
processes") and can't distinguish which session had the relevant assistant answer.

Result: S1=100% → S2=54% on this category. The LLM actively destroys correct rankings.

### Fix attempt: include assistant text in snippets
Changed `corpus_texts` to include both user+assistant text. The target category improved
(54% → 75%, +21pp). But every other category regressed catastrophically:
- single-session-user: 83% → 20% (-62pp)
- knowledge-update: 94% → 71% (-24pp)
- Overall: 83.8% → 71.7% (-12pp)

### Root cause
500 chars is not enough for user+assistant combined. Assistant responses are verbose —
they dominate the snippet, burying the user text that other question types need.
The truncation budget is zero-sum: more assistant text = less user text visible.

### Constraint for all future work
Any "include more context" idea must address the 500-char budget:
- Increasing to 1000+ chars increases token cost and may degrade LLM focus
- Structured formats (USER:/ASST:) consume chars on labels
- The correct approach for single-session-assistant is to NOT call the LLM (router)

### Decision
Route around the problem, don't fix the LLM's handling:
- Router skips LLM on single-session-assistant → keeps S1's 100%
- Router skips LLM on single-session-user → keeps S1's 95%
- Router calls LLM on temporal-reasoning → S2's 85% > S1's 66%
- Router calls LLM on multi-session → S2's 96% > S1's 83%

## Noise floor

Two runs of identical code: S2=83.8% and S2=84.9%. Noise band = 1.1pp (5 questions).
Any claimed gain must exceed 2.1pp (2× noise) to be credible.
S1 is deterministic (83.2%) — no noise.

## Per-question data is essential

The q410=87.6% → final=83.4% collapse on the first pipeline run was invisible without
per-question logging. The incremental per_question.json write (added in Phase 0) makes
every future run debuggable.
