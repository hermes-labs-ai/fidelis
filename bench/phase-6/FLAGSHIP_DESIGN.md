# Flagship Escalation Design

## Scope
On router-missed hardset questions only (~37q), call qwen-max with FULL
session text (up to 2000 chars) for top-5 candidates. Compare to current
qwen-turbo on 500-char snippets.

## Why this should work
- 29/37 hardset questions have gold in S1 top-5 (ranking problem)
- Current qwen-turbo sees 500-char snippets — often insufficient
- qwen-max with 2000 chars sees 4x more context per candidate
- 5 candidates × 2000 chars = 10K chars total — well within qwen-max context

## Cost estimate
- qwen-max: ~$0.02/1K input tokens, ~$0.06/1K output tokens
- Per question: ~3K input tokens (system + query + 5×2000 chars) + ~50 output tokens
- Cost per question: ~$0.06 input + ~$0.003 output = ~$0.063
- 37 hardset questions: ~$2.33 total
- Full 470q if deployed on routed subset (~205q): ~$12.90
- If gated to uncertain cases only (~50q): ~$3.15

## Decision criterion
If qwen-max on hardset gets ≥10/37 additional correct (from 0 to 10+),
that's +2pp overall. Ship it on the gated path.

## Design
1. Same pipeline as step1-gapfix
2. When route_decision = "llm", check if this is a hardset-class question
   (low confidence or temporal)
3. If yes, escalate to qwen-max with 2000-char snippets instead of
   qwen-turbo with 500-char
4. Fallback: if qwen-max fails/times out, use qwen-turbo result
