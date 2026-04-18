# Retriever Contribution Analysis

Based on Step 1 (gapfix) — 470 questions, 37 hardset

## Key Finding

- 29/37 hardset questions have gold in S1 top-5 (78%)
- Only 8/37 are retrieval failures (gold NOT in top-5)
- **The remaining gap is a ranking problem, not a retrieval problem**
- This confirms the hubris agent's analysis: flagship model on top-5 is the path

## Per-Type Verdict

| Type | N | S1 R@1 | S1 R@5 | Gap | Verdict |
|---|---|---|---|---|---|
| temporal-reasoning | 127 | 66% | 94% | 28% | ranking |
| multi-session | 121 | 83% | 100% | 17% | ranking |
| knowledge-update | 72 | 96% | 100% | 4% | near-solved |
| single-session-user | 64 | 95% | 100% | 5% | near-solved |
| single-session-assistant | 56 | 100% | 100% | 0% | near-solved |
| single-session-preference | 30 | 67% | 97% | 30% | ranking |
