# LPCI Temporal Scaffold Design

## Hypothesis
Inject PRE-COMPUTED pairwise temporal ordering into the LLM filter prompt
for temporal queries only. The LLM doesn't reason about time — the scaffold
carries temporal state as language.

## How it differs from Step 2 (date injection at chunk level — FAILED)
Step 2 prepended "2023/05/20: " to every chunk before BM25+embedding.
This added noise to non-temporal queries and consumed truncation budget.
Result: net zero (+2pp temporal, -2pp elsewhere).

The temporal scaffold is different:
1. Only activates on temporal queries (router already classifies these)
2. Only affects the LLM filter prompt (not embeddings or BM25)
3. Provides PAIRWISE ORDERING (relative days between candidates), not raw dates
4. Pre-computes the temporal structure — the LLM reads it, doesn't derive it

## Scaffold format
For temporal queries, the filter prompt gets an extra block before the candidates:

```
Temporal context (candidates ordered by date):
  [1] 2023/05/20 (earliest)
  [3] 2023/06/15 (+26 days after [1])
  [5] 2023/08/02 (+74 days after [1])
  [2] 2023/08/14 (+86 days after [1])
  [4] 2023/09/01 (+104 days, most recent)
```

The numbers [1]-[5] match the candidate numbers in the filter prompt.
This tells the LLM which candidate is oldest/newest and the gaps between them.

## Expected impact
- Temporal-reasoning currently 82% routed → target 87%+
- The 15 temporal questions in the hardset mostly fail because the LLM
  can't determine temporal order from text snippets alone
- With the scaffold, temporal ordering is GIVEN — the LLM just has to
  match the query's temporal reference to the right position

## LPCI connection
This is LPCI applied to retrieval: the scaffold is a language artifact
that carries state (temporal ordering) which the stateless LLM cannot
derive from the snippets alone. The model reads the state, doesn't compute it.
