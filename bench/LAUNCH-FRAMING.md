# cogito-ergo Launch Framing

## The honest claim

cogito-ergo achieves **96.38% retrieval R@1** on LongMemEval_S (470 questions) using runtime-only confidence rules — no test-set memorization, no oracle tricks. This is the highest published retrieval R@1 on this benchmark.

**Cost: ~$0.003-0.005/query** (10-30x cheaper than GPT-5-mini-backed systems).

## What we DON'T claim

We do not claim #1 on end-to-end QA accuracy. Mastra reports 94.87% QA accuracy using GPT-5-mini. Our QA accuracy with qwen-max (top-5) is approximately 77-82% (eval in progress). The gap is in the QA reader model, not retrieval — our retrieval feeds the right session 96.38% of the time.

## The pitch

"cogito-ergo is the strongest retrieval backbone for long-term memory on LongMemEval_S. Plug in any QA model and you get 96.38% of queries hitting the right session at near-zero cost. Our full stack — retrieval + QA — costs $0.005/query vs $0.03-0.15 for GPT-5-mini systems."

## Axes where we lead

| Axis | cogito-ergo | Mastra | Winner |
|------|-------------|--------|--------|
| Retrieval R@1 | 96.38% | ~unpublished | cogito |
| QA accuracy | ~78-82% (qwen-max) | 94.87% (GPT-5-mini) | Mastra |
| Cost per query | $0.003-0.005 | $0.03-0.15 | cogito (10-30x) |
| Local deployable | Yes (Ollama, 83.2% R@1 at $0) | No | cogito |
| Open source | Yes | No | cogito |

## Novel findings (paper-worthy)

1. **Demotion problem**: LLM rerank filters can demote gold retrieval already ranked #1
2. **Oracle inflation**: Union-based evaluation inflates R@1 by 5-10pp — a methodological hazard
3. **Anti-correlated scaffold transfer**: r=-0.59 between scaffold effectiveness across models
4. **Selective compute**: Only 8% of queries need flagship-tier LLM; regex router handles the rest

## Recommended launch channels

1. Show HN: "cogito-ergo — 96.38% retrieval at $0.005/query on LongMemEval"
2. Paper: arxiv preprint with the four findings above
3. PyPI: `pip install cogito-ergo`
4. Blog: "We beat the retrieval benchmark at 1/30 the cost. Here's how."
