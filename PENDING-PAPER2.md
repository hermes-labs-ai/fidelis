# Pending Paper 2: End-to-End Fidelity Decomposition for Agent Memory Systems

**Status:** Idea stage. Case study in hand. Not written.
**Origin:** Emerged during cogito-ergo + zer0lint session, 2026-03-28.
**Repos:** roli-lpci/cogito-ergo, roli-lpci/zer0lint, roli-lpci/zer0dex
**Related:** PENDING-PAPER.md (integer-pointer retrieval fidelity — the retrieval half of this)

---

## The Core Idea

Most agent memory benchmarks measure the wrong thing.

They measure **retrieval quality** — given what's in the store, how well can you find it?
They do not measure **ingestion quality** — given what happened, how well was it stored?

These are two independent failure modes. Both are silent. Both are fixable. Most
practitioners only see one of them.

```
raw signal (conversations, logs, notes)
        ↓
   [INGESTION LAYER]
   extraction LLM summarises raw text → structured facts
        ↓  ← Failure Mode 1: facts corrupted, dropped, or misrepresented at write time
   vector store
        ↓
   [RETRIEVAL LAYER]
   query → candidates → filter/reranker → returned context
        ↓  ← Failure Mode 2: filter LLM corrupts retrieved text at read time
   agent context
```

The integer-pointer fidelity guarantee (PENDING-PAPER.md) eliminates Failure Mode 2
by construction. But Failure Mode 1 — ingestion corruption — is upstream, independent,
and equally silent.

**The claim:** End-to-end fidelity requires decomposing the pipeline into both failure
modes and addressing each with the right diagnostic. A retrieval benchmark is only
meaningful after ingestion health is confirmed.

---

## The Case Study (Reproducible, 2026-03-28)

During development of cogito-ergo, zer0lint was run against the live corpus:

- **System:** cogito-ergo, collection `cogito_main`, ~500 memories
- **Extraction model:** mistral:7b (mem0 default)
- **zer0lint result:** 0/5 (CRITICAL)
- **Root cause:** mistral:7b returning malformed JSON (`Unterminated string starting at
  line 1 column 10`) from mem0's extraction prompt. mem0 silently drops the facts and
  stores degraded fallbacks.
- **Evidence in store:** memories like "tooling discussed" — mem0's fallback when
  extraction fails, zero retrieval value
- **Retrieval benchmark at the same time:** 96% hit@any (bench/eval.py, 31 cases)

The 96% retrieval number and the 0% ingestion number coexisted in the same system
simultaneously. Neither diagnostic alone told the full story.

### Why It Was Invisible

1. `memory.add()` returned `{"results": [...]}` — looked like it worked
2. `memory.search()` returned results — looked like retrieval worked
3. Retrieval benchmark showed 96% hit@any — numbers looked healthy
4. The ingestion failure was silent at every layer

The retrieval benchmark was measuring how well the system retrieves what's stored.
It had no mechanism to detect that what's stored was degraded. The eval adapted to
the bad corpus (dynamic cases were generated FROM stored memories) and reported
healthy numbers against a broken store.

### The Meta-Point

The author of zer0lint ran their own diagnostic tool against their own system and
caught themselves doing the thing zer0lint was built to catch. This is not a
hypothetical failure mode — it's a documented, reproducible incident with a named
system, a timestamped corpus, and a working diagnostic.

---

## Why Most People Miss This

The standard integration path for any mem0-based system:

1. `pip install mem0ai`
2. `memory.add(text, user_id=...)` — returns `{"results": [...]}` ✓
3. `memory.search(query, user_id=...)` — returns memories ✓
4. Build a retrieval benchmark — numbers look decent ✓
5. Ship

Step 2 is where the silent failure happens. The extraction LLM (often mistral:7b,
llama3:8b, or similar local model) receives the raw text and is expected to return
structured JSON facts. If the model outputs malformed JSON, mem0 has no hard failure
— it logs the error internally and stores a degraded fallback. The caller sees a
`{"results": [...]}` response and has no way to know the facts were dropped.

The only way to catch this is a dedicated ingestion health check: inject known facts,
retrieve them, score the round-trip. This is exactly what zer0lint does.

---

## Full Pipeline Fidelity Score

The complete fidelity picture requires two numbers:

| Layer | Diagnostic | Metric | Failure mode |
|---|---|---|---|
| Ingestion | zer0lint check | extraction recall (%) | facts dropped or corrupted at write time |
| Retrieval | bench/eval.py | hit@any, R@1, MRR | relevant memories not surfaced |
| Retrieval content | integer-pointer architecture | structural guarantee | filter LLM corrupts returned text |

A system reporting only retrieval metrics is reporting one of three numbers.
The end-to-end fidelity score is the product: all three must be healthy.

---

## Framing Options

### Option A: Systems paper — "End-to-End Fidelity for Agent Memory"
Frame the full pipeline decomposition: ingestion + retrieval + content fidelity.
Propose a three-layer diagnostic. Present cogito-ergo + zer0lint as the reference
implementation. Length: 6-8 pages. Venue: EMNLP Findings, NAACL agent workshop.

### Option B: Position paper — "Your Memory Benchmark is Measuring the Wrong Thing"
Provocative framing: most retrieval benchmarks for agent memory are invalid because
they measure retrieval quality on unchecked corpora. The ingestion failure is the
confound nobody controls for. Short, citable, blog-post energy. Length: 4 pages.
Venue: arXiv pre-print, ACL workshop.

### Option C: Combined paper with PENDING-PAPER.md
Merge both: integer-pointer fidelity at the retrieval layer + ingestion fidelity
diagnostic as the upstream companion. One paper covering both failure modes with a
unified "fidelity decomposition" framing. The cogito-ergo + zer0lint stack is the
full reference implementation. Strongest empirical story.

**Recommended:** Option C or Option B. Option B has the clearest hook and lowest
word count requirement. Option C has the most complete technical contribution.

---

## What This Is Not Claiming

- Not claiming mem0 is broken (it has an extraction prompt API to fix this)
- Not claiming mistral:7b is broken (it's a chat model being used for JSON extraction)
- Not claiming zer0lint is the only diagnostic (it's one implementation of a pattern)

**The claim:** The failure mode is structural to the design of extraction-based memory
systems, silent by default, and undetectable by retrieval benchmarks. It needs to be
named, measured, and addressed as a first-class concern alongside retrieval quality.

---

## Evidence in Hand

- [x] Live case study: cogito_main corpus, 0/5 ingestion, 96% retrieval (2026-03-28)
- [x] Root cause identified: mistral:7b malformed JSON, silent mem0 fallback
- [x] Working diagnostic: zer0lint (roli-lpci/zer0lint, PyPI)
- [x] Working fix: TECHNICAL_EXTRACTION_PROMPT in zer0lint (validated 5/5 vs 2/5 default)
- [ ] Pre/post comparison: run zer0lint generate → re-seed → re-eval (next step)
- [ ] Broader survey: how common is this across other local models?
- [ ] LOCOMO or standardized eval to complement hand-built benchmark

---

*Created: 2026-03-28*
*Status: case study complete, paper not started*
