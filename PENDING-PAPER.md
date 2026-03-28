# Pending Paper: Integer-Pointer Retrieval and the Fidelity Decomposition

**Status:** Idea stage. Not written. Flagged for future work.
**Origin:** Emerged during cogito-ergo development, March 2026.
**Repo:** roli-lpci/cogito-ergo
**Related:** roli-lpci/memory-exploration-march27

---

## The Core Idea

Most retrieval-augmented memory systems have a fidelity problem they don't name.

When a re-ranker or filter LLM is asked "which of these memories are relevant?", the
standard implementations ask it to *output* the relevant memories — either by summarising
them, quoting them, or scoring and re-ordering them. All of these approaches create a
surface for the LLM to corrupt the content: paraphrase drift, hallucinated details merged
in, dropped qualifiers, wrong entity names preserved but context swapped.

The insight behind cogito-ergo's two-stage pipeline is that you can eliminate this surface
entirely by changing *what the LLM is allowed to output*.

> **The integer-pointer fidelity guarantee:**
> The filter LLM receives numbered candidates. It outputs *only integer indices*.
> The server selects verbatim text from Stage 1 by those indices.
> The LLM never touches the output content — structurally, not by instruction.

This isn't "use structured output" or "constrain the model". It's a pipeline decomposition
where the LLM's role is **classification**, not **generation**, and fidelity is a property
of the architecture rather than a property of the prompt.

---

## Full Decomposition

The pipeline separates three concerns that most RAG implementations conflate:

### Stage 1 — Recall ceiling removal
- Broad vector search, top-N by rank
- **No threshold cut.** This is deliberate.
- Standard RAG applies an L2/cosine threshold here. That threshold creates a hard ceiling
  on recall: the right memory can never surface if it's cut before Stage 2 sees it.
- Stage 2 exists to handle precision. Cutting for precision at Stage 1 defeats Stage 2.

### Stage 2 — Precision via integer-pointer filter
- Cheap LLM (local qwen2.5:1.5b, or Haiku for cloud) receives the candidate list as
  `[1] text... [2] text...`
- System prompt: output ONLY a JSON array of integers. No explanation.
- Model outputs: `[1, 4, 7]` or `[]`
- Thinking tokens (QwQ, DeepSeek) stripped before parsing
- Invalid indices, out-of-range values silently dropped
- On parse failure: fallback returns all candidates (degrades gracefully)

### Stage 3 — Verbatim selection
- Server picks `candidates[idx - 1]` for each selected index
- Returns original text, original score, nothing generated
- The returned content was never seen by the filter LLM's output path

Each stage has a single responsibility. Recall, precision, and fidelity are orthogonal.

---

## Why This Might Be Worth a Paper

### 1. The insight is named and precise
"Integer-pointer fidelity" is a structural guarantee, not a best practice. You can prove
it: if the LLM output path contains only integers in range, the returned text is byte-for-
byte identical to Stage 1 output. That's a provable property, not a probabilistic one.
Most safety/fidelity claims in RAG literature are empirical ("hallucination rate dropped").
This one is architectural.

### 2. It solves a real production failure mode
The zer0dex incident (March 2026) demonstrated the failure mode concretely: a re-ranking
step that asked the LLM to return relevant memories produced subtly wrong facts — correct
topic, wrong detail. The integer-pointer approach makes this class of failure impossible
by construction.

### 3. The Stage 1 threshold insight is underappreciated
The removal of the threshold at Stage 1 is non-obvious and counterintuitive. Every RAG
tutorial recommends a similarity threshold to "filter noise". That advice is correct for
single-stage retrieval but harmful for two-stage retrieval — it moves the precision gate
to the wrong stage and kills recall ceiling. This is worth naming explicitly in literature.

### 4. The write-side corollary is clean
`/store` (verbatim, agent-decided) vs `/add` (extraction LLM) maps onto a clean
principal hierarchy: *who decides what gets remembered?*
- `/add`: the extraction LLM decides (consumer chatbot pattern)
- `/store`: the agent decides (autonomous agent pattern)
Most memory systems only offer the `/add` pattern. The distinction matters for agents
operating in technical domains where extraction quality degrades.

### 5. Empirical results are decent for a short paper
- 87.5% Recall@1, 100% Recall@3, MRR 0.938 on a 2,819-memory technical corpus
- +25pp Recall@1, +37.5pp Recall@3 vs naive vector search
- qwen2.5:1.5b (1.5B params, local, $0) as the filter model
- These aren't SOTA numbers, but for a position/systems paper they support the claim

### 6. The decomposition could generalise
The same integer-pointer pattern applies anywhere an LLM needs to select from a fixed set:
tool selection, skill routing, multi-hop retrieval candidate selection. The paper could
frame this as a general pattern for LLM-in-the-loop selection tasks where output fidelity
matters.

---

## Why It Might Not Be Worth It

### 1. The pieces are all known
Re-ranking is not new. LLM rerankers are not new. Structured output is not new. A reviewer
could reasonably say "this is just constrained decoding applied to RAG" and not be wrong.
The *synthesis* and *naming* is the contribution — and synthesis papers have a harder time
clearing the novelty bar at top venues (NeurIPS, ICLR, ACL).

### 2. Eight test cases is not a benchmark
Recall@1 on 8 hand-crafted queries against a self-built corpus is not publishable as an
evaluation. A real evaluation needs: standardized dataset (LOCOMO, MemGPT eval, or
similar), blind queries, multiple baselines, and ideally a human preference study.
Without this, the empirical section would be thin.

### 3. Someone may have already published this exact framing
We haven't done a thorough literature search. The closest named concepts are:
- RAG-Fusion (multi-query, union, RRF reranking) — different problem, similar decomposition
- FLARE (forward-looking active retrieval) — different mechanism
- Tool use / function calling patterns (LLM selects tool ID, tool executes) — closest analog
- FiD (Fusion-in-Decoder) — multi-document, different fidelity concern
It's possible something under "constrained re-ranking" or "index-based selection" covers
this. Unknown until a proper search is done.

### 4. The claim is narrow
The fidelity guarantee holds only when the LLM output is *only integers*. With any
other structured output format (JSON object, ranked list with reasons, etc.), the
guarantee breaks. The scope of the contribution is therefore limited to this specific
output constraint. That may not be enough for a full paper — more likely a short paper,
workshop note, or well-placed blog post.

### 5. Time / ROI
A proper paper (lit search, standardized eval, writing, submission, review cycle) is 3-6
months of work. That time could go into building more of the system. The ideas can be
captured in this file and the research repo without committing to the publication track.

---

## Evidence to Gather (if pursuing)

1. **Lit search:** Search for "integer pointer selection LLM", "constrained reranking",
   "index-based retrieval filter", "LLM selection fidelity". Use Semantic Scholar + arXiv.
2. **Baseline comparison:** Run the same 8 queries + expanded set against:
   - mem0 default (extraction + query)
   - Standard RAG with LLM reranker (LlamaIndex)
   - Cohere Rerank API
3. **LOCOMO eval:** Seed from LOCOMO training split, evaluate on test questions.
   LOCOMO has ~300 questions across 5h of conversation — reasonable eval target.
4. **Scale test:** Does Recall@3 hold at 10k, 50k memories? The filter receives top-N
   candidates from Stage 1, so it should scale — but needs verification.
5. **Failure mode analysis:** Deliberately craft adversarial inputs where a standard
   LLM reranker would corrupt output. Show integer-pointer is immune by construction.

---

## Recommended Format

If pursued: **short paper or workshop submission**, not a full conference paper.

Venue candidates:
- EMNLP Findings (memory/retrieval track)
- NAACL workshop on LLM agents
- Arxiv pre-print (lowest bar, gets the idea timestamped and citable)

An arXiv pre-print is the minimum viable version: 4-6 pages, the decomposition argument,
the empirical results from LOCOMO, and the failure mode analysis. Doesn't require
acceptance anywhere to be useful — just gets the framing on record.

---

## What This Is Not Claiming

- Not claiming SOTA retrieval performance
- Not claiming the integer-pointer mechanism is novel in isolation
- Not claiming the two-stage pipeline is new

**The claim:** The decomposition into recall-ceiling / precision / fidelity as three
orthogonal concerns — with the structural fidelity guarantee achieved by integer-pointer
output — is a clean, nameable, and underspecified contribution to how practitioners build
memory systems for autonomous agents. Whether it clears a novelty bar at a top venue is
an open question. Whether it's worth writing down clearly: yes.

---

*Last updated: 2026-03-28*
*Author note: This document was written to capture the idea before it evaporates. It is not a commitment to publish.*
