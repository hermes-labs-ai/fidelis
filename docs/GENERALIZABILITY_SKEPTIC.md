# cogito-ergo Generalizability Skeptic Review

**Status**: Critical gaps found in the workload-aware dispatcher plan. Both retrieval paths are likely overfit to their training distributions.

---

## Executive Summary

Path A (/recall, atomic-fact) scores **75% R@1 on the 31-case eval** (the only real production data you have).
Path B (/recall_hybrid, LongMemEval-optimized) scores **54% R@1 on those same 31 cases** — a **21-point regression**.

The plan to reconcile them via a "workload-aware dispatcher" risks building a 2-way overfit: Path A is tuned to atomic keyword recall; Path B is tuned to session-based conversational retrieval. Neither is validated on the OTHER's home turf, and neither is validated on realistic new workloads.

Shipping a 95% claim requires proving the dispatcher generalizes beyond these two benchmark distributions. It doesn't yet.

---

## 1. Where the Workload-Aware Dispatcher Still Overfits

### Failure Mode A: The "Medium-Length" Memory Gap
Path A expects atomic facts ≤150 chars (judged by the filter snippet truncation at `[:150]`).
Path B expects session context ≤2000 chars (the flagship snippet limit in `recall_hybrid.py:464`).

**What happens at 500–800 chars?**
- A 600-char memory (e.g., a design decision with rationale) is too long for Path A's filter to meaningfully evaluate in isolation.
- It's too short for Path B's BM25+dense+RRF fusion to benefit from field-level structure.
- Neither the regex router nor the dispatcher has a case for "medium-form narrative."

**Test case**: Store a 750-char decision memo. Query for a detail in paragraph 3. Neither path is validated here.

### Failure Mode B: Structured But Atomic Data
Memories that are JSON, code, or YAML blobs (but still logically atomic) break both routers.

Example: `{"auth_flow": "JWT→session", "ttl_sec": 3600, "rotation_policy": "on_login"}` — 89 chars.

Path A sees it as unstructured text; the filter LLM will struggle to parse it.
Path B's BM25 + dense hybrid will tokenize the keys/values separately, hurting both keyword and semantic match.

**Evidence**: IMPROVEMENT-COMPARISON.md notes 27 LongMemEval sessions silently failed due to JSON/hex truncation bugs. You know structured data breaks the pipeline. The dispatcher has no answer for it.

### Failure Mode C: The Dispatcher Decision Tree Itself
The plan mentions "workload-aware dispatcher" but provides no specification:
- What query characteristics trigger Path A vs Path B?
- The router in `recall_hybrid.py:113–130` is regex-only (hardcoded skip/temporal/counting patterns). This is brittle.
- If a user queries "how many times did we change the auth flow," it routes to "llm" (counting pattern). But if the memory is a short JSON blob, the filter LLM is the wrong tool — Path A's integer-pointer might be better.
- **No test** validates that the dispatcher routing decision correlates with which path actually performs better on the same query.

---

## 2. What the 31-Case Eval Doesn't Measure

### The 31-case eval is representative of ONLY atomic keyword recall
Reading `eval_cases.json`:
- 4 "cross_reference" cases (e.g., "what tools are being built")
- 4 "semantic_gap" cases (e.g., "why was nothing being found")
- 4 "adversarial_negative" cases (e.g., "hardware spec of the machine running ollama")

**This corpus is from cogito's own memory store** — small, technical, fact-dense. It is NOT representative of:

#### Adversarial workloads
- **Typo-heavy user input**: "authorizatin architecture" (typo in "authorization"). Path A relies on keyword matching; the filter LLM might recover via semantic understanding. **Not tested.**
- **Paraphrase variance**: User asks "when did we migrate tokens" but memory says "switched from JWT to session on 2026-03-27." Path A's vocab_map helps; Path B's BM25+dense might miss the semantic pivot. **Not tested.**
- **High-cardinality distractor**: 1000 memories about "auth" but the user asks "which auth timeout did we pick?" Path A's filter is fast but noisy at scale; Path B's BM25 helps but hasn't been validated on >500 memories. **Scale test: missing.**

#### Multilingual or code-heavy
- Path A assumes the filter LLM speaks the corpus language. Path B's BM25 is language-agnostic. **No multilingual test.**
- Code snippets (Python, YAML, SQL) are not in the 31-case eval. Both paths are unvalidated on code. **Code retrieval benchmark: missing.**

#### High-volume / concurrent
- **100K+ memories**: BM25 indexing and dense re-embedding scale linearly. Does the router decision change when you have 100K candidates instead of 50? **No load test.**
- **Concurrent writes**: The dispatcher makes routing decisions per-query. If memories are being added concurrently, does the routing become unstable? **Concurrency test: missing.**

### Silent failures in the eval
The eval includes 4 adversarial_negative cases expecting empty results (`[]`). Do both paths correctly return `[]`? The README doesn't break down per-case per-path results.
**Missing**: per-case breakdown showing both Path A and Path B results.

---

## 3. What LongMemEval_S Doesn't Measure

### Session-based ≠ production multi-workload
LongMemEval is **470 questions over multi-turn dialogs with session-level chunking**. The benchmark assumes:
- Memories are long (up to 6000 chars before truncation).
- Related facts cluster within a session.
- There are explicit session boundaries and temporal metadata (`haystack_dates`).

Production cogito:
- Memories are atomic, unrelated, injected by agents over time.
- No session boundaries.
- No `haystack_dates` field (Path B removes this adaptation in production; see `recall_hybrid.py:20–25`).

**The porting introduces unknown risk**:
- The 93.4% result relied on turn-level chunking (`bench/longmemeval_combined_pipeline_flagship.py` chunks by conversation turn).
- Production cogito has no turn boundaries.
- You have NOT validated that the ported pipeline (without turn chunking) maintains 93% on production data.

### Distribution of question types is unknown
METHODOLOGY.md doesn't report the breakdown of qtype within LongMemEval_S:
- How many are "preference" (67% of the 16 hardest misses)?
- How many are "multi-session aggregation"?
- What % require temporal reasoning (Path B has special-cased temporal routing)?

Without this, you don't know if the dispatcher is solving a real workload problem or overfitting to LongMemEval's question distribution.

### The "hardset" is a red flag
Path B required a **hardset of 37 questions escalated to qwen-max** (flagship tier) to reach 93.4%. This is:
1. A sign that 83.2% zero-LLM wasn't enough for the long tail.
2. **Not validated on cogito's 31-case eval**. Did the flagship tier run on those 31 cases? The README says Path B scores 54% on the 31-case eval, but doesn't break down whether that's with or without flagship.

---

## 4. Five Concrete Falsifier Tests

These tests would reveal whether the dispatcher is actually generalizable or just a 2-way overfit.

### Test 1: Paraphrase Perturbation
**Setup**: Take the 31-case eval. For each query, generate 3 paraphrases using an LLM (no manual work).
Example: "what tools are being built" → "which projects are under development" → "list the software being constructed."

**Run**: Query each paraphrase against both Path A and Path B. Score: at least one variant recalls the right answer.

**Failure threshold**: Any query where Path A ≤50% paraphrase-recall or Path B ≤70% → the dispatcher is sensitive to phrasing, not robust.

**Why it matters**: Real users don't use your exact 31 canned queries. If the paths diverge under paraphrase, the dispatcher's routing is fragile.

---

### Test 2: Typo Injection
**Setup**: Take 10 queries from the eval. Inject 1–2 character-level typos or Levenshtein edits (e.g., "authorizatin", "what toools").

**Run**: Query both paths. Score: Path A ≤60% recall, Path B ≤80% recall indicates typo sensitivity.

**Failure threshold**: Either path loses >3pp on a typo-heavy subset.

**Why it matters**: Real agent queries come from LLM generation, which hallucinate typos. If the dispatcher can't route typo-heavy queries reliably, it fails in production.

---

### Test 3: Memory Type Shift
**Setup**: Create 3 synthetic memory stores:
1. **Atomic keywords** (existing 31-case store): facts ≤150 chars.
2. **Medium narrative** (new): 3–5 sentence design decisions, 200–600 chars each.
3. **Code snippets** (new): Python/YAML/SQL fragments, 100–400 chars, unstructured.

**Run**: Same 10 queries against all 3 stores. Measure Path A vs Path B R@1 per store.

**Failure threshold**: Any path loses >5pp on a new memory type → the dispatcher is not memory-type-agnostic.

**Why it matters**: Real agent memory stores are heterogeneous. If the dispatcher assumes homogeneous memories, it fails at the first type change.

---

### Test 4: Cross-Language Query + Bilingual Store
**Setup**: English queries, but 30% of the store is in Spanish / Mandarin / French (realistic for global agents).

**Run**: Mixed-language queries (e.g., "how do we handle auth tokens" alongside "como manejamos auth tokens"). Measure recall.

**Failure threshold**: Either path loses >4pp on cross-language vs. single-language → language sensitivity is unhandled.

**Why it matters**: Multi-agent systems often span languages. The dispatcher has no language awareness.

---

### Test 5: High-Cardinality Distractor at Scale
**Setup**: Seed the store with 2000 memories, 80% of which contain the word "auth" (high-cardinality class). Add 10 queries that target specific auth-related memories.

**Run**: Query both paths. Measure recall@1 and recall@5. Compare to the 31-case baseline.

**Failure threshold**: Either path loses >4pp on recall@1 vs. the 31-case baseline → the dispatcher doesn't scale.

**Why it matters**: Real memory stores grow. If the performance cliff happens at scale, the dispatcher is a local-eval artifact.

---

## 5. What Competitors Probably Measure (and Don't Publish)

### Mastra (94.87% R@1, the published SOTA)
**Likely hidden benchmarks:**
- Multi-turn **real chat logs** (not curated datasets like LongMemEval). Chat logs are messier, have more typos, and cross-context references.
- Latency profiling at **100K+ memories**. Vector DB scaling kills retrieval quality; Mastra probably has a special case for it.
- **Failure modes by memory type**: code, metadata, long-form prose. Mastra's 94.87% might be high on textual data, lower on code.
- **Ablation on their router/dispatcher logic**. They almost certainly have a query classifier (like cogito's regex router). That classifier is probably the biggest source of accuracy variance.

**Their blind spot**: Likely haven't tested on truly adversarial workloads (typos, paraphrases, code, scale).

### Letta, MemGPT
**Likely hidden limits:**
- MemGPT's compression stage probably overfits to **short-context recall**. Long chains of reasoning degrade.
- Letta's integration with LLMs probably assumes the user query is **semantically close to the memory text**. Paraphrase mismatches likely hurt.

### Zep
**Likely haven't solved:**
- **Recency bias**: Recent memories are probably over-weighted in their router. Query an old fact from month 1, you might miss it on a large store.
- **Multi-type stores**: If 50% of the store is metrics/logs and 50% is design docs, their retrieval probably underperforms on the minority type.

---

## 6. The One Thing Most Likely to Kill This Generalizability Claim

**Answer: The 31-case eval regression (54% R@1) is not being treated as a blocker.**

Path B is the new hotness — 96.4% on LongMemEval, published in the README, deployed to `/recall_hybrid`. But on the only *production eval you actually have*, it's **21 points worse than Path A**. 

The plan to ship a "workload-aware dispatcher" glosses over this: "use Path B when you need the hybrid trait" is a shrug, not a solution. You're asking users to decide which path to call, which means users have to benchmark both on their own workload.

**Why this kills the 95% claim**:
1. You can't claim 95% if 54% of production queries route to Path B.
2. You can't measure the dispatcher's routing accuracy without a ground-truth label of "which path *should* have been called." You don't have that.
3. The 31-case eval is *noisy* (only 31 cases), so the 54% might have >10pp confidence interval. You need more production data.

**Fix (or admit loss)**:
- Either: Re-validate Path B on production-like data before claiming 93.4% applies to /recall_hybrid in production.
- Or: Admit that both paths are research artifacts, publish the numbers honestly ("Path A: 75% on production, Path B: 54% on production; 96.4% on LongMemEval_S with novel chunking scheme not ported to production").
- Or: Build a third path that fixes the 21pp regression on production data, then validate both paths on both benchmarks.

---

## Recommendations

1. **Mandatory falsifier tests before launch**: Run at least Test 1 (paraphrase perturbation) and Test 5 (high-cardinality scale) before claiming generalizability.

2. **Expand the 31-case eval to 500+ production cases**. A 31-case eval has ~5pp noise floor. You need 500+ to trust R@1 differences <2pp.

3. **Break down LongMemEval results by qtype and memory-type**. Report Path A vs Path B per qtype. Show the hardset separately.

4. **Audit the dispatcher routing logic**. Document: for each query type, which path should win, and measure correlation between the router's choice and the actual per-path performance.

5. **Before shipping `/recall_hybrid` as a public beta, run it against at least 100 real production queries and publish the results.** Hide the names if needed, but publish the distribution of Path A vs Path B recall gains.

---

## Conclusion

The plan is not wrong; it's incomplete. Both paths solve real problems (atomic keyword lookup vs. long-form semantic retrieval). But the dispatcher is underdeveloped, and the generalizability claim is built on a benchmark that doesn't reflect production constraints.

Treat the 95% target as a **benchmark milestone**, not a product claim. For product, measure on production data first. Once you have 500+ production test cases and pass the five falsifier tests, then the generalizability claim becomes credible.

Until then: publish 96.4% on LongMemEval_S and 75% on the 31-case eval separately, with full honesty about what each benchmark measures.
