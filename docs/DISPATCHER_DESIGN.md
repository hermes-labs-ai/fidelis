# Workload-Aware Dispatcher — Design Spec

**Status:** Draft v1 — 2026-04-16  
**Author:** Architect review, Hermes Labs  
**Sources read:**
- `src/cogito/recall.py` — Path A: multi-query RRF + integer-pointer LLM filter
- `src/cogito/recall_hybrid.py` — Path B: BM25+dense+RRF + tiered LLM escalation
- `bench/eval.py` — 31-case eval harness (modes A–E)
- `README.md` — public API surface + benchmark claims
- `CHANGELOG.md` — v0.3.0 architecture history

---

## 1. Memory Classification at Insert Time

### Problem
Path A is tuned for atomic facts (~50–200 chars, single-claim sentences).  
Path B was benchmarked on multi-turn sessions (2,000+ chars, turn-structured).  
Mixing them in one index degrades both.

### Heuristic classifier (no LLM, zero latency)

Applied at `/store` and `/add` time. Result stored as metadata field `mem_type`.

**Signals and thresholds:**

| Signal | Atomic | Session |
|---|---|---|
| `len(text)` | < 400 chars | >= 400 chars |
| Newline count | 0–1 | >= 3 |
| Turn markers present | False | True |
| Sentence count | 1–2 | >= 4 |

**Turn markers** (regex): `r"(User:|Assistant:|Human:|>\s|\[\d{2}:\d{2}\]|Turn \d+)"` — matches common dialog formats.

**Sentence count**: split on `[.!?]\s+[A-Z]`, count separators + 1.

**Decision rule** (all signals weighted, majority wins):

```python
def classify_memory(text: str) -> str:  # "atomic" | "session"
    score = 0
    if len(text) >= 400: score += 2
    if text.count('\n') >= 3: score += 1
    if re.search(TURN_MARKER_RE, text): score += 2
    if _sentence_count(text) >= 4: score += 1
    return "session" if score >= 3 else "atomic"
```

**Edge cases:**
- Code blocks with newlines but no dialog → `len(text)` is the tiebreaker. A 600-char code snippet scores `session` but retrieves fine under Path B (BM25 handles token-exact code terms better). Acceptable.
- Structured logs → same as code. Acceptable.
- Multi-sentence atomic facts (e.g., a decision record with rationale, 300 chars, 3 sentences) → stays `atomic` because len < 400 and no turn markers. Correct.
- Empty text or < 10 chars → `atomic` by default; invalid inserts are mem0's problem.

**No measurement exists** for the production insert-size distribution. Adjust the 400-char threshold after observing the bimodal gap (see Open Questions).

---

## 2. Query Dispatching at Recall Time

### Option analysis

**Option A — Per-type indexes:** Atomic in one ChromaDB collection, sessions in another. Route by mem_type. Clean separation, no cross-contamination.

**Option B — Blended:** Always query both indexes, merge with a reranker. Best coverage, but doubles latency and requires a unified score scale.

**Option C — Query classifier → pick one path:** Single index, query intent selects the retrieval function. No per-memory routing. Misclassified memories surface under the wrong path.

**Decision: Option A — per-type indexes.**

Rationale: the performance gap (21pp on Path A's own eval) comes from path design, not the query. Atomic memories under BM25+dense fusion have worse R@1 because the BM25 component rewards term frequency, which is uninformative for short 1-sentence facts. Session memories under Path A's LLM filter see truncated 150-char snippets (hardcoded in `recall.py:88`) — insufficient context for multi-turn content. Separating the stores eliminates the mismatch without requiring a query classifier.

### Implementation

**Two ChromaDB collections per user:**
- `{collection}_atomic` (e.g., `cogito_memory_atomic`)
- `{collection}_session` (e.g., `cogito_memory_session`)

**At query time:**

```
/recall (or /recall_hybrid)
  │
  ├─ query_type = classify_query(query)  [existing, from recall_hybrid.py]
  │
  ├─ if user passes mem_types=["atomic"]  → run Path A on atomic index only
  ├─ if user passes mem_types=["session"] → run Path B on session index only
  └─ if mem_types=["atomic","session"] (default)
       → parallel fetch from both paths
       → merge by score (cosine-normalized), deduplicate
       → return top N
```

**Merge strategy for dual-index default:**
- Path A scores: cosine similarity (0–1 scale already)
- Path B scores: cosine-blended RRF (0–1 scale already, `recall_hybrid.py:302`)
- Simple merge: interleave by score, deduplicate by text hash. No additional LLM call.

---

## 3. Config + API Surface

### New config keys

```json
{
  "dispatch_mode": "auto",
  "atomic_collection": "cogito_memory_atomic",
  "session_collection": "cogito_memory_session"
}
```

| Env var | Config key | Default | Description |
|---|---|---|---|
| `COGITO_DISPATCH_MODE` | `dispatch_mode` | `"auto"` | `"auto"` \| `"atomic"` \| `"session"` \| `"legacy"` |
| `COGITO_ATOMIC_COLLECTION` | `atomic_collection` | `"{collection}_atomic"` | Atomic facts index name |
| `COGITO_SESSION_COLLECTION` | `session_collection` | `"{collection}_session"` | Session memories index name |

### Endpoint changes

**`POST /store` and `POST /add`** — unchanged signature, new behavior:
- Classify incoming text → write to the appropriate collection
- Response adds `"mem_type": "atomic" | "session"` field (informational)

**`POST /recall`** — unchanged signature, unchanged default behavior when `dispatch_mode="legacy"`:
- `dispatch_mode="auto"` (new default for fresh installs): routes by mem_type index
- `dispatch_mode="legacy"` (set automatically on upgrade for existing users): single-collection behavior, identical to v0.3.0

**`POST /recall_hybrid`** — gains optional `mem_types` field:
```json
{"text": "...", "tier": "filter", "mem_types": ["atomic", "session"]}
```
Default: both. Existing callers without `mem_types` get both, same as before.

### Breaking-change protection

**Upgrade path:** On server start, if the legacy collection exists and the atomic collection does not, set `dispatch_mode="legacy"` automatically and log a warning:
```
[cogito] Legacy single-collection store detected. Running in compatibility mode.
Run `cogito migrate-dispatch` to split into typed indexes.
```

`cogito migrate-dispatch` — new CLI command:
1. Reads all memories from the legacy collection
2. Classifies each, writes to the appropriate typed collection
3. Verifies count parity before deleting legacy records
4. Updates `.cogito.json` → sets `dispatch_mode="auto"`

Existing users are never silently migrated. Manual opt-in only.

---

## 4. Generalizability Guardrails

### Cross-benchmark suite

| Benchmark | Workload type | What it tests | Status |
|---|---|---|---|
| **cogito 31-case eval** (bench/eval.py) | Atomic paraphrase recall | direct_recall, semantic_gap, adversarial, cross-reference | Exists |
| **LongMemEval_S** | Session multi-turn retrieval | Temporal ordering, multi-hop, counting | Exists (bench/longmemeval_combined_pipeline_flagship.py) |
| **LOCOMO** (add this) | Conversational memory + question grounding | Multi-session with structured events | Not yet — download from https://github.com/snap-research/LOCOMO |
| **MemGPT MemoryBench** (add this) | Agent task memory (tool-use sessions) | Procedural + declarative in one corpus | Not yet — https://github.com/cpacker/MemGPT |

**Why LOCOMO:** Structured multi-session with explicit temporal scaffolds — validates Path B on a corpus not used to tune it. First out-of-distribution test.

**Why MemoryBench:** Tool-use agent memories are shorter than LongMemEval sessions but longer than cogito's atomic facts. Tests the classification boundary directly.

### Regression gates (CI)

Add to `bench/ci_gates.yaml`:

```yaml
gates:
  - benchmark: cogito_31case
    metric: recall_at_1
    path: A  # /recall
    floor: 0.72          # current 75%, allow 3pp measurement noise
    block_pr_if_below: true

  - benchmark: cogito_31case
    metric: recall_at_1
    path: E  # /recall_hybrid
    floor: 0.50          # current 54%; any improvement must not regress
    block_pr_if_below: true

  - benchmark: longmemeval_s
    metric: recall_at_1
    path: hybrid_flagship
    floor: 0.944         # current 96.4%, allow 2pp measurement noise
    block_pr_if_below: true

  - benchmark: cogito_31case
    metric: hit_at_any
    path: E
    floor: 0.93          # current 96.4%; must not regress even if R@1 improves
    block_pr_if_below: true
```

**Gate logic:** A PR fails CI if any gate's floor is breached. Floors are set at `(current - measurement_noise)`. Noise budget is 3pp for 31-case (small N, stochastic LLM filter) and 2pp for LongMemEval (larger N, more stable).

### Dual reporting requirement

Every `cogito bench` run (new CLI, see roadmap) must emit:

```
WORKLOAD COVERAGE MATRIX
                          /recall (Path A)   /recall_hybrid (Path B)
cogito 31-case R@1            75%                  54%
cogito 31-case hit@any        96%                  96%
LongMemEval_S R@1             [not measured]       96.4% (runP-v35)
LOCOMO R@1                    [pending]            [pending]
MemoryBench R@1               [pending]            [pending]
```

PRs that only update one cell must still report all measured cells. If a benchmark is not run (e.g., LOCOMO not installed), cells show `[skipped]`, not blank. No commit can merge with a cell that regressed from its previous value without an explicit `# regression-acknowledged: <reason>` comment in the PR.

### Workload coverage matrix (current state)

| Workload | Path A `/recall` | Path B `/recall_hybrid` |
|---|---|---|
| Atomic paraphrase (direct_recall) | **75% R@1** | 54% R@1 |
| Semantic gap (vocab mismatch) | measured in eval | measured in eval |
| Adversarial negative | measured in eval | measured in eval |
| Multi-turn session retrieval | [not measured] | **96.4% R@1** (runP-v35) |
| Temporal ordering queries | [not measured] | tested via regex router |
| Counting/aggregation queries | [not measured] | tested via regex router |

---

## 5. Publishing the Gap

### Why both numbers are competitive advantage

Most retrieval libraries publish one benchmark on one workload and present it as universal accuracy. cogito-ergo already documents the gap honestly in the README (v0.3.0 Regression notice). This is not a liability — it is differentiation. A system that knows *when* it is wrong is more deployable than one that silently overfits.

**Generalizability scorecard (public, in README):**

```
COGITO RETRIEVAL SCORECARD
                        R@1     hit@any   Latency   Workload
/recall                 75%     96%       1197ms    atomic paraphrase
/recall_hybrid          96.4%   —         3500ms    session multi-turn (runP-v35)
/recall_hybrid (filter) 90%+    —         1300ms    session multi-turn
/recall_b               56%     96%       127ms     atomic (zero-LLM)

Benchmark sources: cogito 31-case eval (internal), LongMemEval_S (Xiao et al. 2024)
```

Format rule: every row shows benchmark source. No aggregate. No cherry-picking.

When a user asks "which memory system handles multi-turn sessions?", cogito-ergo answers with a number from a specific benchmark. When they ask "which handles atomic facts cheaply?", it answers with a different number from a different benchmark. That workload-specificity is the pitch.

**Next public claim to support:** After LOCOMO integration, publish Path B R@1 on LOCOMO. If it holds at >=85%, the generalization story is credible without a paper.

---

## Open Questions (measure before building)

1. **Insert distribution:** What fraction of current production memories are session-like (>400 chars)? If <5%, the per-type index split adds complexity with minimal benefit. Measure: `cogito query "." --limit 500 | jq '[.memories[].text | length] | add/length'`.
2. **Cross-type queries:** When a query spans both types (decision = atomic, context = session), the dual-index merge handles it by returning from both. Needs a dedicated test case before shipping.
3. **LongMemEval_S on Path A:** Currently unknown. Measure it to confirm the 21pp regression is directionally real, not a 31-case artifact.
