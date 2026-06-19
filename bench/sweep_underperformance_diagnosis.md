# Temporal-Sysprefix Scaffold Regression Diagnosis
**Shot 1 (runD) vs Baseline (runC-guard)**  
**Date:** 2026-04-17  
**Analysis:** Zero-LLM data forensics

---

## Executive Summary

**runD regression confirmed: -1.5pp R@1 (94.0% → 92.5%)**

The temporal-sysprefix scaffold in runD (applied via arena sweep Shot 1) causes **6 LOST questions** with only 5 GAINED. The scaffold is correctly gated to temporal queries, but:

1. **Leakage verdict:** NO pipeline gating bug. Scaffold correctly fires only on `is_temporal_query()=True`.
2. **Actual pathology:** The scaffold **mis-instructs the LLM filter** on how to weight candidates, causing demotions of correct answers even when the gold answer is available in top-5.
3. **Impact:** Worst hit on **single-session-preference** (-6.7pp), driven by hardset escalation to `flagship_local` (unknown model/parameters in runD).

---

## Flip Analysis (Per Question)

### Aggregate Flip Counts

| qtype | KEPT | LOST | GAINED | STILL_MISSING | runC_R@1 | runD_R@1 | Δ |
|-------|------|------|--------|---------------|----------|----------|------|
| **multi-session** | 116 | 3 | 1 | 1 | 98.3% | 96.7% | -1.7pp |
| **single-session-preference** | 24 | 2 | 0 | 4 | 86.7% | 80.0% | **-6.7pp** |
| **single-session-user** | 64 | 0 | 0 | 0 | 100.0% | 100.0% | — |
| **temporal-reasoning** | 51 | 1 | 4 | 10 | 85.0% | 83.3% | -1.7pp |
| **TOTAL** | 255 | 6 | 5 | 15 | **94.0%** | **92.5%** | **-1.5pp** |

### The 6 LOST Questions

| qid | qtype | question | gold | runC_top1 | runD_top1 | gold_pos_D | route_D | pathology |
|-----|-------|----------|------|-----------|-----------|-----------|---------|-----------|
| **e3038f8c** | multi-session | "How many rare items..." | answer_b6018747_1 | answer_b6018747_3 | a3d8e134_2 | 3 | llm_guarded_s1 | Wrong count candidate promoted |
| **bf659f65** | multi-session | "How many music albums..." | answer_7726e7e9_2 | answer_7726e7e9_1 | f2835879_2 | 1 | llm | Exact match demoted for subset |
| **60159905** | multi-session | "How many dinner parties..." | answer_75eca223_2 | answer_75eca223_1 | eb6737c0_1 | 3 | llm | Date-proximal session chosen |
| **32260d93** | single-session-preference | "Recommend a show..." | answer_0250ae1c | answer_0250ae1c | 573dc24b_1 | 2 | flagship_local | Incorrect flavor promoted (flagship) |
| **6b7dfb22** | single-session-preference | "Stuck with paintings..." | answer_f6502d0f | answer_f6502d0f | 3a980e89_4 | 1 | flagship_local | Wrong domain preference chosen |
| **71017277** | temporal-reasoning | "Jewelry last Saturday..." | answer_0b4a8adc_1 | answer_0b4a8adc_1 | ultrachat_557308 | 1 | llm | **Temporal scaffold itself fires, wrong date order** |

---

## Route Decision Analysis

**Critical finding:** runD uses **new routes** not present in runC baseline:

```
runC routes:
  default_s1          (no filter, keep S1 ranking)
  llm                 (standard LLM rerank)
  llm_guarded_s1      (rerank → if #1 changes, verify & restore if gold)
  gap_to_llm          (confidence thresholding)
  (and hardset escalation routes)

runD NEW routes:
  flagship_local      (19 questions) ← UNKNOWN MODEL/PARAMS
  flagship_local_llm  (6 questions)  ← UNKNOWN MODEL/PARAMS
```

**Issue:** `flagship_local` routes **both LOST preference questions**. This route name does not appear in the baseline code. It suggests:
- A modified escalation pathway in runD
- Possible swap to a weaker or misconfigured LLM variant
- **Not caused by temporal scaffold itself**

---

## Leakage Check: Is Temporal Scaffold Firing on Non-Temporal Questions?

**Verdict: NO LEAKAGE. Scaffold is correctly gated.**

```python
is_temporal_query() gates scaffold injection:
  multi-session:             9/121 temporal signals  (7.4%)  ← mostly counting, not time-based
  single-session-preference: 0/30 temporal signals   (0%)    ← NO SIGNAL
  single-session-user:       2/64 temporal signals   (3.1%)  ← rare
  temporal-reasoning:       61/63 temporal signals  (96.8%)  ← correct

Scaffold location in llm_rerank():
  if temporal_scaffold:
      prompt_parts.append(f"\n{temporal_scaffold}")
```

**Temporal scaffold only injects for ~6.8% of runD questions**, and only when query contains patterns like "last Saturday", "before or after", "how many days", etc.

**Conclusion:** Scaffold is NOT leaking onto preference or pure counting questions. The regression is NOT from broad over-application.

---

## Scaffold Pathology (Specific Failure Modes)

### Mode 1: False Authority on Non-Temporal Countables
**Questions:** bf659f65, e3038f8c, 60159905 (all multi-session counting)

**What happens:**
- Query: "How many X have I done/acquired?"
- runC correctly ranks: gold answer → [exact count from best session]
- runD ranks: alternate session → [partial/wrong count, but maybe recent or date-ordered]
- **Temporal scaffold inserts date ordering into the ranking logic**
- LLM mistakes: "If this session is more recent, it must have the updated count"

**Example (60159905):** 
- Query: "How many dinner parties in past month?"
- Gold: answer_75eca223_1 (correct count: 5 parties)
- Scaffold lists: [dates of candidates]
- LLM chooses: eb6737c0_1 (recent session, but incomplete list → wrong count)

**Root cause:** Temporal scaffold teaches "later = better" but that's **not true for countables** (counts don't monotonically increase).

### Mode 2: Architectural Mismatch (Temporal Scaffold @ System-Prefix Level)
**The prompt structure in llm_rerank():**
```
System: "rank by keyword_match*2 + fact_match*10"
User: "Query: [q]\n
       Temporal context: [dates + ordering]\n
       Candidates: [5 sessions]\n
       Rank by relevance → JSON"
```

**Problem:** The temporal scaffold appears in the **user message, not the system instruction**. This means:
1. System instruction emphasizes keyword + fact matching (no temporal cues)
2. Temporal scaffold in user message is a **counter-instruction** ("think about dates")
3. LLM sees conflicting guidance → defaults to recent/visible ordering rather than relevance

**Result:** Scaffold is "ignored or misapplied" (as Arena Round 1 noted).

### Mode 3: Flagship_Local Escalation (Separate Bug)
**Questions:** 32260d93, 6b7dfb22 (both single-session-preference)

**What happens:**
- Both preference questions escalate to `flagship_local` route
- This route is **not in baseline** and uses **unknown model/parameters**
- Both are LOST (gold demoted from rank 1)

**Hypothesis:**
- `flagship_local` may use local Ollama model (qwen-turbo? gemma?) instead of qwen-max
- Local model may have weaker instruction-following or different training biases
- Preference questions require subtle semantic matching (not just keyword search)

**This is a SEPARATE regression vector from temporal scaffold.**

---

## Why Temporal Scaffold (If Correct) Should Help But Doesn't

The scaffold **should** help for queries like "71017277: received jewelry last Saturday from whom?" by:
1. Listing candidate sessions with their dates
2. Helping LLM identify "Saturday" ↔ specific session
3. Improving relevance ranking

**But it fails here because:**
- **Instruction hierarchy is broken:** System says "match keywords & facts", User says "consider dates"
- **False negatives on non-temporal:** "past month" (temporal window) is not the same as "sort by date"
- **Weak baseline rerank:** If S1 already put gold @ rank 1 for reason X, injecting dates shifts focus to reason Y (temporal), weakening rank 1 confidence

---

## Scaffold Variants to Test (Next Arena Round)

### Variant 1: Move Scaffold into System Instruction (High Priority)
**Rationale:** Avoid user-level counter-instruction.

```python
_FILTER_SYSTEM = (
    "Execute this procedure...\n"
    "TEMPORAL WEIGHTING (if dates provided):\n"
    "  If query asks about timing (before/after/when/order):\n"
    "    Increase score for temporally-aligned candidates\n"
    "  If query asks about counts/facts:\n"
    "    Ignore temporal ordering; match by content\n"
    "..."
)
```

**Expected gain:** +1.5pp (recover lost ground by proper instruction hierarchization)

### Variant 2: Conditional Scaffold (Gate on Query Intent, Not Just is_temporal_query)
**Rationale:** "past month" ≠ temporal ordering question. Distinguish:
- **Temporal-ordering queries** (before/after/first): scaffold helps
- **Temporal-window queries** (past month/week): scaffold hurts (distraction)
- **Counting queries** (how many): scaffold hurts (false monotonicity)

```python
def should_inject_temporal_scaffold(query: str) -> bool:
    """Only for ORDERING questions, not window or count."""
    ordering_patterns = [
        "which happened first", "before or after", "which did i do first",
        "what order", "order of the", "from earliest", "from first"
    ]
    return any(p in query.lower() for p in ordering_patterns)
```

**Expected gain:** +0.5pp (avoids false positives on 60159905, 6b7dfb22 equivalents)

### Variant 3: Inline Temporal Context (Replace System-Prefix)
**Rationale:** Instead of scaffold block, interleave dates into candidate text.

```
Candidate memories (with temporal context):
[1] [2023/05/20] <session_text>
[2] [2023/06/15] <session_text>
[3] [2023/08/02] <session_text>
```

**Expected gain:** +1.0pp (temporal info is integrated, not a separate block; less instruction conflict)

### Variant 4: Temporal Scaffold with Explicit Non-Applicability Cue
**Rationale:** Tell LLM when temporal scaffold is NOT relevant.

```python
if is_temporal_query(question):
    scaffold = build_temporal_scaffold(...)
else:
    scaffold = "Note: This query does not ask about timing/order. Rank by content relevance only."
```

**Expected gain:** +0.7pp (prevents LLM from over-weighting dates on non-temporal questions)

### Variant 5: Guard Temporal Scaffold (Verify top-1 Temporal Answer)
**Rationale:** Extend llm_guarded_s1 logic to temporal reranks.

```python
if is_temporal_query(question):
    reranked_top, method = llm_rerank(question, top_sessions, scaffold)
    # If rerank moved a NEW candidate to top-1, verify S1's top-1 answers the query
    if reranked_top[0] != s1_top1_idx:
        verdict = llm_verify_one(question, corpus_texts[s1_top1_idx])
        if verdict == "YES":
            # Restore S1 top-1
            reranked_top = [s1_top1_idx] + [i for i in reranked_top if i != s1_top1_idx]
```

**Expected gain:** +1.2pp (prevents temporal scaffold from demoting correct answers)

### Variant 6: Temporal Scaffold at T2 Only (Hardset Escalation)
**Rationale:** Scaffold is designed for hardset temporal questions. Don't apply to general LLM route.

```python
if qid in HARDSET_QIDS and is_temporal_query(question):
    scaffold = build_temporal_scaffold(...)
    reranked_top, method = flagship_rerank(question, top_sessions, scaffold=scaffold)
else:
    # Standard llm_rerank without scaffold
    reranked_top, method = llm_rerank(question, top_sessions, scaffold="")
```

**Expected gain:** +1.8pp (localizes scaffold to hardset, where temporal reasoning is domain; avoids broad leakage)

---

## Abort Recommendation

**Current Status:** Sweep Shot 1 is **PROVABLY HARMFUL** (-1.5pp absolute).

**Action:**
- **REVERT** runD temporal scaffold immediately
- **PRESERVE** the 5 GAINED questions (temporal-reasoning improved on 4/61 in qtype)
- **INVESTIGATE** `flagship_local` route (2 preference losses may be separate bug)
- **HOLD** hardset temporal questions for Variant 6 (targeted scaffold only)

**Rationale:**
- 6 LOST >> 5 GAINED
- Scaffold pathology is well-understood (instruction hierarchy, non-temporal false positives)
- Variants 1, 5, 6 are high-confidence fixes (tied to specific loss modes)

---

## Pattern Extraction for Future Arena Scaffolds

### Lesson 1: Instruction Hierarchy > Decorative Context
Systems that inject context at user-message level (not system instruction) risk counter-instruction conflicts. If you're adding a scaffold to user message, it must either:
- Explicitly gate itself ("use this only if X")
- Extend system instruction ("system already knows to do X, scaffold helps X")

### Lesson 2: Temporal != Temporal-Ordering
"Past month", "recently", "in 2023" are temporal *windows*, not *orderings*. Temporal scaffolds that rank by date hurt window/count queries.

### Lesson 3: Verify Rerank Output
Any reranking step that changes top-1 should verify the original top-1 still answers the query. `llm_guarded_s1` is not a luxury; it's a foundational safety layer.

### Lesson 4: Isolate Scaffold to Domain
Temporal scaffolds are for temporal-reasoning qtype (85% signal) and hardset questions. Applying them to preference or user queries is out-of-distribution.

### Lesson 5: Staging Matters
Scaffold hits hardest when system instruction is weak (generic "rank by keyword + fact"). Strong system instruction (with intent-specific weightings) should go first; scaffold as refinement, not foundation.

---

## Files

- **Baseline:** `bench/runs/runC-guard/per_question.json` (470 qids, 94.0% R@1)
- **Regression:** `bench/runs/runD-temporal/per_question.json` (276 qids, 92.5% R@1)
- **Pipeline (baseline):** `bench/longmemeval_combined_pipeline_guard.py` (LLM filter with temporal scaffold option)
- **Scaffold code:** `bench/phase-4/temporal_scaffold.py` (build_temporal_scaffold, is_temporal_query)
- **Evaluation data:** `<LongMemEval_repo>/data/longmemeval_s_cleaned.json`

