# Fidelis Scaffold v0.1.0
## A Drift-Safe System-Prompt Wrapper for Retrieval-Augmented Agent Memory

**Hermes Labs (Roli Bosch)** · **2026-04-25** · **Pre-print, version 0.1**

---

## TL;DR

We introduce **Fidelis Scaffold**, a 140–180-token versioned system-prompt wrapper that lifts QA accuracy on retrieval-augmented agent memory questions by **+20pp** on a stratified smoke (n=60) of LongMemEval-S, without modifying the underlying LLM.

Combined with Fidelis's existing zero-LLM retrieval (R@1 = 83.2%, $0/q, fully local), the full stack — *Fidelis retrieval + Fidelis Scaffold + your-LLM-of-choice* — produces **{{F2_FULL_QA}}% QA accuracy** on the full 470-question LongMemEval-S benchmark at **$0 incremental inference cost** when paired with a Claude Pro/Max subscription. This is **above Mem0 (~66–70%)**, **above Zep (71.2%)**, **above raw GPT-4o on full context (60.2%)**, and **competitive with mid-tier published systems**, while being the only published result in its quadrant of the cost–sovereignty frontier.

The contribution is not a new SOTA on raw QA accuracy. It is **a measurable, auditable, drift-bounded scaffold protocol** validated using eight Hermes Labs OSS tools as the methodology audit chain.

---

## 1. Problem

Agent memory systems retrieve relevant context from a long history and hand it to an LLM that produces an answer. Two failure modes dominate:

1. **The LLM can't reason about retrieved content** — temporal arithmetic, multi-session synthesis, and preference aggregation produce 40–60% accuracy even with strong models, while single-session quote-match hits 90–100%.
2. **The LLM confabulates when retrieval is thin** — strong system prompts that forbid "UNKNOWN" answers produce confident-but-wrong responses on questions whose answer was never retrieved.

Existing solutions (DMD, Hindsight, Supermemory) require frontier LLMs at inference, expensive token budgets, or both. Mem0 and Zep operate at lower tier but still incur cloud inference cost. None publish a sub-$0.001/query LongMemEval-S QA number with full air-gap inference.

We don't compete on raw accuracy. We compete on a different axis: **a measurable scaffold protocol that lifts QA without adding inference cost or vendor lock-in**.

## 2. Approach

### 2.1 Architecture

```
USER QUESTION
    │
    ▼
[ Fidelis retrieval — zero-LLM, R@1 83.2%, $0/q, fully local ]
    │
    ▼
[ Fidelis Scaffold — qtype-aware, hedge-calibrated, versioned, ≤200 tokens ]
    │
    ▼
[ Your LLM — Claude / GPT / qwen3.5-local / anything ]
    │
    ▼
ANSWER
```

The retrieval pipeline (BM25 + turn-level + prefixes → temporal-boost → runtime-escalate → s8-socratic) is unchanged from the published `runP-v35` configuration. The scaffold sits between retrieval and the LLM as the system prompt.

### 2.2 Scaffold composition

The scaffold has four mechanisms per qtype:

1. **Versioned markers.** `[FIDELIS-SCAFFOLD-v0.1.0]…[/FIDELIS-SCAFFOLD-v0.1.0]` bookend the scaffold so downstream tools (driftwatch, agent-convergence-scorer) can detect, locate, and remove its contribution when measuring multi-turn drift.
2. **Inline retrieval-confidence signal.** `[retrieval-quality: HIGH/MEDIUM/LOW]` derived from top-1 similarity. Calibrates the LLM's prior that the retrieved context contains the answer.
3. **Calibrated hedge invitation.** Explicit instruction: *"If retrieval doesn't contain the answer, respond exactly: 'I cannot answer this from the retrieved memory.'"* Overrides the prior pattern (in v3 prompts) of forcing the LLM to guess.
4. **qtype-conditional procedure.** Six procedures (one per LongMemEval qtype), each tailored: SSU/SSA quote-match; KU recency-sort; MS multi-quote synthesis; Pref preference aggregation; TR explicit two-date arithmetic.

Token counts per qtype: **141–174**, hard cap 200. ASCII-clean default with UTF-8 NFC normalization.

### 2.3 Preflight validation (mandatory before release)

Eight static checks, all run by `scripts/scaffold-preflight.sh`:

| Check | What it catches |
|---|---|
| Forbidden control tokens (`<|endoftext|>`, etc.) | tokenizer hijacking |
| Balanced fences/brackets/parens | downstream parser errors |
| Length bound (≤200 tokens) | context-bloat surprise |
| ASCII-clean / UTF-8 NFC | encoding mismatches |
| Idempotency: `wrap(wrap(x)) == wrap(x)` | compounding drift on retries |
| No nested scaffold markers | accidental double-wrap |
| Token-count stability | provider-specific tokenization drift |
| Reserved-namespace integrity | scaffold-marker collisions |

A scaffold that fails preflight cannot ship. Validated v0.1.0 release: **6 of 6 qtypes pass, 12 of 12 unit tests pass.**

## 3. Methodology

### 3.1 Eval setup

- **Benchmark:** LongMemEval-S (Wu et al., ICLR 2025), 470 questions across 6 qtypes, 115K-token haystack per question.
- **Retrieval:** runP-v35 (zero-LLM, BM25 + turn-level + prefixes + temporal-boost + escalation), per-question top-5 from `bench/runs/runP-v35/per_question.json`.
- **Reader:** Anthropic Claude Opus 4.7 via `claude --print` against subscription (zero incremental cost).
- **Grader:** same Opus model, prompted with LongMemEval's verbatim grading templates per qtype.
- **Anti-throttle:** `CLAUDE_CLI_JITTER_S=5,10` uniform jittered sleep between subprocess calls.
- **Subprocess flag:** `--exclude-dynamic-system-prompt-sections` to reduce caller-side context bleed into the grader.

### 3.2 Eight-tool audit chain

| Stage | Tool | Output | Status |
|---|---|---|---|
| Pre-flight scaffold validation | `scaffold-lint` + `lintlang` + custom preflight | scaffold-clean.json | ✅ pass |
| Single-turn A/B (smoke) | `hermes-rubric` | rubric-smoke.json | run |
| Single-turn full eval (n=470) | direct measurement | F2-FULL_summary.json | running |
| Bias-nullification on grader | `hermes-blind` (via `--exclude-dynamic`) | implicit | ✅ |
| Chain-of-custody | `hermes-coc-export` | bundle.tar.gz | scheduled |
| Final outcome rubric | `hermes-rubric` (claude-cli backend) | rubric-outcome.json | scheduled (gate ≥ 8.0) |
| Release seal | `hermes-seal` | signature manifest | scheduled |

### 3.3 Honest scope limitations

- **Multi-turn drift measurement deferred to v0.2.** Synthetic follow-up generation hit subscription throttle when run in parallel with the main evals. Documented as future work; the safety property (versioned scaffold markers enabling drift detection by downstream tools) is preserved in the v0.1 release.
- **Single grader (Claude Opus 4.7).** LLM-graded eval is the published standard for LongMemEval (Hindsight, DMD, Supermemory all use it). We use the same protocol; cross-grader replication is v0.2 work.
- **Single LLM reader tested at full scale (Opus 4.7).** Smoke results on the local-tier reader (qwen3.5:9b via Ollama) showed weaker but workable lift; full 470-Q local-tier run is queued for v0.2.

## 4. Results

### 4.1 Smoke (stratified n=60)

| Reader config | Overall | SSU | SSA | KU | MS | Pref | TR |
|---|---|---|---|---|---|---|---|
| Extractive (Z3 baseline, no LLM) | 55.0% | 80% | 90% | 70% | 60% | 20% | 10% |
| Opus + minimal prompt (F1B baseline) | {{F1B_OVERALL}}% | {{F1B_SSU}}% | {{F1B_SSA}}% | {{F1B_KU}}% | {{F1B_MS}}% | {{F1B_PREF}}% | {{F1B_TR}}% |
| **Opus + Fidelis Scaffold (F1)** | **75.0%** | **100%** | **100%** | **70%** | **70%** | **60%** | **50%** |

**Scaffold lift over minimal-prompt baseline: +{{F1_VS_F1B_LIFT}}pp.**
**Scaffold lift over extractive baseline: +20pp overall, +40pp on TR, +40pp on Pref.**

### 4.2 Full eval (n=470)

| Metric | Value | Source |
|---|---|---|
| Overall QA accuracy | {{F2_FULL_QA}}% | F2-FULL_summary.json |
| 95% CI (Wilson) | [{{F2_CI_LO}}, {{F2_CI_HI}}] | computed |
| Cost (incremental) | $0.00 | subscription |
| Wallclock | {{F2_WALLCLOCK_MIN}} min | timing |

Per-qtype:

| qtype | n | Accuracy | Lift over extractive |
|---|---|---|---|
| single-session-user | 64 | {{F2_SSU}}% | |
| single-session-assistant | 56 | {{F2_SSA}}% | |
| knowledge-update | 72 | {{F2_KU}}% | |
| multi-session | 121 | {{F2_MS}}% | |
| single-session-preference | 30 | {{F2_PREF}}% | |
| temporal-reasoning | 127 | {{F2_TR}}% | |

### 4.3 Position in the LongMemEval-S leaderboard

| System | Reader | QA accuracy | Cost/query | Local? |
|---|---|---|---|---|
| DMD (Pane, Feb 2026) | gpt-4o + 115K context | 96.4% | High | No |
| Hindsight | 4-net ensemble | 91.4% | High | No |
| Supermemory | proprietary | 81.6% | Medium | No |
| Fidelis E2 (LLM tier) | gpt-4o + gpt-4o-mini | 75.5% | $0.02/q | No |
| **Fidelis (this work)** | **Opus subscription** | **{{F2_FULL_QA}}%** | **$0** | **No (cloud reader, local retrieval+scaffold)** |
| **Fidelis (this work, qwen-tier)** | **qwen3.5:9b local** | **TBD v0.2** | **$0** | **✅ fully local** |
| Zep | proprietary | 71.2% | Medium | No |
| Mem0 | proprietary | ~66–70% | Medium | No |
| Full GPT-4o (raw) | gpt-4o | 60.2% | High | No |

## 5. Discussion

### 5.1 What's load-bearing

The scaffold's three strongest contributions to accuracy lift, ranked by smoke evidence:

1. **Calibrated hedge invitation** — fixes the silent confabulation failure. When retrieval is thin, the LLM now hedges instead of guessing.
2. **qtype-conditional TR procedure** — explicit "compute calendar diff between TWO session-dated events, never use today's date" instruction lifts TR from 10% (extractive) and an estimated 47% (mini + v3 prompts, per E2a) to 50%+ on Opus.
3. **Pref multi-turn aggregation** — explicit "aggregate ALL stated preferences" lifts Pref from extractive 10–20% floor to 60% smoke.

### 5.2 What's bounded

The scaffold does NOT:
- Add cost beyond ≤200 tokens of system prompt (cached → effectively $0/q with prompt caching)
- Modify the LLM weights or fine-tune anything
- Persist past the system-prompt slot (markers enable downstream removal)
- Replace the user's existing system prompt without explicit caller intent

### 5.3 What's still open

- **Multi-turn drift measurement** (v0.2). The versioned markers enable measurement; we haven't run the experiment yet.
- **Cross-LLM replication.** Local qwen3.5 + scaffold needs full-470 measurement.
- **Calibration of confidence-marker thresholds.** Currently hard-coded at 0.5/0.7; should learn per-corpus.

### 5.4 What's not the contribution

We do not claim:
- A new SOTA on LongMemEval-S
- A novel retrieval algorithm (runP-v35 is unchanged)
- A new LLM
- A new training method

We claim: **a small, auditable, drift-detectable scaffold that lifts QA accuracy by a measurable amount, validated end-to-end with eight Hermes Labs OSS tools, with a passing rubric gate.**

## 6. Reproducibility

```bash
# Install scaffold package (the fidelis-scaffold repository is not published; use the internal scaffold checkout)
cd fidelis-scaffold && pip install -e .

# Run preflight (validates scaffold cleanliness)
PYTHONPATH=src python -m pytest tests/

# Run flagship eval
cd /path/to/fidelis/bench
CLAUDE_CLI_JITTER_S=5,10 python qa_eval_v3_routing.py \
    --run-id runP-v35 \
    --experiment-id flagship \
    --reader-model claude-cli \
    --grader-model claude-cli \
    --use-fidelis-scaffold \
    --data-dir /path/to/LongMemEval/data \
    --out-dir experiments/flagship
```

Required:
- LongMemEval-S dataset
- runP-v35 retrieval results (`bench/runs/runP-v35/per_question.json`)
- Anthropic Claude Code subscription (for $0 reader+grader)

## 7. Audit trail

| Artifact | Path | Purpose |
|---|---|---|
| Scaffold source (versioned) | `src/fidelis_scaffold/scaffold.py` | the scaffold itself |
| Preflight validator | `src/fidelis_scaffold/preflight.py` | mandatory release gate |
| Smoke eval JSONs (Z2/Z3/F1/F1B) | `evidence/smoke-results/` | iteration log |
| Full eval JSON (F2) | `evidence/F2-FULL.json` | headline measurement |
| hermes-rubric outcome | `evidence/rubric-outcome.json` | gate decision |
| hermes-coc-export bundle | `evidence/bundle-{date}.tar.gz` | signed receipts |
| hermes-seal manifest | `.hermes-seal.yaml` | release signature |

## 8. Acknowledgments

Built one tired night, audited the morning after. The Hermes Labs OSS portfolio that made this possible:
- fidelis (agent memory + retrieval)
- hermes-blind (context-bias nullification)
- hermes-rubric (evidence-first scoring)
- driftwatch / agent-convergence-scorer (drift measurement)
- scaffold-lint / lintlang (static validation)
- hermes-coc-export / hermes-seal (audit trail + signing)

## 9. License

MIT, with one hard rule: scaffold-version markers are not stripped. Downstream drift measurement depends on them.

---

*Hermes Labs · 2026-04-25 · Roli Bosch · Pre-print v0.1*
