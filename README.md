# fidelis

> **v0.0.6 (2026-04-25).** Adds the **Fidelis Scaffold** — a 140–180-token
> drift-safe, hedge-calibrated, qtype-aware system-prompt wrapper that lifts
> end-to-end QA accuracy on hard question types when paired with any LLM at
> zero incremental inference cost. Smoke-validated; full 470-Q LongMemEval-S
> evaluation in progress (companion paper landing on full-eval completion).

**Agent memory with zero-LLM retrieval and a $0-incremental-cost QA scaffold.** Fully local retrieval. No API keys required to get a working memory system.

**Headline numbers (Fidelis v0.0.6):**

- **83.2% R@1** on LongMemEval-S retrieval (470 questions) — pure retrieval, no LLM, fully local
- **75% QA accuracy** on LongMemEval-S smoke (n=60 stratified) when the Fidelis scaffold wraps an Opus reader — **above Mem0 (~66–70%), above Zep (71.2%)**
- **$0 incremental cost** per query when paired with a Claude Pro/Max subscription — **10–50× cheaper than every published memory system at our accuracy tier or above**
- **Per-qtype scaffold lifts** over a minimal-prompt LLM baseline: **+20pp on knowledge-update, +16pp on preference, +8pp on multi-session**
- **~90 ms retrieval latency** end-to-end

The architectural fidelity contract holds: when the optional LLM tier is
enabled, the filter outputs only integer pointers — it structurally cannot
corrupt, rephrase, or hallucinate into the content returned to your agent.
See [Hybrid recall](#hybrid-recall) for the tier table and the honest
ceiling on the benchmark-tuned path.

## Fidelis Scaffold (new in v0.0.6)

The QA scaffold technique: a 140–180-token versioned, hedge-calibrated, qtype-aware system prompt that sits between Fidelis retrieval and your existing LLM, lifting end-to-end QA accuracy on hard question types without modifying the LLM and at zero incremental inference cost.

```python
from fidelis.scaffold import wrap_system_prompt, preflight

system = wrap_system_prompt("temporal-reasoning", top_score=0.65)
assert preflight(system).passed  # 8 static safety checks
# Hand `system` as the system prompt to your LLM (Claude / GPT / qwen-local / anything).
```

The scaffold is **versioned** (`[FIDELIS-SCAFFOLD-vX.Y.Z]…[/FIDELIS-SCAFFOLD-vX.Y.Z]`), **bounded** (≤200 tokens, hard cap), and **idempotent** (`wrap(wrap(x)) == wrap(x)`). Versioned markers enable downstream multi-turn drift measurement (driftwatch / agent-convergence-scorer can locate and remove scaffold contributions).

**Smoke evidence (n=60 stratified, Opus reader + Opus grader via Anthropic subscription, $0 incremental cost):**

| qtype | Opus + minimal prompt | Opus + Fidelis Scaffold | scaffold lift |
|---|---|---|---|
| knowledge-update | 50% | 70% | **+20pp** |
| single-session-preference | 44% | 60% | **+16pp** |
| multi-session | 62% | 70% | **+8pp** |
| temporal-reasoning | 50% | 50% | 0pp (Opus does it natively) |
| single-session-user | 100% | 100% | 0pp (ceiling) |
| single-session-assistant | 100% | 100% | 0pp (ceiling) |

**Cost-frontier comparison on LongMemEval-S** (published numbers, 2026-04):

| System | QA accuracy | Cost/query at inference | Local? |
|---|---|---|---|
| DMD | 96.4% | High (115K-token frontier API) | No |
| Hindsight | 91.4% | High (multi-net ensemble) | No |
| Supermemory | 81.6% | Medium (proprietary API) | No |
| Fidelis E2 (paid LLM tier) | 75.5% | $0.02/query | No |
| **Fidelis + scaffold (subscription)** | **~75% (smoke)** | **$0 incremental** | **No (cloud reader, local retrieval+scaffold)** |
| Zep | 71.2% | Medium | No |
| Mem0 | ~66–70% | Medium | No |
| Full GPT-4o (raw context) | 60.2% | High | No |

Full 470-question evaluation in progress; smoke evidence + machine-readable summary in [`experiments/zeroLLM-FLAGSHIP-evidence/`](experiments/zeroLLM-FLAGSHIP-evidence/) (see `SUMMARY.json` for per-arm aggregates and per-qtype scaffold lift). Companion technical report lands when the full eval completes. See [`docs/scaffold.md`](docs/scaffold.md) for the full scaffold contract, preflight, and audit chain.

A separate `/recall` atomic path is purpose-built for short-fact lookup (not session retrieval); it scores 85% R@1 combined with the snapshot layer on a 31-case internal atomic-fact eval. This is a secondary surface; the headline retrieval number above (83.2% R@1 on the full 470-question LongMemEval-S) is the authoritative session-retrieval measurement.

[![PyPI version](https://img.shields.io/pypi/v/fidelis)](https://pypi.org/project/fidelis/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/fidelis/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Made by Hermes Labs](https://img.shields.io/badge/made%20by-Hermes%20Labs-purple)](https://hermes-labs.ai)

---

## How this is different from mem0 / Zep / Letta

Short version: **they call an LLM during retrieval; fidelis doesn't.**

| | mem0 / Zep / Letta (typical) | fidelis (zero-LLM default) |
|---|---|---|
| LLM call on retrieval hot path | yes | **no** |
| Cost per retrieval | ~$0.001–0.02 | **$0** |
| Latency | ~1–3 s | **~90 ms** |
| Works fully offline / air-gapped | no (cloud LLM required for headline numbers) | **yes** (local Ollama) |
| API keys required to run | yes | **no** |
| R@1 on LongMemEval_S | 92–96% (cloud LLM tier) | 83.2% (zero-LLM) |
| LLM can rephrase / hallucinate into returned content | yes | **no** (integer-pointer contract on optional LLM tier) |
| Write survives upstream LLM outage | mixed | **yes** (degrade queue) |

Your agent already calls an LLM. fidelis feeds that LLM the right memory
without adding a second LLM call to retrieve it.

### Where this matters (and where it doesn't)

**Use fidelis when:**
- You can't send memory contents to a third-party LLM (compliance, legal, healthcare, defense, regulated finance).
- You need per-query cost to be zero at scale.
- You need sub-100 ms retrieval.
- You want fidelity by construction — memory text is verbatim, not LLM-paraphrased.

**Use mem0 / Zep / Letta when:**
- You need the highest possible benchmark accuracy and cost isn't a constraint.
- Temporal reasoning and preference-ranking queries dominate your workload (fidelis's weakest categories at 66–67% zero-LLM).
- You're fine with a cloud LLM dependency in the memory layer.

The LLM tier is available (`tier="filter"` / `tier="flagship"`) if you want
to opt in to benchmark-parity mode. It's labeled experimental until the
calibration miss (80% escalation vs 10% intended) is fixed — see
[Known limitations](#when-to-enable-the-optional-llm-tier).

---

## The Problem

Every retrieval system that uses an LLM to select or rank memories has the same failure mode: the LLM rephrases on the way out. You store `"auth tokens expire after 3600 seconds"` and get back `"authentication has a configurable timeout."` The specific fact is gone.

- **Raw vector search** returns candidates by similarity, but precision plateaus at 50–60% R@1 on real workloads
- **LLM-based re-rankers** improve relevance but generate text — they summarize, merge, or hallucinate into the content your agent receives
- **Full RAG pipelines** add latency and cost without solving the fidelity problem

fidelis fixes this structurally. The filter LLM outputs only integer pointers (`[3, 7, 12]`). The server dereferences them to verbatim stored text. The LLM never sees, generates, or touches memory content. Fidelity is architectural, not a prompting convention.

**Headline retrieval (LongMemEval-S, 470 questions, zero-LLM tier, $0/q, fully local):**

| Metric | Value |
|---|---|
| **R@1** | **83.2%** |
| **R@5** | **98.3%** |
| **R@10** | **99.1%** |
| Cost | $0/query |
| Latency | ~90 ms end-to-end |
| Per-qtype R@1 | SSA 100% · KU 95.8% · SSU 95.3% · MS 83.5% · SSP 66.7% · TR 66.1% |

**Secondary `/recall` atomic-fact surface** (purpose-built for short-fact lookup, not session retrieval; 31 internal test cases, qwen3.5:2b filter, fully local):

| Mode | R@1 | hit@any | Latency |
|---|---|---|---|
| Combined (snapshot + recall) | 85% | 96% | 1303 ms |
| recall only | 63% | 81% | 1197 ms |
| recall_b (zero-LLM) | 56% | 96% | 127 ms |

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              fidelis server              │
                    │                 :19420                       │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │             SNAPSHOT LAYER                   │
                    │   Compressed markdown index (~741 tokens)    │
                    │   Built once from corpus via `fidelis snapshot`│
                    │   Returned with /recall — no vector search   │
                    │   Solves cross-reference queries (0%→50% R@1)│
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │         STAGE 1 — recall_b (zero-LLM)       │
                    │   Query decomposition → sub-queries          │
                    │   Stop-word stripping + bigrams + trigrams   │
                    │   Vocab expansion (fidelis calibrate)         │
                    │   Up to 8 sub-queries, merged with RRF       │
                    │   Latency: ~127ms                            │
                    └──────────────────┬──────────────────────────┘
                                       │ up to 100 candidates
                    ┌──────────────────▼──────────────────────────┐
                    │       STAGE 2 — integer-pointer filter       │
                    │   Filter LLM sees: [1] text  [2] text ...   │
                    │   Filter LLM outputs: [3, 7, 12]  ← ONLY    │
                    │   Server fetches candidates[3], [7], [12]   │
                    │   Returns: verbatim stored text              │
                    │   Added latency: ~1176ms                     │
                    └─────────────────────────────────────────────┘
```

The filter LLM never generates memory text. Out-of-range integers are silently ignored. Fidelity is a structural property of the pipeline, not a prompting convention.

---

## Benchmarks

### Primary: LongMemEval-S (470 questions, public benchmark)

Measured 2026-04-24. Zero-LLM retrieval pipeline (BM25 + turn-level + prefixes + temporal-boost + escalation), `runP-v35` configuration, fully local on Ollama. $0/query.

| Metric | Value | Notes |
|---|---|---|
| **R@1** | **83.2%** | top-1 contains gold |
| **R@5** | **98.3%** | gold in top-5 across all qtypes |
| **R@10** | **99.1%** | |
| Cost | $0/query | no LLM in retrieval path |
| Wallclock | 101.3 s for 470 questions | ~215 ms/question |

Per-qtype R@1: single-session-assistant 100%, knowledge-update 95.8%, single-session-user 95.3%, multi-session 83.5%, single-session-preference 66.7%, temporal-reasoning 66.1%.

Raw aggregate: [`bench/runs/zeroLLM-full-20260424/aggregate.json`](bench/runs/zeroLLM-full-20260424/aggregate.json).

### Secondary: internal `/recall` atomic-fact eval

Measured 2026-03-28. 31 internal test cases (short-fact lookup, not session retrieval). qwen3.5:2b as filter model, local Ollama. Different surface from LongMemEval — this measures the atomic-fact path, not session retrieval.

| Mode | R@1 | hit@any | MRR | Latency |
|---|---|---|---|---|
| Combined (snapshot + recall) | **85%** | **96%** | **0.878** | 1303ms |
| recall only | 63% | 81% | — | 1197ms |
| recall_b (zero-LLM) | 56% | 96% | — | 127ms |
| snapshot only | 41% | — | — | — |

Key results:
- Snapshot layer contributes **+15% hit@any** vs recall-only
- Cross-reference queries: recall alone gets **0% R@1**; combined gets **50%**
- recall_b matches combined hit@any (96%) at 10x lower latency — use it when cost matters

---

## Quick Start

**1. Install**

```bash
pip install fidelis
```

**2. Pull Ollama models**

```bash
ollama pull mistral:7b
ollama pull nomic-embed-text
```

Requires a running [Ollama](https://ollama.ai) instance.

**3. Configure filter LLM**

```bash
# Option A: direct Anthropic key
export ANTHROPIC_API_KEY=sk-ant-...

# Option B: any OpenAI-compatible gateway (local LM Studio, OpenClaw, etc.)
export COGITO_FILTER_ENDPOINT=http://your-gateway/
export COGITO_FILTER_TOKEN=your-token
export COGITO_FILTER_MODEL=anthropic/claude-haiku-4-5
```

Or via `.cogito.json` in your working directory:

```json
{
  "filter_endpoint": "http://your-gateway/",
  "filter_token": "your-token",
  "filter_model": "anthropic/claude-haiku-4-5"
}
```

**4. Start the server**

```bash
fidelis-server
# or: fidelis server
```

**5. First recall**

```bash
fidelis recall "what did we decide about the auth architecture"
```

```bash
curl -X POST http://127.0.0.1:19420/recall \
  -H "Content-Type: application/json" \
  -d '{"text": "auth architecture decisions"}'
```

---

## Hybrid recall

`recall_hybrid` is the session-retrieval path. Three tiers, one contract:

| Tier | Default? | R@1 on LongMemEval_S | Cost/query | Latency |
|---|---|---|---|---|
| `zero_llm` | **yes (v0.0.8 default)** | **83.2%** | **$0** | ~90 ms |
| `filter` | no | ~92% (runO-v34) | ~$0.002–0.003 | ~1.3 s |
| `flagship` | no | 96.4% (runP-v35, 2026-04-18) | higher, see below | ~3.5 s |

The **zero-LLM tier is the production recommendation.** It's the one
shipped by default and the one we stand behind as battle-ready.

The **filter and flagship tiers are benchmark-tuned and experimental.**
They ported the architecture that reached 96.4% on LongMemEval_S, but that
run escalates to flagship on ~80% of queries vs the 10% the threshold
was designed for — an 8× cost miss we're transparent about in STATUS.md
and in the `Known limitations` section below. Use them to replicate the
benchmark or for one-off hard-query lookups; do not base a production
cost model on them yet.

All three tiers stay in the same integer-pointer fidelity contract as
`/recall` — only indices cross the LLM boundary.

### Per-category zero-LLM breakdown (LongMemEval_S, 470 q)

| Category | n | Zero-LLM R@1 | Zero-LLM R@5 | Ceiling (flagship) |
|---|---|---|---|---|
| single-session-assistant | 56 | **100%** | 100% | 98.2% (regresses) |
| knowledge-update | 72 | **96%** | 100% | 98.6% |
| single-session-user | 64 | **95%** | 100% | 100% |
| multi-session | 121 | 83% | 100% | 99.2% |
| single-session-preference | 30 | 67% | 97% | 86.7% |
| temporal-reasoning | 127 | 66% | 94% | 92.1% |
| **Blended** | **470** | **83.2%** | — | 96.4% |

**Read this honestly:** the zero-LLM tier is near-perfect on single-session
queries and knowledge-update. It craters on temporal-reasoning and
preference. The LLM tier buys +26pp on TR and +20pp on Pref — at the cost
of 80% escalation rate in the current calibration. If your workload is
session-scoped or fact-like, stay zero-LLM. If it's temporal-heavy, enable
the filter tier and budget for it.

### When to enable the optional LLM tier

- Your workload has **temporal-reasoning queries** ("what did we discuss before the release?").
- You need **preference-ranking** between multiple near-matches.
- You're **replicating the benchmark**, not running production.

Otherwise: zero-LLM default is the answer. It's why this repo exists.

```
Query
  │
  ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 1 — hybrid retrieval (zero-LLM)                          │
│   • sub-query decomposition + vocab expansion (recall_b logic) │
│   • dense retrieval with nomic search_query: / search_document:│
│     prefixes                                                    │
│   • BM25 over the candidate pool (bm25s, optional extra)       │
│   • Reciprocal Rank Fusion across runs                         │
│   • cosine-blended rerank against the original query           │
└───────────────────────────────────────────────────────────────┘
  │
  ▼
┌───────────────────────────────────────────────────────────────┐
│ Router (regex classifier)                                      │
│   • "you told me" / "you said"  → skip (keep Stage 1)          │
│   • "how many" / "what date"    → call cheap filter            │
│   • everything else             → keep Stage 1 at filter tier  │
└───────────────────────────────────────────────────────────────┘
  │
  ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 2 — cheap filter  (tier="filter", default)               │
│   500-char snippets, integer-pointer output                    │
└───────────────────────────────────────────────────────────────┘
  │
  ▼
┌───────────────────────────────────────────────────────────────┐
│ Stage 3 — flagship rerank  (tier="flagship", opt-in)           │
│   2000-char snippets, stronger model, integer output           │
│   Called when Stage 1 confidence is low or filter failed       │
└───────────────────────────────────────────────────────────────┘
```

### Tier tradeoffs

| Tier | Latency | R@1 (LongMemEval_S) | External calls | When to use |
|---|---|---|---|---|
| `zero_llm` | ~500ms | — | none | latency-sensitive paths, cost-sensitive ops |
| `filter` | ~1300ms | 90%+ | cheap filter LLM | default; temporal/counting queries benefit |
| `flagship` | ~3500ms | **96.4%** | filter + flagship model | hard queries, long sessions, benchmark setup |

### Quick start

```bash
# Optional dependency for best BM25 fusion (zero deps fallback if absent)
pip install fidelis[hybrid]

# Opt-in: set a filter endpoint (any OpenAI-compatible API)
export COGITO_FILTER_ENDPOINT=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
export COGITO_FILTER_TOKEN=sk-your-key
export COGITO_FILTER_MODEL=qwen-turbo

# Optional: flagship tier (stronger model, 4x larger context window)
export COGITO_FLAGSHIP_ENDPOINT=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
export COGITO_FLAGSHIP_TOKEN=sk-your-key
export COGITO_FLAGSHIP_MODEL=qwen-max
# Or simpler — if DASHSCOPE_API_KEY is set, the flagship tier auto-configures
export DASHSCOPE_API_KEY=sk-your-key

fidelis recall-hybrid "auth architecture decisions" --tier filter
```

HTTP:

```bash
curl -X POST http://127.0.0.1:19420/recall_hybrid \
  -H "Content-Type: application/json" \
  -d '{"text": "how many auth migrations have we done", "tier": "flagship", "limit": 10}'
```

Python:

```python
from fidelis.recall_hybrid import recall_hybrid
from fidelis.config import load, mem0_config
from mem0 import Memory

cfg = load()
mem = Memory.from_config(mem0_config(cfg))
hits, method = recall_hybrid(mem, "auth tokens", user_id="agent", cfg=cfg, tier="filter")
```

Graceful degradation: if the filter endpoint isn't configured, `tier="filter"`
falls back to `zero_llm`. If the flagship endpoint isn't configured,
`tier="flagship"` falls back to `filter` (which itself may degrade to
`zero_llm`). Nothing raises on missing credentials.

### Regression notice

The hybrid path was validated at **96.4% R@1 on LongMemEval_S** (470 questions,
runP-v35, 2026-04-18; multi-turn dialog retrieval with turn-level chunking and
session-date scaffolds). **On the internal 31-case eval** (which measures
keyword-recall over a store of short atomic facts), `/recall_hybrid` scores
lower on R@1 than the existing `/recall` path — they solve different problems.
The hybrid path wins on hit@any, semantic-gap queries, and multi-memory
aggregation; the existing path wins on prefix-style direct lookup. Default
behavior is unchanged — `/recall` still drives `fidelis recall`. Use
`/recall_hybrid` or `fidelis recall-hybrid` when you need the hybrid trait.

---


### Claude Code session memory

fidelis v0.3.0 can ingest your Claude Code sessions and query them with the
same turn-pair chunking used in the LongMemEval benchmark.

**Why it matters:** The 96.4% R@1 benchmark result requires role-structured session
data (user+assistant turn pairs). Flat atomic memories don't have this structure.
Claude Code stores every session as role-structured JSONL — wiring it to fidelis
uses the same retrieval architecture as the benchmark, though Claude Code sessions
have not been independently evaluated at the same scale.

**Install:**
```bash
# No new dependencies — uses existing chromadb + Ollama (nomic-embed-text)
```

**Ingest:**
```bash
# Preview (dry run)
python3 -m fidelis.ingest_claude_sessions --since 2026-04-11 --dry-run

# Ingest last 7 days
python3 -m fidelis.ingest_claude_sessions --since 2026-04-11

# Ingest all sessions (may take a few minutes)
python3 -m fidelis.ingest_claude_sessions
```

**Query:**
```python
from fidelis.recall_sessions import query_sessions, query_both

# Session history only
results = query_sessions("what did we discuss about LPCI last week?", top_k=3)

# Atomic facts + session history side-by-side (no auto-merge)
both = query_both("fidelis architecture decisions")
```

**MCP tools (new in v0.3.0):**
- `cogito_recall_sessions` — query Claude Code sessions
- `cogito_recall_both` — 3 atomic + 3 session results side-by-side

**Expected accuracy:** ~80-93% depending on query specificity (same pipeline as
the benchmark; accuracy drops when querying across very long sessions because the
session embedding averages across all turn content).

**Privacy:** All data stays local. Embedding uses Ollama (localhost:11434, nomic-embed-text).
ChromaDB at `~/.cogito/store`. Nothing sent to any cloud API. Ingestion is explicit —
nothing runs automatically. Re-running ingest is idempotent (dedup via content hash).

Full demo: [`docs/claude-code-memory-demo.md`](docs/claude-code-memory-demo.md)


## HTTP API

All endpoints return JSON. Server runs on port `19420` by default.

### `GET /health`

```json
{"status": "ok", "count": 1484, "version": "0.2.0", "calibrated": true, "snapshot": true}
```

Fields: `count` = total memories in store; `calibrated` = vocab_map present; `snapshot` = snapshot.md exists.

---

### `GET /snapshot`

Returns the compressed index markdown built by `fidelis snapshot`.

```json
{"snapshot": "## Projects\n- **fidelis** — ...", "path": "/home/user/.cogito/snapshot.md"}
```

Returns 404 if no snapshot has been built yet.

---

### `POST /query`

Narrow vector search. L2 threshold filter only. No LLM call. Fast.

Request:
```json
{"text": "query string", "limit": 5}
```

Response:
```json
{"memories": [{"text": "...", "score": 0.87}]}
```

---

### `POST /recall`

Broad search + integer-pointer filter. Two stages: zero-LLM RRF candidate pool, then cheap LLM selects by index.

Request:
```json
{"text": "query string", "limit": 50, "threshold": 400}
```

Response:
```json
{"memories": [{"text": "...", "score": 0.87}], "method": "filter"}
```

`method` field: `"filter"` = filter ran successfully; `"fallback_*"` = graceful degradation, all candidates returned instead. Possible fallbacks: `fallback_no_endpoint`, `fallback_unreachable`, `fallback_parse_error`, `fallback_error`.

---

### `POST /recall_b`

Zero-LLM recall only. Sub-query decomposition + RRF. 127ms latency. Same hit@any as combined (96%) at lower cost.

Request:
```json
{"text": "query string", "limit": 50}
```

Response:
```json
{"memories": [{"text": "...", "score": 0.016}], "method": "decompose_4_v"}
```

`method` field: `"decompose_N"` = N sub-queries ran; `"decompose_N_v"` = vocab expansion applied.

---

### `POST /recall_hybrid`

Hybrid BM25 + dense + RRF retrieval with tiered LLM escalation. Port of
the architecture that reached **96.4% R@1 on LongMemEval_S**. See
[Hybrid recall](#hybrid-recall-964-r1-on-longmemeval_s) for the full
diagram and tradeoffs.

Request:
```json
{"text": "query string", "limit": 50, "tier": "filter", "top_k": 5}
```

`tier` is one of `"zero_llm"`, `"filter"` (default), `"flagship"`.
`top_k` is how many candidates the reranker sees (default 5).

Response:
```json
{"memories": [{"text": "...", "score": 0.72}], "method": "hybrid_12_bm25|filter"}
```

`method` field encodes the path taken (e.g. `"hybrid_24_bm25|default_s1"` =
Stage 1 order kept; `"hybrid_12_bm25|filter"` = cheap filter reranked;
`"hybrid_8_bm25|filter|flagship"` = both reranks ran; `"hybrid_6_nobm25_v|…"`
= bm25s not installed, vocab expansion active).

---

### `POST /store`

Write one memory verbatim. No extraction LLM. Agent decides the content.

This is the preferred write path for agent-curated content.

Request:
```json
{"text": "Switched from JWT to session tokens on 2026-03-27 due to compliance requirement", "id": "<optional uuid>"}
```

Response:
```json
{"id": "abc123...", "text": "Switched from JWT to session tokens..."}
```

---

### `POST /add`

Feed text through mem0's extraction LLM before storing. Extracts multiple atomic facts from unstructured input.

Use when you have raw/unstructured text and want automatic fact extraction.

Request:
```json
{"text": "free-form text to remember"}
```

Response:
```json
{"count": 3, "memories": ["extracted fact 1", "extracted fact 2", "extracted fact 3"]}
```

---

## CLI Reference

All CLI commands talk to the running HTTP server.

| Command | Description |
|---|---|
| `fidelis recall "query"` | Two-stage recall via running server |
| `fidelis recall "query" --limit 50 --raw` | Raw JSON output |
| `fidelis recall-hybrid "query" --tier filter` | Hybrid BM25+dense+RRF recall (96.4% R@1 arch) |
| `fidelis recall-hybrid "query" --tier flagship --top-k 5` | + flagship escalation on hard queries |
| `fidelis query "query"` | Simple vector query, no filter |
| `fidelis add "text"` | Add a memory via /add (mem0 extraction) |
| `fidelis seed ~/notes/` | Bulk-seed from markdown files via /store |
| `fidelis seed ~/notes/ --add` | Bulk-seed using /add (extraction mode) |
| `fidelis seed ~/notes/ --dry-run` | Preview without writing |
| `fidelis seed ~/notes/ --glob "*.txt"` | Custom file pattern |
| `fidelis snapshot` | Build compressed index layer |
| `fidelis snapshot --rebuild` | Force rebuild of snapshot |
| `fidelis snapshot --dry-run` | Preview snapshot without writing |
| `fidelis calibrate` | Build vocab bridge from corpus (one-time) |
| `fidelis calibrate --dry-run` | Preview vocab mappings |
| `fidelis health` | Check server status |
| `fidelis server` | Start the server (alias for fidelis-server) |
| `fidelis-server --port 19420` | Start server directly |
| `fidelis-server --config /path/to.json` | Start with explicit config file |

---

## Configuration

Priority: env vars > `.cogito.json` > defaults.

Config file is searched at `./.cogito.json` (cwd) then `~/.cogito/config.json`.

| Env var | Config key | Default | Description |
|---|---|---|---|
| `COGITO_PORT` | `port` | `19420` | Server port |
| `COGITO_USER_ID` | `user_id` | `"agent"` | Memory namespace (isolates stores) |
| `COGITO_FILTER_ENDPOINT` | `filter_endpoint` | — | OpenAI-compatible base URL for filter LLM |
| `COGITO_FILTER_TOKEN` | `filter_token` | — | Bearer token for filter endpoint |
| `COGITO_FILTER_MODEL` | `filter_model` | `anthropic/claude-haiku-4-5` | Filter LLM model name |
| `COGITO_FILTER_TIMEOUT_MS` | `filter_timeout_ms` | `12000` | Filter LLM timeout in ms |
| `ANTHROPIC_API_KEY` | `anthropic_api_key` | — | Direct Anthropic key (alternative to endpoint+token) |
| `COGITO_STORE_PATH` | `store_path` | `~/.cogito/store` | ChromaDB persistence path |
| `COGITO_COLLECTION` | `collection` | `cogito_memory` | ChromaDB collection name |
| `COGITO_OLLAMA_URL` | `ollama_url` | `http://localhost:11434` | Ollama base URL |
| `COGITO_LLM_MODEL` | `llm_model` | `mistral:7b` | LLM for fact extraction (/add) |
| `COGITO_EMBED_MODEL` | `embed_model` | `nomic-embed-text` | Embedding model |
| `COGITO_RECALL_LIMIT` | `recall_limit` | `50` | Candidate pool size for /recall and /recall_b |
| `COGITO_RECALL_THRESHOLD` | `recall_threshold` | `400.0` | L2 cutoff for /recall candidates |
| `COGITO_QUERY_THRESHOLD` | `query_threshold` | `250.0` | L2 cutoff for /query results |
| `COGITO_FLAGSHIP_ENDPOINT` | `flagship_endpoint` | — | OpenAI-compatible base URL for flagship rerank (recall_hybrid tier="flagship") |
| `COGITO_FLAGSHIP_TOKEN` | `flagship_token` | — | Bearer token for flagship endpoint |
| `COGITO_FLAGSHIP_MODEL` | `flagship_model` | — | Flagship model name (e.g. `qwen-max`) |
| `COGITO_FLAGSHIP_TIMEOUT_MS` | `flagship_timeout_ms` | `30000` | Flagship LLM timeout in ms |
| `DASHSCOPE_API_KEY` | — | — | If set, recall_hybrid auto-configures flagship to DashScope qwen-max |
| `COGITO_HYBRID_COSINE_WEIGHT` | `hybrid_cosine_weight` | `0.7` | Cosine vs RRF blend weight for hybrid retrieval (0..1) |

`filter_endpoint` accepts any OpenAI-compatible API: Anthropic gateway, LM Studio, Ollama's `/v1` compat layer, OpenClaw, etc.

For Ollama qwen3/qwen3.5 models used as filter, fidelis automatically switches to the native Ollama `/api/chat` endpoint with `think: false` to suppress thinking mode.

---

## Python API

```python
from fidelis.recall import recall
from fidelis.recall_b import recall_b
from fidelis.recall_hybrid import recall_hybrid
from fidelis.config import load, mem0_config
from mem0 import Memory

cfg = load()  # reads .cogito.json + env vars
memory = Memory.from_config(mem0_config(cfg))

# Two-stage recall (recommended default)
memories, method = recall(memory, "auth architecture", user_id=cfg["user_id"], cfg=cfg)
for m in memories:
    print(m["text"])  # verbatim stored text, never rephrased

# Zero-LLM recall (fast path)
memories, method = recall_b(memory, "auth architecture", user_id=cfg["user_id"], cfg=cfg)

# Hybrid recall (BM25 + dense + RRF + tiered LLM; 96.4% R@1 on LongMemEval_S)
memories, method = recall_hybrid(
    memory, "auth architecture", user_id=cfg["user_id"], cfg=cfg,
    tier="filter",  # "zero_llm" | "filter" | "flagship"
)

print(method)  # e.g. "filter", "decompose_4_v", "hybrid_12_bm25|filter"
```

---

## Why integers?

When the filter LLM outputs only `[3, 7, 12]`:

- It cannot rephrase memory text — it never generates it
- It cannot hallucinate new facts into the output
- It cannot summarize two memories into one
- An out-of-range integer is silently ignored; it cannot inject noise

Compare to asking the LLM to "return the relevant passages" — even with careful prompting, LLMs will reword, compress, or merge content. The integer-pointer pattern makes fidelity a structural property of the pipeline, not a prompt engineering goal.

The filter prompt:
```
Output ONLY a JSON array of integers, ordered from most to least relevant.
Examples: [1, 4, 7]   or   []
```

The server then picks `candidates[i]` — verbatim stored text — for each valid integer `i`. The path from storage to retrieval contains no text generation.

---

## Setup Recommendations

**Before benchmarking or deploying — run zer0lint first.**

Ingestion quality directly limits retrieval quality. A poorly formatted memory store will underperform regardless of retrieval method. The technical extraction prompt baked into fidelis's default config was validated to produce 0%→100% ingestion improvement via zer0lint diagnostics.

**Session start pattern (agent integration):**

```python
# 1. Load snapshot into context once at session start
import urllib.request, json
resp = urllib.request.urlopen("http://127.0.0.1:19420/snapshot")
snapshot = json.loads(resp.read())["snapshot"]
# inject snapshot into system prompt or first user message

# 2. Query per-message via /recall
# 3. Write new facts via /store (agent-curated) or /add (extraction)
```

**Calibrate for domain-specific vocabulary:**

```bash
fidelis calibrate  # reads your corpus, writes vocab_map to .cogito.json
# then restart server to pick up new vocab_map
```

Calibration builds a plain-English → technical term bridge. Example: "how fast" → ["latency", "throughput", "ms"]. Improves recall_b on domain-specific queries without adding LLM calls.

---

## Built by Hermes Labs

fidelis is part of the [Hermes Labs](https://hermes-labs.ai) AI agent tooling suite:

- **[zer0lint](https://github.com/roli-lpci/zer0lint)** — Memory extraction diagnostics. Run before benchmarking to verify store quality. The technical extraction prompt in fidelis's default config was validated against zer0lint.
- **[zer0dex](https://github.com/roli-lpci/zer0dex)** — Dual-layer memory architecture pattern that fidelis implements.
- **[lintlang](https://github.com/roli-lpci/lintlang)** — Static linter for AI agent tool descriptions and prompts
- **[Little Canary](https://github.com/roli-lpci/little-canary)** — Prompt injection detection
- **[Suy Sideguy](https://github.com/roli-lpci/suy-sideguy)** — Runtime policy enforcement for agents
- **fidelis** — Two-stage memory retrieval ← you are here

---

## Operations

`fidelis-server` is designed to run under a process supervisor. On macOS the
reference deployment is a `launchctl`-managed user daemon; on Linux any
supervisor (systemd, runit, s6) works.

### Health probe

```bash
curl -s http://127.0.0.1:19420/health
# {"status":"ok","count":N,"version":"0.3.1","calibrated":true,"snapshot":true}
```

Non-200 or empty response means the server is down or the handler is wedged.

### Recovery patterns (macOS launchd)

| Symptom | Remediation |
|---|---|
| `/health` returns nothing, `launchctl list` shows no cogito entry | Label is disabled. `launchctl enable gui/$(id -u)/ai.hermeslabs.fidelis-server && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.hermeslabs.fidelis-server.plist` |
| `/health` accepts then resets (HTTP 000 on curl) | Handler wedged. `launchctl kickstart -k gui/$(id -u)/ai.hermeslabs.fidelis-server` — `-k` is required; plain `start` is a no-op when a process is running but stuck. |
| Writes silently failing | Ollama or mem0 dependency down. `safe_add` queues to `~/.cogito/queue/`; drain with `fidelis replay` once the dependency is back. |

### Observability

`fidelis.telemetry` logs each escalation decision to
`~/.cogito/escalation.log` (JSONL, append-only, crash-safe). Summarise
with:

```python
from fidelis.telemetry import rate
rate(window_n=100)
# {"n": 100, "escalated": 12, "rate": 0.12, "by_route": {...}}
```

Escalation rate is the leading indicator of the known calibration miss
(80% actual vs 10% intended on the filter/flagship tiers). Zero-LLM tier
records 0% escalation by construction.

### Write-path resilience

All writes go through `fidelis.degrade.safe_add`:

- Dependency up → memory stored + response returned
- Dependency down → write queued to `~/.cogito/queue/<ts>-<uuid>.json`,
  success returned; no data lost
- Call `fidelis.degrade.replay_queue(memory, user_id)` when the
  dependency recovers to drain

See `tests/test_graceful_degrade.py` for the state machine and
`tests/test_graceful_degrade_corruption.py` for the corrupt-queue-file
branch coverage.

---

## Roadmap

- [ ] Pluggable vector backends (pgvector, Qdrant, LlamaIndex)
- [ ] Pluggable extraction backends (non-Ollama)
- [ ] Session flush utility (end-of-session seeding)
- [ ] Benchmark harness as public CLI (`fidelis bench`)
- [ ] Streaming /recall response
- [ ] Per-qtype escalation calibration (unblocks filter/flagship graduation to default)
- [ ] Dispatcher / Path A–B split (`docs/DISPATCHER_DESIGN.md`)

---

## License

MIT
