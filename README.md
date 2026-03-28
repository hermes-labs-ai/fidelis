# cogito-ergo

Two-stage memory retrieval for AI agents. **85% R@1, 96% hit@any** on 31 test cases. The filter LLM outputs only integers — it structurally cannot corrupt, rephrase, or hallucinate into the content returned to your agent.

[![PyPI version](https://img.shields.io/pypi/v/cogito-ergo)](https://pypi.org/project/cogito-ergo/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/cogito-ergo/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Made by Hermes Labs](https://img.shields.io/badge/made%20by-Hermes%20Labs-8a2be2)](https://github.com/roli-lpci)

---

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              cogito-ergo server              │
                    │                 :19420                       │
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │             SNAPSHOT LAYER                   │
                    │   Compressed markdown index (~741 tokens)    │
                    │   Built once from corpus via `cogito snapshot`│
                    │   Returned with /recall — no vector search   │
                    │   Solves cross-reference queries (0%→50% R@1)│
                    └──────────────────┬──────────────────────────┘
                                       │
                    ┌──────────────────▼──────────────────────────┐
                    │         STAGE 1 — recall_b (zero-LLM)       │
                    │   Query decomposition → sub-queries          │
                    │   Stop-word stripping + bigrams + trigrams   │
                    │   Vocab expansion (cogito calibrate)         │
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

Measured 2026-03-28. 31 test cases, Anthropic claude-haiku-4-5 as filter model.

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
pip install cogito-ergo
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
cogito-server
# or: cogito server
```

**5. First recall**

```bash
cogito recall "what did we decide about the auth architecture"
```

```bash
curl -X POST http://127.0.0.1:19420/recall \
  -H "Content-Type: application/json" \
  -d '{"text": "auth architecture decisions"}'
```

---

## HTTP API

All endpoints return JSON. Server runs on port `19420` by default.

### `GET /health`

```json
{"status": "ok", "count": 1484, "version": "0.1.0", "calibrated": true, "snapshot": true}
```

Fields: `count` = total memories in store; `calibrated` = vocab_map present; `snapshot` = snapshot.md exists.

---

### `GET /snapshot`

Returns the compressed index markdown built by `cogito snapshot`.

```json
{"snapshot": "## Projects\n- **cogito-ergo** — ...", "path": "/home/user/.cogito/snapshot.md"}
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
{"memories": [{"text": "...", "score": 93.4}]}
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
{"memories": [{"text": "...", "score": 93.4}], "method": "filter"}
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
| `cogito recall "query"` | Two-stage recall via running server |
| `cogito recall "query" --limit 50 --raw` | Raw JSON output |
| `cogito query "query"` | Simple vector query, no filter |
| `cogito add "text"` | Add a memory via /add (mem0 extraction) |
| `cogito seed ~/notes/` | Bulk-seed from markdown files via /store |
| `cogito seed ~/notes/ --add` | Bulk-seed using /add (extraction mode) |
| `cogito seed ~/notes/ --dry-run` | Preview without writing |
| `cogito seed ~/notes/ --glob "*.txt"` | Custom file pattern |
| `cogito snapshot` | Build compressed index layer |
| `cogito snapshot --rebuild` | Force rebuild of snapshot |
| `cogito snapshot --dry-run` | Preview snapshot without writing |
| `cogito calibrate` | Build vocab bridge from corpus (one-time) |
| `cogito calibrate --dry-run` | Preview vocab mappings |
| `cogito health` | Check server status |
| `cogito server` | Start the server (alias for cogito-server) |
| `cogito-server --port 19420` | Start server directly |
| `cogito-server --config /path/to.json` | Start with explicit config file |

---

## Configuration

Priority: env vars > `.cogito.json` > defaults.

Config file is searched at `./cogito.json` (cwd) then `~/.cogito/config.json`.

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

`filter_endpoint` accepts any OpenAI-compatible API: Anthropic gateway, LM Studio, Ollama's `/v1` compat layer, OpenClaw, etc.

For Ollama qwen3/qwen3.5 models used as filter, cogito automatically switches to the native Ollama `/api/chat` endpoint with `think: false` to suppress thinking mode.

---

## Python API

```python
from cogito.recall import recall
from cogito.recall_b import recall_b
from cogito.config import load, mem0_config
from mem0 import Memory

cfg = load()  # reads .cogito.json + env vars
memory = Memory.from_config(mem0_config(cfg))

# Two-stage recall (recommended)
memories, method = recall(memory, "auth architecture", user_id=cfg["user_id"], cfg=cfg)
for m in memories:
    print(m["text"])  # verbatim stored text, never rephrased

# Zero-LLM recall (fast path)
memories, method = recall_b(memory, "auth architecture", user_id=cfg["user_id"], cfg=cfg)

print(method)  # e.g. "filter" or "decompose_4_v"
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

Ingestion quality directly limits retrieval quality. A poorly formatted memory store will underperform regardless of retrieval method. The technical extraction prompt baked into cogito's default config was validated to produce 0%→100% ingestion improvement via zer0lint diagnostics.

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
cogito calibrate  # reads your corpus, writes vocab_map to .cogito.json
# then restart server to pick up new vocab_map
```

Calibration builds a plain-English → technical term bridge. Example: "how fast" → ["latency", "throughput", "ms"]. Improves recall_b on domain-specific queries without adding LLM calls.

---

## Related Tools

Both tools by [Hermes Labs](https://github.com/roli-lpci).

**[zer0lint](https://github.com/roli-lpci/zer0lint)** — ingestion diagnostics for cogito-ergo stores. Run before benchmarking to verify store quality. Detects malformed extractions, duplicate facts, low-information entries, and coverage gaps. The technical extraction prompt in cogito's default config was validated against zer0lint.

**[zer0dex](https://github.com/roli-lpci/zer0dex)** — the architecture pattern cogito-ergo implements. zer0dex defines the dual-layer retrieval pattern: compressed index (snapshot layer) + vector store with integer-pointer fidelity. cogito-ergo is the Python implementation of this pattern for agent memory.

---

## Roadmap

- [ ] Pluggable vector backends (pgvector, Qdrant, LlamaIndex)
- [ ] Pluggable extraction backends (non-Ollama)
- [ ] Session flush utility (end-of-session seeding)
- [ ] Benchmark harness as public CLI (`cogito bench`)
- [ ] Streaming /recall response

---

## License

MIT
