# cogito-ergo

A memory layer for AI agents.

Two-stage retrieval: broad vector search + cheap LLM re-ranker that outputs only integer indices, never memory text. The re-ranker cannot corrupt, summarize, or hallucinate into the content returned to your agent — structurally, not by convention.

```
Agent calls /recall
    ↓
Broad vector search  →  50 candidates
    ↓
Haiku sees: [1] memory text...  [2] memory text...  ...
Haiku outputs: [3, 7, 12, 19]   ← integers only, no text
    ↓
Server fetches candidates[3], candidates[7], ...
    ↓
Agent receives: verbatim stored text
```

---

## Install

```bash
pip install cogito-ergo
```

Requires a running [Ollama](https://ollama.ai) instance with `mistral:7b` and `nomic-embed-text`.

```bash
ollama pull mistral:7b
ollama pull nomic-embed-text
```

---

## Quickstart

**1. Configure**

```bash
cp .cogito.example.json .cogito.json
# edit: set filter_endpoint + filter_token, or set ANTHROPIC_API_KEY
```

Or via env vars:

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # direct Anthropic (no gateway needed)
export COGITO_FILTER_MODEL=anthropic/claude-haiku-4-5
```

**2. Start the server**

```bash
cogito-server
# or: cogito server
```

**3. Use it**

```bash
# CLI
cogito recall "what did we decide about the auth architecture"
cogito add "Switched from JWT to session tokens on 2026-03-27 due to compliance requirement"
cogito health

# HTTP (any language)
curl -X POST http://127.0.0.1:19420/recall \
  -H "Content-Type: application/json" \
  -d '{"text": "auth architecture decisions"}'
```

---

## HTTP API

All endpoints return JSON.

### `GET /health`
```json
{"status": "ok", "count": 1484, "version": "0.1.0"}
```

### `POST /query`
Narrow search with L2 threshold filter. Fast, no LLM call.
```json
{"text": "query string", "limit": 5}
→ {"memories": [{"text": "...", "score": 93.4}]}
```

### `POST /recall`
Broad search + Haiku integer-pointer filter. Smarter, one extra LLM call.
```json
{"text": "query string", "limit": 50, "threshold": 400}
→ {"memories": [{"text": "...", "score": 93.4}], "method": "filter"}
```

`method` tells you what happened: `"filter"` (Haiku ran), `"fallback_*"` (graceful degradation — all candidates under threshold returned instead).

### `POST /add`
Add a memory (extracted via LLM, stored in vector DB).
```json
{"text": "free-form text to remember"}
→ {"count": 3, "memories": ["extracted fact 1", ...]}
```

---

## Configuration

Priority: env vars > `.cogito.json` > defaults.

| Env var | Config key | Default | Description |
|---|---|---|---|
| `COGITO_PORT` | `port` | `19420` | Server port |
| `COGITO_USER_ID` | `user_id` | `"agent"` | Memory namespace |
| `COGITO_FILTER_ENDPOINT` | `filter_endpoint` | — | OpenAI-compatible base URL for filter LLM |
| `COGITO_FILTER_TOKEN` | `filter_token` | — | Bearer token for filter endpoint |
| `COGITO_FILTER_MODEL` | `filter_model` | `anthropic/claude-haiku-4-5` | Filter model name |
| `ANTHROPIC_API_KEY` | `anthropic_api_key` | — | Direct Anthropic key (alternative to endpoint+token) |
| `COGITO_STORE_PATH` | `store_path` | `~/.cogito/store` | ChromaDB persistence path |
| `COGITO_OLLAMA_URL` | `ollama_url` | `http://localhost:11434` | Ollama base URL |
| `COGITO_LLM_MODEL` | `llm_model` | `mistral:7b` | LLM for fact extraction |
| `COGITO_EMBED_MODEL` | `embed_model` | `nomic-embed-text` | Embedding model |
| `COGITO_RECALL_LIMIT` | `recall_limit` | `50` | Candidate pool size for /recall |
| `COGITO_RECALL_THRESHOLD` | `recall_threshold` | `400.0` | L2 cutoff for candidates |
| `COGITO_QUERY_THRESHOLD` | `query_threshold` | `250.0` | L2 cutoff for /query |

The filter endpoint accepts **any OpenAI-compatible API** — Anthropic gateway, local LM Studio, Ollama's OpenAI-compat layer, etc.

---

## Python API

```python
from cogito.recall import recall
from cogito.config import load, mem0_config
from mem0 import Memory

cfg = load()  # reads .cogito.json + env vars
memory = Memory.from_config(mem0_config(cfg))

memories, method = recall(memory, "auth architecture", user_id=cfg["user_id"], cfg=cfg)
for m in memories:
    print(m["text"])
```

---

## Why integers?

When the filter LLM outputs only `[3, 7, 12]`:

- It cannot rephrase memory text (it never generates it)
- It cannot hallucinate new facts into the output
- It cannot summarize two memories into one
- An out-of-range integer is silently ignored

Compare to asking the LLM to "return the relevant passages" — even with careful prompting, LLMs will reword, compress, or merge content. The integer-pointer pattern makes fidelity a structural property of the pipeline, not a prompt engineering goal.

---

## Roadmap

- [ ] Pluggable vector backends (LlamaIndex, pgvector, Qdrant)
- [ ] Pluggable extraction backends (non-Ollama)
- [ ] MEMORY.md hot-layer management (always-in-context index)
- [ ] Session flush utility (end-of-session seeding)
- [ ] Benchmark against naive top-N retrieval

---

## License

MIT
