---
name: fidelis
description: Use when you need agent memory with a zero-LLM default retrieval path — returning stored passages verbatim via BM25 + dense-vector + reciprocal-rank-fusion — and want an HTTP server plus CLI that can also do compressed-index snapshotting and Claude Code/Codex/Copilot/Gemini/OpenClaw MCP wiring. Local-first, PyPI package `fidelis-memory`.
license: MIT
compatibility: Requires Python 3.10+; installs via `pip install fidelis-memory==0.1.0`, executable name is `fidelis`. Zero-LLM retrieval path needs no external API key; optional filter/extraction tiers need Ollama or an Anthropic/OpenAI-compatible endpoint.
---

# fidelis

fidelis is local-first retrieval memory for AI agents and Claude Code. The
default zero-LLM path returns stored passages verbatim — no LLM call, no
rephrasing risk. It layers a compressed markdown snapshot index (~741 tokens)
on top of two-stage recall (BM25 + dense + reciprocal-rank-fusion) for
cross-reference queries flat memory or vector-only RAG miss.

## Use it for

- Standing up a local HTTP memory server an agent host queries before/instead
  of a model call
- Zero-LLM recall (`recall-hybrid`, `recall_b`) where verbatim, unmodified
  text must come back
- Bulk-seeding a corpus from markdown/text files and querying it immediately
- Wiring an MCP memory tool into Claude Code, Codex, GitHub Copilot CLI,
  Gemini CLI, or OpenClaw via `fidelis mcp install`

## Do not use it for

- A hosted, multi-tenant memory platform (this is a local process/service)
- A guarantee that retrieval accuracy transfers unchanged to a different
  corpus or workload — published numbers are project measurements on
  LongMemEval-S, not independent replication
- Proving a model's stored claim is factually true — fidelis returns what was
  stored, it does not fact-check it

## Quickstart

```bash
pip install "fidelis-memory==0.1.0"
fidelis health
```

Or without installing, via [uv](https://docs.astral.sh/uv/):

```bash
uvx --from fidelis-memory fidelis health
```

Real output against a running local instance:

```
status: ok  |  memories: 147980  |  version: 1.0.0a2  |  calibrated: yes  |  snapshot: yes
```

Zero-LLM vector query:

```bash
uvx --from fidelis-memory fidelis query "test query" --limit 2
```

```
2 memories:

  [1]  score 0.686
      User is 'test'

  [2]  score 0.686
      [fact] Reliable testing procedure for Google Rich Results Test: navigate fresh to page, fill test URL textbox, press Escape, click test URL button, wait ~20s for results
```

## Output shape

- `health`: one-line status, memory count, version, calibration/snapshot state
- `query` / `recall` / `recall-hybrid`: ranked `{text, score}` memories, plus
  a `method` field naming the retrieval path taken (`filter`,
  `fallback_*`, `decompose_N[_v]`)
- HTTP endpoints mirror the CLI 1:1 (`/health`, `/recall`, `/recall_hybrid`,
  `/query`, `/store`, `/add`, `/snapshot`, `/replay`)

## Common gotchas

- Executable name is `fidelis`, not `fidelis-memory` — `uvx fidelis-memory`
  fails; use `uvx --from fidelis-memory fidelis <cmd>`.
- The PyPI project literally named `fidelis` is unrelated; install
  `fidelis-memory`.
- The optional filter/extraction tiers call out to Ollama or an
  Anthropic/OpenAI-compatible endpoint — the default `/recall_b` and
  `/query` paths do not.

## More

Full docs, HTTP reference, and module map:
https://github.com/hermes-labs-ai/fidelis
