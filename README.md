# fidelis

**Fidelity-preserving memory for AI agents. Local. 60 seconds. No API keys.**

You store: `auth tokens expire after 3600 seconds`.
A normal RAG memory hands your agent back: `authentication has a configurable timeout`.
**fidelis** hands your agent back: `auth tokens expire after 3600 seconds` — the original passage, not a paraphrase.

Memory doesn't just forget. It mutates. Every retrieval system that uses an LLM to rank or rewrite memories has the same failure mode: the specific fact gets summarized into something general. fidelis solves this structurally — there is no LLM in the default retrieval path.

```bash
pip install fidelis
fidelis init                  # background service (launchd / systemd)
# Requires Ollama with nomic-embed-text — see Requirements below
fidelis watch ~/notes         # auto-ingests markdown
fidelis mcp install           # wires Claude Code
# Restart Claude Code. Memory is on.
```

v0.0.8 — pre-release. Benchmarked on LongMemEval-S; results below.

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: pre-release](https://img.shields.io/badge/status-pre--release-orange)](#known-limitations)
[![Made by Hermes Labs](https://img.shields.io/badge/made%20by-Hermes%20Labs-purple)](https://hermes-labs.ai)

---

## How it fits

```
your notes / sessions
       ↓
local memory store      (~/.cogito/, fully local)
       ↓
fidelis retrieval       (BM25 + dense + RRF, no LLM)
       ↓
original passages       (verbatim — never rephrased)
       ↓
Claude Code / your agent
```

Claude Code is the easiest place to start. The retrieval engine is agent-agnostic — pair it with any LLM client.

## A quick demo

You drop a note into a watched folder:

```text
# 2026-04-15 — auth call
Steve confirmed: auth tokens expire after 3600 seconds.
Token refresh requires an exchange of the refresh-token JWT against /v2/refresh.
The 3600s window is non-configurable in our current contract.
```

Two days later you ask Claude Code: *"how long do auth tokens last?"*

Without fidelis, the agent answers from training: *"OAuth tokens typically last 3600 seconds, though this varies by provider."* Generic, plausible, wrong if your contract differs.

With fidelis, the MCP `fidelis_recall` tool fires before Claude composes its answer. Claude sees:

```text
Steve confirmed: auth tokens expire after 3600 seconds.
The 3600s window is non-configurable in our current contract.
```

The answer comes back grounded in your stored passage, with the *non-configurable* qualifier intact — a detail an LLM-rephrasing memory would have lost.

## Why "fidelity-preserving"

| | typical RAG memory | **fidelis** |
|---|---|---|
| LLM in retrieval path | yes | **no** (zero-LLM default) |
| Returned content | LLM-paraphrased | **original passages** |
| Cost per query | $0.001–0.02 | **$0** |
| Works offline / air-gapped | no | **yes** |
| API keys to start | yes | **no** |

The optional LLM tier (`tier="filter"` / `tier="flagship"`) returns only integer pointers (`[3, 7, 12]`) — the server dereferences them to the original stored text. The LLM is structurally prevented from rephrasing memory content. Fidelity is a property of the architecture, not a prompting convention.

## Benchmarks

LongMemEval-S, 470 questions, public benchmark.

| Metric | Value |
|---|---|
| Retrieval R@1 | **83.2%** |
| Retrieval R@5 | **98.3%** |
| End-to-end QA accuracy | **73.0%**, Wilson 95% CI [68.7%, 77.0%] |
| Cost per query | **$0** (local + Claude subscription) |
| Mean retrieval latency | ~90 ms |

Raw evidence: [`bench/runs/zeroLLM-full-20260424/aggregate.json`](bench/runs/zeroLLM-full-20260424/aggregate.json) · [`experiments/zeroLLM-FLAGSHIP-evidence/SUMMARY.json`](experiments/zeroLLM-FLAGSHIP-evidence/SUMMARY.json)

The QA tier wraps your existing LLM with a 140–180-token system prompt — the Fidelis Scaffold. See [`docs/scaffold.md`](docs/scaffold.md).

## Quick reference

```bash
fidelis recall "what did the user say about Sarah"
fidelis query  "Sarah" --limit 5
fidelis health
fidelis seed   ~/memory/   ~/notes/
```

Python helper for direct integration:

```python
from fidelis.augment import augment
from anthropic import Anthropic

client = Anthropic()
answer = augment(
    question="What did I say about Sarah?",
    qtype="single-session-user",
    llm_call=lambda system, user: client.messages.create(
        model="claude-opus-4-5",
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=512,
    ).content[0].text,
)
```

## What's running on your machine

After `fidelis init`:

- **Service:** `fidelis-server` runs at `http://127.0.0.1:19420` under your OS service manager (launchd on macOS, systemd on Linux). Auto-starts on boot. Logs at `~/.fidelis/server.log`.
- **Storage:** vector store (Chroma + SQLite) at `~/.cogito/`. Fully local. No data leaves your machine in the default zero-LLM path.
- **MCP:** if you ran `fidelis mcp install`, Claude Code sees three tools: `fidelis_recall`, `fidelis_query`, `fidelis_health`.

To stop: `fidelis init --uninstall`. To wipe: `rm -rf ~/.cogito ~/.fidelis`.

## Requirements

- macOS or Linux (Windows not yet supported)
- Python 3.10+
- [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled (~280 MB):

  ```bash
  brew install ollama && ollama serve &
  ollama pull nomic-embed-text   # ~280 MB, one-time
  ```

The full init-to-first-recall cycle is under 60 seconds once Ollama is up.

## Known limitations (v0.0.8 honest list)

- **Pre-release.** API may change. Pin the version if you build on it.
- **Best on macOS Sequoia / Ubuntu 24.04 LTS.** Other OSes likely work but aren't gate-tested.
- **Temporal-reasoning and preference questions are the weakest qtypes** in the QA scaffold (TR ~58%, Pref ~37% on the full eval). Single-session and knowledge-update qtypes are strong (95–100%).
- **The optional LLM tier ("flagship" mode) currently escalates ~80% of queries instead of the intended ~10%** — an 8× cost miss we're transparent about. The default zero-LLM tier is unaffected.
- **qwen3.5:9b in thinking mode does not reliably follow the literal hedge instruction** in the Fidelis Scaffold. Use Claude, an OpenAI-format API, or non-thinking-mode local models for reliable hedging.

## Going deeper

- [`docs/full-reference.md`](docs/full-reference.md) — full architecture, hybrid recall tiers, HTTP API, troubleshooting
- [`docs/scaffold.md`](docs/scaffold.md) — Fidelis Scaffold contract + drift-detection markers
- [`experiments/zeroLLM-FLAGSHIP-evidence/`](experiments/zeroLLM-FLAGSHIP-evidence/) — raw eval JSONs + machine-readable SUMMARY

## License

MIT. Built by Hermes Labs (Roli Bosch). Issues + PRs welcome.

*Part of the Hermes Labs agent reliability stack.*
