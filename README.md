# Fidelis Memory

<!-- mcp-name: io.github.hermes-labs-ai/fidelis-memory -->

## Local-first, zero-LLM memory for Codex, Claude Code, and AI agents.

**73.0% end-to-end QA on LongMemEval-S. 83.2% R@1 retrieval. $0/query. No LLM in the default retrieval path.**

Stop re-explaining context to your agent. fidelis returns your original notes verbatim, local-first, fast, about 60 seconds to install. Your agent already calls an LLM to think; it should not need another one just to remember. Designed for developers. The default zero-LLM retrieval path does not send memory content to an LLM. The documented `fidelis init` service configuration also disables mem0 and Chroma telemetry. That can reduce third-party data exposure, but deployments still own their security and compliance assessment.

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: pre-release](https://img.shields.io/badge/status-pre--release-orange)](#known-limitations)
[![CI tests: 368 passing](https://img.shields.io/badge/CI%20tests-368%20passing-brightgreen)](tests/)
[![Official MCP Registry](https://img.shields.io/badge/MCP%20Registry-active-5b5bd6)](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.hermes-labs-ai%2Ffidelis-memory/versions/0.0.95)
[![Made by Hermes Labs](https://img.shields.io/badge/made%20by-Hermes%20Labs-purple)](https://hermes-labs.ai)

```
your notes / sessions
       ↓
local memory store      (~/.cogito/, fully local)
       ↓
fidelis retrieval       (BM25 + dense + RRF, no LLM)
       ↓
original passages       (verbatim, never rephrased)
       ↓
Codex / Claude Code / your agent
```

What fidelis is:

- **fast** - ~216 ms local retrieval (full benchmark mean; vector-only path is faster)
- **cheap** - $0/query retrieval cost
- **private** - local memory store by default
- **faithful** - original stored passages returned, not paraphrases
- **proven** - benchmarked on LongMemEval-S (470 questions, public benchmark), with raw evidence in [`experiments/zeroLLM-FLAGSHIP-evidence/`](experiments/zeroLLM-FLAGSHIP-evidence/)
- **installable** - Codex or Claude Code via MCP in about 60 seconds

---

## Quickstart

```bash
# 0. one-time: Ollama + the local embedder (~280 MB)
brew install ollama && ollama serve &
ollama pull nomic-embed-text

# 1. install Fidelis Memory from PyPI
python3 -m pip install "fidelis-memory==0.0.95"
fidelis init                  # background service (launchd / systemd)
fidelis watch ~/notes         # auto-ingests markdown
fidelis mcp install --client codex   # or omit --client for Claude Code
fidelis mcp serve             # runs the MCP server over stdio
# Restart your agent client. Memory is on.
```

> **Package-name note:** install Hermes Labs' package as `fidelis-memory`.
> The import name and CLI remain `fidelis`. The separate PyPI project named
> `fidelis` belongs to [NGdust/fidelis](https://github.com/NGdust/fidelis).

Linux users swap `brew install ollama` for the equivalent install from [ollama.com](https://ollama.com). [See Requirements](#requirements).

Fidelis Memory 0.0.95 is also published in the
[official MCP Registry](https://registry.modelcontextprotocol.io/v0.1/servers/io.github.hermes-labs-ai%2Ffidelis-memory/versions/0.0.95)
as `io.github.hermes-labs-ai/fidelis-memory`. Registry-aware clients can launch
the same released server directly from PyPI:

```bash
uvx --from "fidelis-memory==0.0.95" fidelis mcp serve
```

This starts the MCP stdio process; run `fidelis init` first when the local
Fidelis service and store have not already been configured. Version 0.0.94
introduced supported Codex MCP installation and context-sensitive orientation;
0.0.95 added the independently discoverable registry release.

## What you notice immediately

After the four commands above, the next time you open Codex or Claude Code:

- It stops asking you to repeat context you already wrote down.
- You can ask "what did we decide last week about auth?" - and the answer cites your actual decision, not a generic OAuth lecture.
- Architecture rationale you wrote in a markdown file two months ago surfaces when relevant.
- Your project context carries across sessions instead of resetting at every new conversation.
- Failed migration notes, naming conventions, founder voice memos - all queryable in your agent's normal flow.

Most of fidelis's value is *not* the benchmark; it's not having to explain the same thing twice.

## Most AI memory systems rewrite your notes

Most memory systems rephrase content on the way out. The specific fact gets summarized into something general. fidelis solves this structurally - there is no LLM in the default retrieval path, so the store returns exactly what you put in.

You store:

```text
auth tokens expire after 3600 seconds.
The 3600s window is non-configurable in our current contract.
```

A lossy memory layer may return:

```text
authentication has a configurable timeout
```

fidelis returns:

```text
auth tokens expire after 3600 seconds.
The 3600s window is non-configurable in our current contract.
```

The non-configurable qualifier survives. So does every other detail you wrote down.

## What this enables in Codex and Claude Code

Once `fidelis mcp install --client codex` (or the default Claude install) is run, ask your agent:

- *"What did we decide about auth?"*
- *"What failed last time we tried this migration?"*
- *"Which billing constraint was non-configurable?"*
- *"What did I say about Sarah's onboarding flow?"*

The MCP `fidelis_recall` tool gives the agent the original passages before it composes an answer, not paraphrased summaries. The answer can stay grounded in what you wrote, with the qualifiers intact.

> **fidelis retrieves memory without an LLM. Your agent still uses its normal LLM to answer using the retrieved context.** "Zero-LLM" applies to the memory hot path, not to your agent.

## Use cases & ROI

Three concrete reasons teams pick fidelis over hosted memory:

- **Cost reduction.** Stop paying for redundant context-window tokens on every turn. Memory lives on disk; the agent pulls only what's relevant per query. At a few thousand calls/day the math against per-query memory APIs adds up fast.
- **Local data boundary.** The default zero-LLM path keeps notes and retrieval on the local machine, reducing third-party processor exposure. This architecture does not by itself confer SOC 2 or HIPAA compliance.
- **Team context.** Agents that remember historical decisions, naming conventions, failed migrations, and the *qualifiers* on those decisions. The non-configurable detail you wrote down two months ago surfaces when relevant, in the founder's voice, not paraphrased.

## How it fits

The diagram is at the top. Codex and Claude Code are the fastest paths to value. The retrieval engine is agent-agnostic - pair it with any LLM client. Codex registration uses its supported `codex mcp` CLI, and the resulting server configuration is shared by the Codex desktop app, CLI, and IDE extension on that host.

## Benchmarks

LongMemEval-S, 470 questions, public benchmark.

| Metric | Value |
|---|---|
| Retrieval R@1 | **83.2%** |
| Retrieval R@5 | **98.3%** |
| End-to-end QA accuracy | **73.0%**, Wilson 95% CI [68.7%, 77.0%] |
| Cost per query (retrieval) | **$0** (local) |
| Mean retrieval latency | 216 ms (zero-LLM hybrid: BM25 + dense + RRF) |

For context: published Mem0 results on LongMemEval-S are in the ~66–70% end-to-end QA range; Zep is 71.2%; Supermemory is 81.6%; full GPT-4o on raw context (no memory system) is 60.2%. fidelis reaches 73.0% with no LLM in the default retrieval path.

Raw evidence: [retrieval aggregate](bench/runs/runP-v35/aggregate.json) ·
[end-to-end QA summary](experiments/zeroLLM-FLAGSHIP-evidence/SUMMARY.json)

The QA tier wraps your existing LLM with a 140–180-token system prompt - the Fidelis Scaffold. See [`docs/scaffold.md`](docs/scaffold.md).

## Verify the zero-LLM claim yourself

```bash
# Unset any LLM API keys for this shell
unset OPENAI_API_KEY ANTHROPIC_API_KEY DASHSCOPE_API_KEY

# Optional: drop your network. Ollama runs on 127.0.0.1:11434 (loopback).

# `recall-hybrid` is the explicit-tier command. zero_llm is the default.
fidelis recall-hybrid "what did the user say about Sarah" --tier zero_llm
tail ~/.fidelis/server.log
```

The default `zero_llm` tier never makes an outbound LLM call. Optional `--tier filter` and `--tier flagship` modes do call an LLM, but only to select integer pointers - the server dereferences those pointers to the original stored text. The LLM cannot rephrase memory content.

### Context-sensitive orientation (MCP)

The bundled MCP server also exposes `fidelis_orient`. It recognizes when a
turn invokes prior work—even when it is a statement such as “I need to
remember our Fidelis work”—and selects a bounded evidence lane for identity,
maintenance, conceptual reuse, comparison, decisions, historical state, or
current state. The returned orientation is a derived index; retrieved records
remain verbatim evidence with their existing IDs and metadata. Unrelated turns
explicitly abstain without calling the memory server.

## Requirements

- macOS or Linux (Windows not yet supported)
- Python 3.10+
- [Ollama](https://ollama.com) running locally with `nomic-embed-text` pulled (~280 MB):

  ```bash
  brew install ollama && ollama serve &
  ollama pull nomic-embed-text   # ~280 MB, one-time
  ```

The full init-to-first-recall cycle is under 60 seconds once Ollama is up. No memory API keys required.

## Quick reference

```bash
fidelis recall "what did the user say about Sarah"
fidelis query  "Sarah" --limit 5
fidelis add    "raw text to extract into memories"
fidelis health
fidelis seed   ~/memory/   ~/notes/
```

`fidelis add` normally stores facts produced by the configured extraction
model. If extraction returns no facts, Fidelis preserves the original input
verbatim instead of silently losing it. The command still exits 0 because the
write succeeded, but stdout reports a stable degraded status:

```text
status=stored degraded=verbatim-fallback-empty-extraction id=<uuid> count=1
```

Automation that requires successful extraction must inspect `degraded`; exit 0
means the memory was stored, not necessarily that extraction succeeded. Because
mem0 does not distinguish a swallowed extractor failure from a legitimate
zero-fact result, the fallback intentionally favors durability.

Python helper for direct integration:

```python
from fidelis.augment import augment
from anthropic import Anthropic

client = Anthropic()
answer = augment(
    question="What did I say about Sarah?",
    qtype="single-session-user",
    llm_call=lambda system, user: client.messages.create(
        model="claude-haiku-4-5",  # any current Claude Messages model works
        system=system,
        messages=[{"role": "user", "content": user}],
        max_tokens=512,
    ).content[0].text,
)
```

## What's running on your machine

After `fidelis init`:

- **Service:** `fidelis-server` runs at `http://127.0.0.1:19420` under your OS service manager (launchd on macOS, systemd on Linux). Auto-starts on boot. Logs at `~/.fidelis/server.log`.
- **Storage:** Chroma + SQLite at `~/.cogito/` (the directory name is preserved from the project's pre-rename codename for v0.0.x compatibility - it will move to `~/.fidelis/` in a later major bump). No data leaves your machine in the default zero-LLM path.
- **MCP:** after installing for your selected client, Codex or Claude Code sees four tools: `fidelis_recall`, `fidelis_query`, `fidelis_health`, and `fidelis_orient`.

To stop: `fidelis init --uninstall`. To wipe: `rm -rf ~/.cogito ~/.fidelis`.

## Known limitations (v0.0.95)

- **Pre-release.** Python function names and CLI commands may change. Pin the version if you build on it.
- **Best on macOS Sequoia / Ubuntu 24.04 LTS.** Other OSes likely work but aren't gate-tested.
- **Direct server launches disable mem0 telemetry by default.** This matches
  the service installed by `fidelis init` and avoids telemetry exit handlers
  delaying graceful shutdown. An explicit `MEM0_TELEMETRY=True` still opts in.
  For the same boundary across Chroma, set `ANONYMIZED_TELEMETRY=False` and
  `CHROMA_TELEMETRY_DISABLED=True` before a direct launch; `fidelis init`
  includes all three settings automatically.
- **Temporal-reasoning and preference questions are the weakest qtypes** in the QA scaffold (TR ~58%, Pref ~37% on the full eval). Single-session and knowledge-update qtypes are strong (95–100%).
- **The optional LLM tier ("flagship" mode) currently escalates ~80% of queries instead of the intended ~10%** - an 8× cost miss we're transparent about. The default zero-LLM tier is unaffected.
- **qwen3.5:9b in thinking mode does not reliably follow the literal hedge instruction** in the Fidelis Scaffold. Use Claude, an OpenAI-format API, or non-thinking-mode local models for reliable hedging.

## What this turns into over time

Day 1: drop notes into `~/notes`, run the four commands.
Day 2: ask your agent about yesterday's decision - the answer cites your original passage.
Day 7: your agent starts carrying project context across sessions; you stop re-explaining.

Useful for solo builders today; relevant for teams that need memory to stay local tomorrow.

## Fidelis Memory for teams

fidelis is open-source under MIT and free for any use, including commercial. If your team has deployment requirements that the OSS path does not yet cover (centralized memory, multi-namespace isolation, custom authentication), write to **founders@hermes-labs.ai**.

## For technical users

- [`docs/full-reference.md`](docs/full-reference.md) - full architecture, hybrid recall tiers, local server endpoints, troubleshooting
- [`docs/scaffold.md`](docs/scaffold.md) - Fidelis Scaffold contract + drift-detection markers
- [`experiments/zeroLLM-FLAGSHIP-evidence/`](experiments/zeroLLM-FLAGSHIP-evidence/) - raw eval JSONs + machine-readable SUMMARY (per-qtype breakdowns, Wilson CI, F1/F1B baselines)

## License

MIT. Built by Hermes Labs (Roli Bosch). Issues + PRs welcome.

---

## About Hermes Labs

Hermes Labs develops open-source reliability, evaluation, memory, and
containment tools for AI agents. Fidelis is its local-first memory project.
Other public software is listed at
[github.com/hermes-labs-ai](https://github.com/hermes-labs-ai), with research
artifacts published separately on [Zenodo](https://zenodo.org).

For enterprise deployments and AI-reliability engagements: [roli@hermes-labs.ai](mailto:roli@hermes-labs.ai) · [hermes-labs.ai](https://hermes-labs.ai)

On naming. Hermes Labs is named for Hermes, the Greek messenger god - patron of communication and interpretation, the herald who carries meaning between worlds. The thread to the work: hermeneutics, the theory of interpretation that takes its name from Hermes, is the philosophical anchor for an AI reliability engineering studio whose substrate is linguistic. Not affiliated with NousResearch's Hermes LLM line or their hermes-agent framework - different companies, different work.

Founder: Rolando (Roli) Bosch.
Site: [hermes-labs.ai](https://hermes-labs.ai)
Citation: Bosch, R. (2026). Hermes Labs: AI reliability infrastructure for autonomous agents. https://hermes-labs.ai

Quantitative source for the Fidelis claims above: the 470-question
LongMemEval-S aggregate and Wilson interval in
[`experiments/zeroLLM-FLAGSHIP-evidence/`](experiments/zeroLLM-FLAGSHIP-evidence/),
evaluated 2026-04-24.
