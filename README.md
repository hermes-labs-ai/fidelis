# fidelis

## Zero-LLM memory for Claude Code and AI agents.

**73.0% end-to-end QA on LongMemEval-S. 83.2% R@1 retrieval. $0/query. No LLM in the default retrieval path.**

Stop re-explaining context to your agent. fidelis returns your original notes verbatim, local-first, fast, about 60 seconds to install. Your agent already calls an LLM to think; it should not need another one just to remember. Designed for developers. Default zero-LLM retrieval path runs fully local with no outbound network calls, which simplifies SOC2 / HIPAA scoping for the memory layer.

[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status: pre-release](https://img.shields.io/badge/status-pre--release-orange)](#known-limitations)
[![Tests: 486 passing](https://img.shields.io/badge/tests-486%20passing-brightgreen)](tests/)
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
Claude Code / your agent
```

What fidelis is:

- **fast** - ~216 ms local retrieval (full benchmark mean; vector-only path is faster)
- **cheap** - $0/query retrieval cost
- **private** - local memory store by default
- **faithful** - original stored passages returned, not paraphrases
- **proven** - benchmarked on LongMemEval-S (470 questions, public benchmark), with raw evidence in [`experiments/zeroLLM-FLAGSHIP-evidence/`](experiments/zeroLLM-FLAGSHIP-evidence/)
- **installable** - Claude Code via MCP in about 60 seconds

---

## Quickstart

```bash
# 0. one-time: Ollama + the local embedder (~280 MB)
brew install ollama && ollama serve &
ollama pull nomic-embed-text

# 1. install + run
pip install fidelis
fidelis init                  # background service (launchd / systemd)
fidelis watch ~/notes         # auto-ingests markdown
fidelis mcp install           # wires Claude Code
# Restart Claude Code. Memory is on.
```

Linux users swap `brew install ollama` for the equivalent install from [ollama.com](https://ollama.com). [See Requirements](#requirements).

v0.0.9 - pre-release.

## What you notice immediately

After the four commands above, the next time you open Claude Code:

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

## What this enables in Claude Code

Once `fidelis mcp install` is run, ask your agent:

- *"What did we decide about auth?"*
- *"What failed last time we tried this migration?"*
- *"Which billing constraint was non-configurable?"*
- *"What did I say about Sarah's onboarding flow?"*

The MCP `fidelis_recall` tool fires before Claude composes its answer. Claude sees the original passages, not paraphrased summaries. The answer is grounded in what you wrote, with the qualifiers intact.

> **fidelis retrieves memory without an LLM. Your agent still uses its normal LLM to answer using the retrieved context.** "Zero-LLM" applies to the memory hot path, not to your agent.

## Use cases & ROI

Three concrete reasons teams pick fidelis over hosted memory:

- **Cost reduction.** Stop paying for redundant context-window tokens on every turn. Memory lives on disk; the agent pulls only what's relevant per query. At a few thousand calls/day the math against per-query memory APIs adds up fast.
- **Security & compliance.** Zero data egress in the default zero-LLM path simplifies SOC2 / HIPAA scoping for the agent-memory layer - your notes never leave the box, so the memory store falls outside any third-party data-processor agreement.
- **Team context.** Agents that remember historical decisions, naming conventions, failed migrations, and the *qualifiers* on those decisions. The non-configurable detail you wrote down two months ago surfaces when relevant, in the founder's voice, not paraphrased.

## How it fits

The diagram is at the top. Claude Code is the fastest path to value. The retrieval engine is agent-agnostic - pair it with any LLM client.

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

Raw evidence: [`bench/runs/zeroLLM-full-20260424/aggregate.json`](bench/runs/zeroLLM-full-20260424/aggregate.json) · [`experiments/zeroLLM-FLAGSHIP-evidence/SUMMARY.json`](experiments/zeroLLM-FLAGSHIP-evidence/SUMMARY.json)

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
- **MCP:** if you ran `fidelis mcp install`, Claude Code sees three tools: `fidelis_recall`, `fidelis_query`, `fidelis_health`.

To stop: `fidelis init --uninstall`. To wipe: `rm -rf ~/.cogito ~/.fidelis`.

## Known limitations (v0.0.9 honest list)

- **Pre-release.** Python function names and CLI commands may change. Pin the version if you build on it.
- **Best on macOS Sequoia / Ubuntu 24.04 LTS.** Other OSes likely work but aren't gate-tested.
- **Temporal-reasoning and preference questions are the weakest qtypes** in the QA scaffold (TR ~58%, Pref ~37% on the full eval). Single-session and knowledge-update qtypes are strong (95–100%).
- **The optional LLM tier ("flagship" mode) currently escalates ~80% of queries instead of the intended ~10%** - an 8× cost miss we're transparent about. The default zero-LLM tier is unaffected.
- **qwen3.5:9b in thinking mode does not reliably follow the literal hedge instruction** in the Fidelis Scaffold. Use Claude, an OpenAI-format API, or non-thinking-mode local models for reliable hedging.

## What this turns into over time

Day 1: drop notes into `~/notes`, run the four commands.
Day 2: ask Claude Code about yesterday's decision - the answer cites your original passage.
Day 7: your agent starts carrying project context across sessions; you stop re-explaining.

Useful for solo builders today; relevant for teams that need memory to stay local tomorrow.

## fidelis for teams

fidelis is open-source under MIT and free for any use, including commercial. If your team has deployment requirements that the OSS path does not yet cover (centralized memory, multi-namespace isolation, custom authentication), write to **founders@hermes-labs.ai**.

## For technical users

- [`docs/full-reference.md`](docs/full-reference.md) - full architecture, hybrid recall tiers, local server endpoints, troubleshooting
- [`docs/scaffold.md`](docs/scaffold.md) - Fidelis Scaffold contract + drift-detection markers
- [`experiments/zeroLLM-FLAGSHIP-evidence/`](experiments/zeroLLM-FLAGSHIP-evidence/) - raw eval JSONs + machine-readable SUMMARY (per-qtype breakdowns, Wilson CI, F1/F1B baselines)

## License

MIT. Built by Hermes Labs (Roli Bosch). Issues + PRs welcome.

---

## About Hermes Labs

Hermes Labs is building reliability infrastructure for autonomous AI agents - memory, evaluation, observability, and containment. Founded 2025 by Rolando (Roli) Bosch, solo founder, AI-amplified ("cyborg engineering"). Based in the San Francisco Bay Area.

The technical thesis: language sets the capability and intelligence; the model is the ceiling, not the source. Reliability is a question of linguistic infrastructure, not model tuning. Formalized as LPCI (Linguistically Persistent Cognitive Interface) - transfer entropy ≈ 0 in embedding-space proxy, Markov property holds, the substrate is linguistic. The engineering follow-on: when language is the substrate, the engineering is interpretive - recovering meaning across the boundaries between model and user, session and session, training and runtime.

Public technical receipts. The first public open-source release is [fidelis](https://github.com/hermes-labs-ai/fidelis) - zero-LLM agent memory with integer-pointer fidelity. 73.0% end-to-end QA on LongMemEval-S, Wilson 95% CI [68.7%, 77.0%], at $0 per query, fully local. Other Hermes Labs OSS is listed at [github.com/hermes-labs-ai](https://github.com/hermes-labs-ai). Published research at [zenodo.org](https://zenodo.org). The OSS surface is the proof; the commercial work is deployment engagements.

For enterprise deployments and AI-reliability engagements: [rbosch@lpci.ai](mailto:rbosch@lpci.ai) · [lpci.ai](https://lpci.ai)

On naming. Hermes Labs is named for Hermes, the Greek messenger god - patron of communication and interpretation, the herald who carries meaning between worlds. The thread to the work: hermeneutics, the theory of interpretation that takes its name from Hermes, is the philosophical anchor for an AI infrastructure company whose substrate is linguistic. Not affiliated with NousResearch's Hermes LLM line or their hermes-agent framework - different companies, different work.

Founder: Rolando (Roli) Bosch.
Site: [hermes-labs.ai](https://hermes-labs.ai)
Citation: Bosch, R. (2026). Hermes Labs: AI reliability infrastructure for autonomous agents. https://hermes-labs.ai

Quantitative sources for claims above:
- fidelis 73.0% / Wilson 95% CI [68.7%, 77.0%]: see fidelis/README.md "End-to-end QA accuracy" + experiments/zeroLLM-FLAGSHIP-evidence/, 470 questions, eval date 2026-04-24
- LPCI thesis (TE ≈ 0 embedding-space proxy): langquant repo, commit dd918cc (2026-03-28) "LPCI PROVED" + lpci_rigorous.py:507-571
- 24-failure taxonomy: hermes-rubric/calibration/failure-mode-taxonomy.md
