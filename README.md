<!-- mcp-name: io.github.hermes-labs-ai/fidelis-memory -->

<p align="center">
  <img src="https://raw.githubusercontent.com/hermes-labs-ai/fidelis/v0.3.0rc1/assets/fidelis-memory-artwork.jpg" width="420" alt="Fidelis Memory, the golden retriever mascot" />
</p>

# Fidelis Memory

**Agent memory that brings back the source, not another summary.**

Fidelis is a local memory and retrieval service for Codex, Claude Code, and other AI agents. Keep your notes available across sessions and retrieve stored text without generative rewriting.

Fidelis is developed by [Hermes Labs](https://hermes-labs.ai).

Hermes Labs studies failure modes in agent and LLM systems, develops open-source tools that treat language as part of the runtime, and works with teams to remediate reliability failures in production.

A summary can preserve "we tried the migration" while dropping why it failed, what it affected, and what must change before trying again. Fidelis's verbatim ingestion path keeps those details in the stored note instead of requiring a generated fact to replace it.

[![PyPI pre-release](https://img.shields.io/badge/PyPI-0.3.0rc1-blue)](https://pypi.org/project/fidelis-memory/0.3.0rc1/)
[![CI](https://github.com/hermes-labs-ai/fidelis/actions/workflows/ci.yml/badge.svg)](https://github.com/hermes-labs-ai/fidelis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/fidelis-memory)](https://pypi.org/project/fidelis-memory/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

[Quickstart](#quickstart) · [Connect your agent](#connect-your-agent) · [How it works](#how-it-works) · [Benchmarks](#benchmarks) · [Documentation](#documentation)

## Quickstart

You need **Python 3.10+, macOS or Ubuntu, and Ollama running locally**. Ubuntu service installation uses systemd. Install [Ollama](https://docs.ollama.com/quickstart) first; if its server is not running, start `ollama serve` in another terminal. This walkthrough needs no model API key.

### 1. Install and start Fidelis

```bash
ollama pull nomic-embed-text

python3 -m venv ~/.venvs/fidelis
source ~/.venvs/fidelis/bin/activate
python3 -m pip install "fidelis-memory[hybrid]==0.3.0rc1"

fidelis init
```

The `hybrid` extra adds BM25 keyword search. `fidelis init` installs the background memory service using this Python environment, so keep the environment in place. The package is **`fidelis-memory`**; the command is **`fidelis`**.

### 2. Store a note and retrieve it

```bash
demo_dir=$(mktemp -d)
cat > "$demo_dir/atlas.md" <<'NOTE'
Atlas billing migration, 2026-09-20:
Duplicate charges appeared in staging. Rolled back.
Do not retry until the idempotency fix is verified.
NOTE

fidelis watch "$demo_dir" --once
fidelis recall-hybrid "Atlas billing migration retry condition" --tier zero_llm
```

**Success means the returned text includes both the rollback and the retry condition.** The command retrieves stored text; it does not generate an answer. Scores and ordering depend on your store.

For your own notes, run `fidelis watch ~/notes --once`. Omit `--once` to keep watching in a separate terminal. The watcher ingests Markdown and text files, not every conversation in your agent clients.

Trouble retrieving? Run `fidelis health` and check that Ollama is running with `nomic-embed-text` available. A responding health endpoint alone does not prove that ingestion and retrieval work.

## Connect your agent

After the local retrieval works, register Fidelis with the client you use:

| Client | Install command |
| --- | --- |
| Codex | `fidelis mcp install --client codex` |
| Claude Code | `fidelis mcp install` |
| GitHub Copilot CLI | `fidelis mcp install --client copilot` |
| Gemini CLI | `fidelis mcp install --client gemini` |
| OpenClaw | `fidelis mcp install --client openclaw` |

Restart your client, then try:

> Use Fidelis to retrieve my Atlas billing migration note. What must happen before we retry? Quote the relevant text.

Fidelis exposes six MCP tools: `fidelis_recall`, `fidelis_store`, `fidelis_correct`, `fidelis_get`, `fidelis_recent`, and `fidelis_health`. Your agent decides when to call them. Installing the integration does not guarantee automatic recall on every turn. See the [technical reference](docs/full-reference.md) for client prerequisites and configuration.

**MCP update in 0.3.0rc1:** recall, recent results, and correction chains return full stored text; the old silent 300-character previews are removed. Corrections retain superseded records, and recall supports validity dates and historical views. Replace old `fidelis_query` calls with `fidelis_recall` and restart clients to refresh their tool lists. See the [upgrade and rollback notes](docs/releases/0.3.0rc1.md).

## Why keep the source?

Summaries are useful for navigating a long history. They can also leave out information that becomes important to a later question. Once the summary is all that remains, retrieval cannot recover what was discarded.

Fidelis is built for work where you need to revisit the evidence:

- **Decisions and constraints:** recover the rationale, exceptions, and exact conditions in a saved note.
- **Failed approaches:** retrieve what broke and what must change before another attempt.
- **Work across sessions:** make your saved project context accessible to different agent clients on the same machine.

The principle is simple: **use derived representations to find evidence, not to replace it.**

## How it works

```text
Your Markdown or text files
          |
   Verbatim ingestion
          |
   Local memory store
          |
   Retrieve and rank candidates
          |
   Stored text for your agent
```

Default MCP recall uses fast local vector retrieval without a generative LLM. Explicit `mode: "thorough"` selects the hybrid path. The hybrid retrieval path combines keyword search, dense-vector similarity, and reciprocal rank fusion. Its default `zero_llm` tier does not call a generative LLM. Local embeddings are still required.

Optional model-assisted tiers can help select candidates. Their accepted output is a list of candidate numbers. Code resolves those numbers to stored text rather than returning the model's prose as memory.

Fidelis builds on mem0 and ChromaDB for storage and adds its retrieval, fidelity, service, and agent-integration layers.

### The fidelity boundary

The source-preserving paths include `fidelis watch`, `fidelis store`, `fidelis add`, and HTTP `POST /store`. Explicit `fidelis add --extract` and `fidelis seed` use extraction or curation and can transform input before storage. Snapshots are derived summaries, not source evidence.

Fidelity means preserving the text supplied through the verbatim path. It does not prove that the text is true, current, complete, or the original record of an event. Store only a summary and only that summary can be recovered. Full provenance tracking is not a release guarantee.

With local Ollama, the quickstart keeps storage and retrieval local. Your agent may send retrieved text to its model provider when answering. Optional LLM features follow their configured data boundaries.

## Benchmarks

The redesigned default zero-LLM retrieval path completed a fresh LongMemEval-S run of **470 questions on September 21, 2026**, with zero errors. The [results](bench/results-default-0.3.0rc1.json) and [methodology](bench/DEFAULT-RETRIEVAL-METHODOLOGY.md) record the source snapshot, corpus construction, metrics, and limitations.

These are whole-session retrieval measurements, not answer accuracy or a matched comparison with competitors. Historical chunked retrieval and QA scores do not measure this redesign and are not reused as release evidence. Full LLM/QA evaluation is post-release work.

Fast recall remains the default. Optional thorough hybrid retrieval needs further tuning; that work is deferred beyond this pre-release.

## Is Fidelis a fit?

Choose Fidelis when your working context lives in local notes, you want to retrieve their text rather than replace it with synthesized memory, and you can run a local service.

Version 0.3.0rc1 is an early, single-machine pre-release. It is not a hosted team-memory platform. Windows service installation and managed multi-user authorization are not supported contracts. Preserving a past statement also does not make it current: review dates and conflicting records before acting.

See the [user-fit guide](docs/user-fit.md) and [security policy](SECURITY.md) before deploying.

## Documentation

| Need | Start here |
| --- | --- |
| Commands, HTTP API, configuration, and client setup | [Technical reference](docs/full-reference.md) |
| Supported workflows and limitations | [User-fit guide](docs/user-fit.md) |
| Optional guidance for an LLM reading retrieved evidence | [QA scaffold](docs/scaffold.md) |
| Release history | [Changelog](CHANGELOG.md) |
| Contributing or reporting security issues | [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) |

## Contributing

Found a missed passage, an unexpected rewrite, or an installation problem? [Open an issue](https://github.com/hermes-labs-ai/fidelis/issues) with a minimal, redacted example. Retrieval regressions, fidelity tests, and documentation fixes are welcome.

## License

[MIT](LICENSE).
