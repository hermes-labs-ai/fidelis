<!-- mcp-name: io.github.hermes-labs-ai/fidelis-memory -->

<div align="center">

# Fidelis Memory

<img src="assets/fidelis-memory-artwork.png" width="420" alt="Fidelis Memory mascot" />

**Your agent can forget a summary. Fidelis brings back the note.**

Local memory for Codex, Claude Code, and other agents: save text across sessions and retrieve the original passage when it matters.

by [Hermes Labs](https://hermes-labs.ai)

[PyPI](https://pypi.org/project/fidelis-memory/) · [Connect an agent](#connect-an-agent) · [Technical reference](docs/full-reference.md)

</div>

A project summary might say “billing migration rolled back.” The original note says **why** it failed and **do not retry until the idempotency fix is verified**. When your next session depends on that condition, the missing sentence matters.

Fidelis stores your Markdown and text notes and retrieves their stored wording. Its default retrieval does not ask a generative model to rewrite your memory.

## Try it with one note

This path supports Python 3.10+, macOS or Ubuntu, and a local [Ollama](https://docs.ollama.com/quickstart) service. Start Ollama if needed (`ollama serve` in a separate terminal). No model API key is required. The current release is **0.3.0rc1**.

```bash
ollama pull nomic-embed-text
python3 -m venv ~/.venvs/fidelis
source ~/.venvs/fidelis/bin/activate
python3 -m pip install "fidelis-memory[hybrid]==0.3.0rc1"
fidelis init

demo_dir=$(mktemp -d)
cat > "$demo_dir/atlas.md" <<'NOTE'
Atlas billing migration:
Duplicate charges appeared in staging. Rolled back.
Do not retry until the idempotency fix is verified.
NOTE

fidelis watch "$demo_dir" --once
fidelis recall-hybrid "Atlas billing migration retry condition" --tier zero_llm
```

Look for both the rollback and the retry condition in the retrieved text. This is retrieval of a saved note, not a generated answer. For your own files, replace `"$demo_dir"` with a notes directory. If retrieval fails, run `fidelis health` and confirm Ollama has `nomic-embed-text` available. Keep the virtual environment after `fidelis init`; the background service uses it.

## Connect an agent

Once the note is retrievable locally, install the MCP connection for your client:

```bash
fidelis mcp install --client codex
# Or, for Claude Code:
fidelis mcp install
```

Restart the client and try: **“Use Fidelis to find my Atlas billing migration note. What must happen before we retry? Quote the relevant text.”**

The client gets tools to recall, store, correct, fetch, and inspect memory. It decides when to call them; installing the connection does not make every turn recall automatically. [Other clients and uninstall commands](docs/full-reference.md) include Cursor, Gemini CLI, Copilot CLI, and OpenClaw. A separate [Pi extension](extensions/) provides prompt-time recall.

## What Fidelis preserves

- **Original wording.** `watch`, `store`, and other verbatim write paths keep the text you supplied, so retrieval can return the condition that a summary might omit.
- **Local retrieval.** Fast vector search is the default MCP recall path. Explicit hybrid search combines keyword and vector results without a generative LLM at its `zero_llm` tier; local embeddings still require Ollama.
- **Corrections over time.** A correction can supersede a prior record without erasing it. Recall can show validity and historical status instead of silently replacing the old note.

Fidelis also has optional extraction and model-assisted routes. Those can transform or filter inputs; use the verbatim path when the original wording is the point. Retrieved text can still be wrong or out of date, and your agent's model provider may see it when answering.

## Current scope

Fidelis 0.3.0rc1 is a single-machine pre-release, not a hosted team memory service. The [LongMemEval-S retrieval run](bench/DEFAULT-RETRIEVAL-METHODOLOGY.md) measured whether the redesigned default path retrieved material across 470 questions; it did not measure answer accuracy. See the [results](bench/results-default-0.3.0rc1.json), [user-fit guide](docs/user-fit.md), and [security policy](SECURITY.md) for deeper detail.

[Full reference](docs/full-reference.md) · [Upgrade notes](docs/releases/0.3.0rc1.md) · [Contribute](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

Apache-2.0. [Hermes Labs](https://hermes-labs.ai) builds agentic infrastructure for autonomous systems.
