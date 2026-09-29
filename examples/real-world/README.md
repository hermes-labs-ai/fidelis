# Real-world examples

Each folder here is an independently linkable example with an `input*.json`, an
`expected*.json`, and a README that says what job it models, where the material came from,
and what it does not show. Every example is re-verified by
[`tests/test_real_world_examples.py`](../../tests/test_real_world_examples.py) without a
running Fidelis server, without Ollama, and without opening any memory store: the tests call
pure functions or parse files.

| Example | Job it models | Checked with |
|---|---|---|
| [correction-chain-temporal](correction-chain-temporal/) | An old fact is corrected without erasing it; recall shows what is superseded, and `as_of` replays the earlier view | `fidelis.temporal` |
| [retraction-and-ephemera-read-filter](retraction-and-ephemera-read-filter/) | An operator retracts a claim and hides tool-dump noise at read time | `fidelis.supersession` with a temporary pointers file |
| [write-gate-screening](write-gate-screening/) | Decide which strings may be written to the store at all | `fidelis.write_gate.evaluate` |
| [mcp-client-config](mcp-client-config/) | Register the stdio MCP server in a client and check the names it relies on | static JSON, `pyproject.toml`, and AST checks |
| [bench-hardset-hit-at-5](bench-hardset-hit-at-5/) | Read a recorded retrieval-benchmark case and recompute what it measures | JSON parsing only |

Not covered: anything that needs the HTTP service (`/store`, `/query`, `/recall_hybrid`),
Ollama embeddings, or a populated store. The examples do not measure retrieval quality of the
0.3.0rc1 default path.
