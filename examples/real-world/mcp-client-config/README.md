# MCP client configuration

## Job
Register the Fidelis stdio MCP server in a client that reads a JSON `mcpServers` block, and
confirm statically that the names in the block resolve to real entry points and tools.

## Files
- `input-mcp.json`: a copy of the repository's `mcp.json`.
- `input-gemini-extension.json`: the `mcpServers` block of `gemini-extension.json`.
- `expected.json`: what the configuration resolves to.

## How it is verified (no server)
The test parses both JSON files; checks that the launch is `uvx --from fidelis-memory==<the
version in pyproject.toml> fidelis mcp serve`; checks `[project.scripts]` in `pyproject.toml`
for `fidelis` and `fidelis-mcp`; uses `ast` (never `import`) on `src/fidelis/cli.py` to confirm
`mcp` and `serve` subcommands, on `src/fidelis/mcp_server.py` to confirm a `main` function, and on
the same file to read the six tool names. Nothing is executed or connected.

## Provenance
Both JSON inputs are copied from files in this repository. The tool list matches
`docs/full-reference.md`.

## Limits
- This proves the names are consistent, not that a client will accept the file, that `uvx`
  is installed, or that the server starts. It never starts the server.
- The version pin follows `pyproject.toml`. After a release bump, refresh the copies here.
- The entry has no `env` block. The service address comes from `FIDELIS_PORT` (default 19420)
  and `COGITO_USER_ID` is a local storage namespace, not authentication.
- `fidelis mcp install --client <name>` writes different entries for Copilot, Cursor, Gemini
  and OpenClaw (interpreter path plus `mcp_server.py`); those are machine-specific and not shown.
