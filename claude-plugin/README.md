# fidelis plugin root

Portable plugin root for the Hermes Labs catalog. It carries two things and no
product code:

- `.mcp.json` — the fidelis MCP connection, launched from the released
  `fidelis-memory` distribution on PyPI via `uvx`. No repository checkout and
  no SSH access are needed to install or run it.
- `skills/fidelis-memory/SKILL.md` — the workflow skill: when to recall before
  answering, and what to write back after a decision.

## What the host gets

| Capability | Surface | Provided by |
|---|---|---|
| `mcp` | `fidelis_orient`, `fidelis_recall`, `fidelis_query`, `fidelis_health` | `.mcp.json` |
| `skill` | `fidelis-memory` | `skills/fidelis-memory/SKILL.md` |

These are separate claims. A loaded skill does not prove the MCP server
connected, and a connected MCP server does not prove a store is populated —
`fidelis_health` is what distinguishes those states at runtime.

## Prerequisites

The MCP server is a thin stdio client. It reads from a local `fidelis-server`
over HTTP on `127.0.0.1:19420` and holds no store of its own. Start one first:

```bash
uvx --from "fidelis-memory==0.1.0" fidelis init     # install the background service
# or, in the foreground:
uvx --from "fidelis-memory==0.1.0" fidelis server
```

`uvx` comes from [uv](https://docs.astral.sh/uv/). Without a running server the
MCP tools answer `fidelis-server unreachable at http://127.0.0.1:19420`, which
is a distinct condition from an empty store.

## Verify the connection

```bash
printf '%s\n%s\n%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' \
  | uvx --from "fidelis-memory==0.1.0" fidelis mcp serve
```

The second response lists the four tools above.

## Configuration

`FIDELIS_PORT` in the `.mcp.json` `env` block points the MCP client at a
non-default server port. The server process reads `COGITO_PORT`; set both when
you move off 19420.

## Version binding

`plugin.json` `version`, the `fidelis-memory` pin in `.mcp.json`, and the
package version in the repository's `pyproject.toml` and `server.json` are
asserted equal by `tests/test_claude_plugin_root.py`. Bump them together.
