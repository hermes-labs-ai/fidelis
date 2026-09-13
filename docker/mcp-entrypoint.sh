#!/bin/sh
# Start fidelis-server in the background, then serve MCP over stdio.
# Server output goes to stderr: stdout is the MCP JSON-RPC channel.
set -eu

fidelis-server --host 127.0.0.1 --port "${FIDELIS_PORT:-19420}" 1>&2 &

exec fidelis mcp serve
