"""MCP stdio smoke test with no Ollama reachable.

Glama (and any MCP registry inspector) builds the container image and talks
`initialize` then `tools/list` to the container's CMD over stdio — no daemon,
no running fidelis-server, and (per this test) no reachable Ollama either.
This drives the shipped mcp_server.py exactly the way such an inspector does,
using OLLAMA_URL=http://127.0.0.1:1 (a port nothing can ever be listening on)
to guarantee the embedder backend is unreachable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fidelis import __version__

SRC = str(Path(__file__).resolve().parents[1] / "src")
SERVER = Path(SRC) / "fidelis" / "mcp_server.py"

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "glama-build-inspector", "version": "0"},
    },
}
TOOLS_LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


def _converse(requests: list[dict], timeout: float = 20.0) -> list[dict]:
    """Run the shipped MCP server as a subprocess with no Ollama reachable."""
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": SRC,
        # Unreachable on purpose: guarantees no Ollama, no fidelis-server.
        "OLLAMA_URL": "http://127.0.0.1:1",
        "FIDELIS_PORT": "0",
    }
    proc = subprocess.run(
        [sys.executable, str(SERVER)],
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


def test_initialize_then_tools_list_without_ollama():
    responses = _converse([INITIALIZE, TOOLS_LIST])

    assert len(responses) == 2
    init, tools = responses

    assert init["id"] == 1
    assert init["result"]["protocolVersion"] == "2024-11-05"
    assert init["result"]["capabilities"] == {"tools": {}}
    assert init["result"]["serverInfo"] == {"name": "fidelis", "version": __version__}

    assert tools["id"] == 2
    tool_list = tools["result"]["tools"]
    assert isinstance(tool_list, list)
    assert len(tool_list) > 0
    for tool in tool_list:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
    assert {t["name"] for t in tool_list} == {
        "fidelis_recall", "fidelis_query", "fidelis_health", "fidelis_orient",
    }


def test_exits_zero_within_budget_with_no_ollama():
    # A hard sub-20s budget as required for the registry build check; the
    # subprocess timeout above already enforces it, this just asserts on
    # the return path explicitly rather than relying on a timeout exception.
    responses = _converse([INITIALIZE, TOOLS_LIST], timeout=20.0)
    assert len(responses) == 2
