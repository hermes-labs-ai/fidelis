"""The MCP `ping` request, over the real stdio wire.

Gemini CLI's `gemini mcp list` connects, then sends the standard MCP `ping`
request and reports a server that does not answer it as Disconnected — even
though the connection succeeded and every tool works. Fidelis's server used to
fall through to its unknown-method branch and return -32601, so a healthy
server was listed as down and the README pointed users at that diagnostic.

`ping` is defined by the MCP base protocol (Utilities > Ping): a request with
no params, answered with an empty result. These tests drive the shipped
`mcp_server.py` the way a client does, so they hold on the bytes Gemini reads
rather than on an internal call.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fidelis import __version__
from fidelis.mcp_server import _handle

SRC = str(Path(__file__).resolve().parents[1] / "src")
SERVER = Path(SRC) / "fidelis" / "mcp_server.py"

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "0"},
    },
}


def _converse(requests: list[dict], env_extra: dict | None = None) -> list[dict]:
    """Run the shipped server as a subprocess and read back its stdout lines."""
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": SRC,
        # Port 0 can never be connected to: `ping` must never touch
        # fidelis-server, and this can't flake if some other process happens
        # to be listening on a fixed port.
        "FIDELIS_PORT": "0",
    }
    env.update(env_extra or {})
    proc = subprocess.run(
        [sys.executable, str(SERVER)],
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# the wire response
# --------------------------------------------------------------------------


def test_ping_is_answered_with_an_empty_result_on_the_wire():
    responses = _converse([
        INITIALIZE,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
    ])

    assert len(responses) == 2, "the notification must not draw a response"
    assert responses[0]["id"] == 1
    assert responses[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}


def test_ping_carries_no_error_and_no_params_are_required():
    """MCP sends `ping` with no params at all; some clients send an empty dict."""
    for request in (
        {"jsonrpc": "2.0", "id": 7, "method": "ping"},
        {"jsonrpc": "2.0", "id": 8, "method": "ping", "params": {}},
    ):
        response = _handle(request)
        assert response == {"jsonrpc": "2.0", "id": request["id"], "result": {}}
        assert "error" not in response


def test_ping_answers_without_a_running_fidelis_server():
    """Liveness is a protocol fact. It must not depend on the HTTP backend.

    `fidelis_health` against the same dead port reports the backend down —
    which is what proves the ping answer was not an accident of a live server.
    """
    responses = _converse([
        INITIALIZE,
        {"jsonrpc": "2.0", "id": 2, "method": "ping"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "fidelis_health", "arguments": {}},
        },
    ])

    assert responses[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert "unreachable" in responses[2]["result"]["content"][0]["text"]


def test_repeated_pings_are_each_answered_and_keep_the_session_alive():
    requests = [INITIALIZE]
    requests += [{"jsonrpc": "2.0", "id": i, "method": "ping"} for i in range(2, 6)]
    requests.append({"jsonrpc": "2.0", "id": 6, "method": "tools/list"})

    responses = _converse(requests)

    assert [r["id"] for r in responses] == [1, 2, 3, 4, 5, 6]
    for r in responses[1:5]:
        assert r["result"] == {}
    assert [t["name"] for t in responses[5]["result"]["tools"]] == [
        "fidelis_recall", "fidelis_query", "fidelis_health", "fidelis_orient",
    ]


# --------------------------------------------------------------------------
# nothing else moved
# --------------------------------------------------------------------------


def test_initialize_and_tools_list_are_unchanged():
    responses = _converse([INITIALIZE, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}])

    assert responses[0]["result"] == {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "fidelis", "version": __version__},
    }
    assert len(responses[1]["result"]["tools"]) == 4


@pytest.mark.parametrize("method", [
    "pings", "ping/list", "Ping", "resources/list", "prompts/list", "",
])
def test_other_unknown_methods_still_report_method_not_found(method):
    """The fix adds one method. It must not turn the server permissive."""
    response = _handle({"jsonrpc": "2.0", "id": 9, "method": method})
    assert response["error"]["code"] == -32601
    assert "result" not in response


def test_notifications_still_get_no_response():
    assert _handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
