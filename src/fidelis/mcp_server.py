"""Minimal MCP server bundled with fidelis — exposes recall + health as MCP tools.

Implements the JSON-RPC stdio transport for MCP clients. Relies
on a running fidelis-server (port 19420 by default) for the actual recall.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from fidelis import __version__
from fidelis.context import context_packet, plan_context


def _server_url() -> str:
    port = os.environ.get("FIDELIS_PORT", os.environ.get("COGITO_PORT", "19420"))
    return f"http://127.0.0.1:{port}"


def _http_post(path: str, payload: dict, timeout: float = 30.0) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_server_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": f"fidelis-server unreachable at {_server_url()}: {e}"}


def _http_get(path: str, timeout: float = 5.0) -> dict:
    try:
        with urllib.request.urlopen(f"{_server_url()}{path}", timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        return {"error": f"fidelis-server unreachable at {_server_url()}: {e}"}


TOOLS = [
    {
        "name": "fidelis_recall",
        "description": (
            "Retrieve memories from the local fidelis store. Two-stage recall "
            "with optional LLM filter; zero-LLM by default."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query"},
                "limit": {"type": "integer", "description": "Max results", "default": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fidelis_query",
        "description": "Fast vector-only query over fidelis memories (no filter).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fidelis_health",
        "description": "Check the fidelis server's health and memory count.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "fidelis_orient",
        "description": (
            "Context-sensitive re-entry for prior work. Call when a turn mentions "
            "a known project, decision, earlier work, maintenance, comparison, or "
            "possible reuse—even when the turn is not phrased as a question. "
            "Returns an evidence-bound orientation packet or explicitly abstains."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "utterance": {"type": "string", "description": "Current user turn"},
                "recent_turns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Up to four recent turns for referent resolution",
                },
                "entity": {
                    "type": "string",
                    "description": "Optional exact known project or concept name",
                },
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["utterance"],
        },
    },
]


def _tool_recall(arguments: dict) -> str:
    query = arguments.get("query", "")
    limit = int(arguments.get("limit", 20))
    res = _http_post("/recall", {"text": query, "limit": limit})
    if res.get("error"):
        return f"error: {res['error']}"
    memories = res.get("memories", [])
    if not memories:
        return "no memories found."
    lines = [f"{len(memories)} memories:"]
    for i, m in enumerate(memories[:limit], 1):
        text = m.get("text", "")
        score = m.get("score", "")
        lines.append(f"  [{i}] (score {score}) {text[:300]}")
    return "\n".join(lines)


def _tool_query(arguments: dict) -> str:
    query = arguments.get("query", "")
    limit = int(arguments.get("limit", 5))
    res = _http_post("/query", {"text": query, "limit": limit})
    if res.get("error"):
        return f"error: {res['error']}"
    memories = res.get("memories", [])
    if not memories:
        return "no memories found."
    lines = [f"{len(memories)} memories:"]
    for i, m in enumerate(memories[:limit], 1):
        text = m.get("text", "")
        score = m.get("score", "")
        lines.append(f"  [{i}] (score {score}) {text[:300]}")
    return "\n".join(lines)


def _tool_health(arguments: dict) -> str:
    res = _http_get("/health")
    if res.get("error"):
        return f"error: {res['error']}"
    return (
        f"status: {res.get('status')} | memories: {res.get('count')} | "
        f"version: {res.get('version')} | calibrated: {'yes' if res.get('calibrated') else 'no'} | "
        f"snapshot: {'yes' if res.get('snapshot') else 'no'}"
    )


def _tool_orient(arguments: dict) -> str:
    utterance = str(arguments.get("utterance", ""))
    recent_turns = arguments.get("recent_turns", [])
    if not isinstance(recent_turns, list):
        recent_turns = []
    plan = plan_context(
        utterance,
        recent_turns=[str(turn) for turn in recent_turns[-4:]],
        entity_hint=arguments.get("entity"),
    )
    if plan.disposition == "abstain":
        return json.dumps(context_packet(plan, []), ensure_ascii=False)
    limit = max(1, min(int(arguments.get("limit", 5)), 20))
    res = _http_post(
        "/recall_hybrid",
        {"text": plan.retrieval_query, "limit": limit, "tier": "zero_llm"},
    )
    if res.get("error"):
        packet = context_packet(plan, [])
        packet["evidence_status"] = "unavailable"
        packet["error"] = res["error"]
        return json.dumps(packet, ensure_ascii=False)
    return json.dumps(
        context_packet(plan, res.get("memories", [])[:limit]),
        ensure_ascii=False,
    )


TOOL_HANDLERS = {
    "fidelis_recall": _tool_recall,
    "fidelis_query": _tool_query,
    "fidelis_health": _tool_health,
    "fidelis_orient": _tool_orient,
}


def _send(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _handle(req: dict) -> dict | None:
    method = req.get("method", "")
    rid = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": rid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fidelis", "version": __version__},
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})
        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32601, "message": f"unknown tool: {tool_name}"},
            }
        try:
            text = handler(arguments)
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {"content": [{"type": "text", "text": text}]},
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": rid,
                "error": {"code": -32603, "message": str(e)},
            }
    if method == "notifications/initialized":
        return None  # no response needed for notifications
    return {
        "jsonrpc": "2.0",
        "id": rid,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(req)
        if resp is not None:
            _send(resp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
