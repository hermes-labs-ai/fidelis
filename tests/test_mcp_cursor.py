"""Cursor's native MCP config round-trip preserves unrelated servers."""

import json
from types import SimpleNamespace

from fidelis import mcp_cmd


def _args(path, force=False):
    return SimpleNamespace(client="cursor", settings=str(path), scope=None, force=force)


def test_install_is_idempotent_and_uninstall_preserves_other_servers(tmp_path):
    path = tmp_path / "mcp.json"
    other = {"type": "stdio", "command": "other-server", "args": []}
    path.write_text(json.dumps({"mcpServers": {"other": other}, "setting": True}))

    assert mcp_cmd.cmd_mcp_install(_args(path)) == 0
    installed = json.loads(path.read_text())
    assert installed["mcpServers"]["other"] == other
    assert installed["mcpServers"]["fidelis"] == mcp_cmd._cursor_entry()
    assert installed["setting"] is True
    assert mcp_cmd.cmd_mcp_install(_args(path)) == 0

    assert mcp_cmd.cmd_mcp_uninstall(_args(path)) == 0
    assert json.loads(path.read_text()) == {"mcpServers": {"other": other}, "setting": True}


def test_collision_and_foreign_removal_require_force(tmp_path):
    path = tmp_path / "mcp.json"
    original = {"mcpServers": {"fidelis": {"type": "stdio", "command": "someone-else"}}}
    path.write_text(json.dumps(original))

    assert mcp_cmd.cmd_mcp_install(_args(path)) == 1
    assert mcp_cmd.cmd_mcp_uninstall(_args(path)) == 1
    assert json.loads(path.read_text()) == original


def test_invalid_config_is_not_replaced(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{not-json")
    assert mcp_cmd.cmd_mcp_install(_args(path)) == 1
    assert path.read_text() == "{not-json"
