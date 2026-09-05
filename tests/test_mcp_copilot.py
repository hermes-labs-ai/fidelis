"""Regression tests for `fidelis mcp install|uninstall --client copilot`.

GitHub Copilot CLI reads MCP servers from ``mcp-config.json`` under
``~/.copilot`` (or ``$COPILOT_HOME``). These tests exercise the documented
file shape without touching the real home directory or requiring the
``copilot`` binary.
"""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from fidelis import cli, mcp_cmd
from fidelis.mcp_cmd import (
    MCP_SERVER_FILE,
    cmd_mcp_install,
    cmd_mcp_uninstall,
    copilot_config_path,
    copilot_server_entry,
)


def _args(settings: str | None, force: bool = False, client: str = "copilot") -> Namespace:
    return Namespace(client=client, settings=settings, force=force)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.stem}.json.bak.*"))


def test_install_writes_documented_stdio_entry(tmp_path, capsys):
    config = tmp_path / "mcp-config.json"
    assert cmd_mcp_install(_args(str(config))) == 0

    data = _read(config)
    entry = data["mcpServers"]["fidelis"]
    assert entry == {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(MCP_SERVER_FILE)],
        "tools": ["*"],
    }
    assert Path(entry["args"][0]).is_file()
    assert entry == copilot_server_entry()
    assert not _backups(config), "fresh install must not create a backup"
    out = capsys.readouterr().out
    assert str(config) in out
    assert "restart Copilot CLI" in out


def test_install_preserves_unrelated_servers_and_backs_up(tmp_path):
    config = tmp_path / "mcp-config.json"
    original = {
        "mcpServers": {
            "github": {"type": "http", "url": "https://api.githubcopilot.com/mcp/", "tools": ["*"]},
        },
        "unrelatedKey": {"keep": True},
    }
    config.write_text(json.dumps(original))

    assert cmd_mcp_install(_args(str(config))) == 0

    data = _read(config)
    assert data["mcpServers"]["github"] == original["mcpServers"]["github"]
    assert data["unrelatedKey"] == {"keep": True}
    assert "fidelis" in data["mcpServers"]
    backups = _backups(config)
    assert len(backups) == 1
    assert _read(backups[0]) == original


def test_install_is_idempotent(tmp_path, capsys):
    config = tmp_path / "mcp-config.json"
    assert cmd_mcp_install(_args(str(config))) == 0
    first = config.read_text()
    assert cmd_mcp_install(_args(str(config))) == 0
    assert config.read_text() == first
    assert not _backups(config), "no-op reinstall must not churn backups"
    assert "nothing to change" in capsys.readouterr().out


def test_install_refuses_foreign_entry_without_force(tmp_path, capsys):
    config = tmp_path / "mcp-config.json"
    foreign = {"type": "stdio", "command": "npx", "args": ["some-other-fidelis"], "tools": ["*"]}
    config.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))

    assert cmd_mcp_install(_args(str(config))) == 1
    assert _read(config)["mcpServers"]["fidelis"] == foreign
    assert not _backups(config)
    assert "refusing to overwrite" in capsys.readouterr().err

    assert cmd_mcp_install(_args(str(config), force=True)) == 0
    assert _read(config)["mcpServers"]["fidelis"] == copilot_server_entry()


def test_install_updates_stale_fidelis_entry_in_place(tmp_path):
    """An older fidelis entry (different interpreter / missing tools) is ours; refresh it."""
    config = tmp_path / "mcp-config.json"
    stale = {"type": "local", "command": "/old/python", "args": [str(MCP_SERVER_FILE)]}
    config.write_text(json.dumps({"mcpServers": {"fidelis": stale}}))

    assert cmd_mcp_install(_args(str(config))) == 0
    assert _read(config)["mcpServers"]["fidelis"] == copilot_server_entry()
    assert len(_backups(config)) == 1


def test_install_rejects_invalid_json_without_writing(tmp_path, capsys):
    config = tmp_path / "mcp-config.json"
    config.write_text("{not json")
    assert cmd_mcp_install(_args(str(config))) == 1
    assert config.read_text() == "{not json"
    assert "not valid JSON" in capsys.readouterr().err


def test_install_rejects_non_object_mcp_servers(tmp_path, capsys):
    config = tmp_path / "mcp-config.json"
    config.write_text(json.dumps({"mcpServers": []}))
    assert cmd_mcp_install(_args(str(config))) == 1
    assert "not a JSON object" in capsys.readouterr().err


def test_uninstall_removes_only_fidelis(tmp_path):
    config = tmp_path / "mcp-config.json"
    assert cmd_mcp_install(_args(str(config))) == 0
    data = _read(config)
    data["mcpServers"]["github"] = {"type": "http", "url": "https://api.githubcopilot.com/mcp/"}
    config.write_text(json.dumps(data))

    assert cmd_mcp_uninstall(_args(str(config))) == 0
    after = _read(config)
    assert "fidelis" not in after["mcpServers"]
    assert after["mcpServers"]["github"]["type"] == "http"
    assert len(_backups(config)) == 1


def test_uninstall_refuses_foreign_entry(tmp_path, capsys):
    config = tmp_path / "mcp-config.json"
    foreign = {"type": "stdio", "command": "npx", "args": ["other"]}
    config.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))
    assert cmd_mcp_uninstall(_args(str(config))) == 1
    assert _read(config)["mcpServers"]["fidelis"] == foreign
    assert "refusing to remove" in capsys.readouterr().err


def test_uninstall_without_config_or_entry_is_a_noop(tmp_path, capsys):
    config = tmp_path / "mcp-config.json"
    assert cmd_mcp_uninstall(_args(str(config))) == 0
    assert "nothing to uninstall" in capsys.readouterr().out

    config.write_text(json.dumps({"mcpServers": {}}))
    assert cmd_mcp_uninstall(_args(str(config))) == 0
    assert _read(config) == {"mcpServers": {}}


def test_default_path_honors_copilot_home(tmp_path, monkeypatch):
    monkeypatch.delenv("COPILOT_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    assert copilot_config_path(None) == tmp_path / "home" / ".copilot" / "mcp-config.json"

    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "custom"))
    assert copilot_config_path(None) == tmp_path / "custom" / "mcp-config.json"
    assert copilot_config_path(str(tmp_path / "explicit.json")) == tmp_path / "explicit.json"


def test_install_under_copilot_home_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "copilot-home"))
    assert cmd_mcp_install(_args(None)) == 0
    config = tmp_path / "copilot-home" / "mcp-config.json"
    assert config.is_file()
    assert _read(config)["mcpServers"]["fidelis"] == copilot_server_entry()
    assert cmd_mcp_uninstall(_args(None)) == 0
    assert "fidelis" not in _read(config)["mcpServers"]


def test_install_never_touches_claude_or_codex_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_cmd, "DEFAULT_SETTINGS", tmp_path / "claude-settings.json")
    monkeypatch.setattr(mcp_cmd, "_codex_cli", lambda: pytest.fail("codex CLI must not be invoked"))
    config = tmp_path / "mcp-config.json"
    assert cmd_mcp_install(_args(str(config))) == 0
    assert not (tmp_path / "claude-settings.json").exists()


def test_cli_accepts_copilot_client(tmp_path, monkeypatch):
    config = tmp_path / "mcp-config.json"
    monkeypatch.setattr(sys, "argv", ["fidelis", "mcp", "install", "--client", "copilot", "--settings", str(config)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert _read(config)["mcpServers"]["fidelis"] == copilot_server_entry()

    monkeypatch.setattr(sys, "argv", ["fidelis", "mcp", "uninstall", "--client", "copilot", "--settings", str(config)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert "fidelis" not in _read(config)["mcpServers"]
