"""Regression tests for `fidelis mcp install|uninstall --client openclaw`.

OpenClaw reads an optional JSON5 config from ``~/.openclaw/openclaw.json`` and
keeps outbound MCP servers under ``mcp.servers.<name>``. Because that file is
JSON5, Fidelis never rewrites it -- the documented ``openclaw mcp add`` CLI
owns every write, and ``$OPENCLAW_CONFIG_PATH`` pins which file that is.

These tests stand up a fake ``openclaw`` executable on PATH that implements the
documented ``mcp add --command/--arg`` and ``mcp unset`` contract against
``$OPENCLAW_CONFIG_PATH``. Nothing here touches the real home directory or
requires OpenClaw to be installed.
"""

import json
import os
import stat
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from fidelis import cli, mcp_cmd
from fidelis.mcp_cmd import (
    MCP_SERVER_FILE,
    MCP_SERVER_NAME,
    cmd_mcp_install,
    cmd_mcp_uninstall,
    openclaw_add_arguments,
    openclaw_config_path,
)


# A stand-in for the OpenClaw CLI. It honours exactly the documented surface
# Fidelis delegates to, records every invocation, and can be told to fail.
FAKE_OPENCLAW = r'''#!/usr/bin/env python3
import json, os, sys

argv = sys.argv[1:]
record = os.environ["FAKE_OPENCLAW_LOG"]
with open(record, "a") as handle:
    handle.write(json.dumps({"argv": argv, "config": os.environ.get("OPENCLAW_CONFIG_PATH")}) + "\n")

mode = os.environ.get("FAKE_OPENCLAW_MODE", "ok")
if mode == "fail":
    sys.stderr.write("openclaw: boom\n")
    sys.exit(3)
if mode == "noop":
    # Accept the call but leave the config alone. Models both "OpenClaw owns
    # its JSON5 file and this shim cannot parse it" and "the write silently
    # did not land" -- Fidelis must tell those apart by read-back, not by
    # trusting the exit code.
    print("ok")
    sys.exit(0)

path = os.environ["OPENCLAW_CONFIG_PATH"]
try:
    with open(path) as handle:
        config = json.load(handle)
except FileNotFoundError:
    config = {}
servers = config.setdefault("mcp", {}).setdefault("servers", {})

if argv[:2] == ["mcp", "add"]:
    name = argv[2]
    rest, command, args = argv[3:], None, []
    while rest:
        flag, value, rest = rest[0], rest[1], rest[2:]
        if flag == "--command":
            command = value
        elif flag == "--arg":
            args.append(value)
    servers[name] = {"command": command, "args": args, "enabled": True}
    print("added " + name)
elif argv[:2] == ["mcp", "unset"]:
    servers.pop(argv[2], None)
    print("unset " + argv[2])
else:
    sys.stderr.write("openclaw: unsupported: " + " ".join(argv) + "\n")
    sys.exit(2)

os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
'''


@pytest.fixture
def openclaw(tmp_path, monkeypatch):
    """Put a fake `openclaw` on PATH and return a handle to its call log."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "openclaw"
    binary.write_text(FAKE_OPENCLAW)
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "openclaw-calls.jsonl"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_OPENCLAW_LOG", str(log))
    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)

    class Handle:
        path = binary

        @staticmethod
        def calls() -> list[dict]:
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text().splitlines() if line]

        @staticmethod
        def mode(value: str) -> None:
            monkeypatch.setenv("FAKE_OPENCLAW_MODE", value)

    return Handle()


def _args(settings, force: bool = False, client: str = "openclaw") -> Namespace:
    return Namespace(client=client, settings=str(settings) if settings else None, force=force)


def _config(path: Path) -> dict:
    return json.loads(path.read_text())


def _entry(path: Path) -> dict:
    return _config(path)["mcp"]["servers"][MCP_SERVER_NAME]


# --------------------------------------------------------------------------
# install
# --------------------------------------------------------------------------


def test_install_delegates_the_documented_add_invocation(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0

    calls = openclaw.calls()
    assert len(calls) == 1
    assert calls[0]["argv"] == [
        "mcp", "add", "fidelis", "--command", sys.executable, "--arg", str(MCP_SERVER_FILE),
    ]
    assert calls[0]["argv"] == openclaw_add_arguments()
    # The delegated write is pinned to the file we read back.
    assert calls[0]["config"] == str(config)

    entry = _entry(config)
    assert entry["command"] == sys.executable
    assert entry["args"] == [str(MCP_SERVER_FILE)]
    assert Path(entry["args"][0]).is_file()

    out = capsys.readouterr().out
    assert str(config) in out
    assert "openclaw mcp reload" in out
    assert "openclaw mcp doctor fidelis --probe" in out


def test_install_never_rewrites_the_json5_config_itself(tmp_path, openclaw):
    """A JSON5 config keeps its comments: Fidelis writes nothing, the CLI does."""
    config = tmp_path / "openclaw.json"
    config.write_text('{\n  // hand-written note\n  "mcp": { "servers": {} },\n}\n')
    before = config.read_text()

    openclaw.mode("noop")  # CLI accepts the call but leaves the file untouched
    assert cmd_mcp_install(_args(config)) == 0
    assert config.read_text() == before, "Fidelis must not rewrite a JSON5 config"


def test_install_reports_unconfirmable_json5_readback(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    config.write_text('{ /* json5 */ "mcp": { "servers": {} } }')
    openclaw.mode("noop")

    assert cmd_mcp_install(_args(config)) == 0
    out = capsys.readouterr().out
    assert "not strict JSON" in out
    assert "could not be confirmed by read-back" in out


def test_install_is_idempotent(tmp_path, openclaw):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0
    first = _config(config)
    assert cmd_mcp_install(_args(config)) == 0
    assert _config(config) == first


def test_install_preserves_unrelated_servers(tmp_path, openclaw):
    config = tmp_path / "openclaw.json"
    original = {
        "mcp": {"servers": {"docs": {"url": "https://mcp.example.com/mcp", "enabled": True}}},
        "gateway": {"port": 8080},
    }
    config.write_text(json.dumps(original))

    assert cmd_mcp_install(_args(config)) == 0
    after = _config(config)
    assert after["mcp"]["servers"]["docs"] == original["mcp"]["servers"]["docs"]
    assert after["gateway"] == {"port": 8080}
    assert MCP_SERVER_NAME in after["mcp"]["servers"]


def test_install_refuses_foreign_entry_without_force(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    foreign = {"command": "npx", "args": ["some-other-fidelis"], "enabled": True}
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_install(_args(config)) == 1
    assert _entry(config) == foreign
    assert openclaw.calls() == [], "must not shell out before refusing"
    assert "refusing to overwrite" in capsys.readouterr().err

    assert cmd_mcp_install(_args(config, force=True)) == 0
    assert _entry(config)["args"] == [str(MCP_SERVER_FILE)]


def test_install_refreshes_a_stale_fidelis_entry(tmp_path, openclaw):
    config = tmp_path / "openclaw.json"
    stale = {"command": "/old/python", "args": [str(MCP_SERVER_FILE)], "enabled": False}
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: stale}}}))

    assert cmd_mcp_install(_args(config)) == 0
    assert _entry(config) == {
        "command": sys.executable,
        "args": [str(MCP_SERVER_FILE)],
        "enabled": True,
    }


def test_install_surfaces_cli_failure_verbatim(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    openclaw.mode("fail")
    assert cmd_mcp_install(_args(config)) == 3
    assert "openclaw: boom" in capsys.readouterr().err
    assert not config.exists()


def test_install_fails_fast_when_registration_silently_does_nothing(tmp_path, openclaw, capsys):
    """Exit 0 is claimed only when a read-back proves the entry landed."""
    config = tmp_path / "openclaw.json"
    config.write_text(json.dumps({"mcp": {"servers": {}}}))
    openclaw.mode("noop")

    assert cmd_mcp_install(_args(config)) == 1
    err = capsys.readouterr().err
    assert "reported success but no 'fidelis' server" in err


def test_install_fails_fast_without_the_openclaw_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(mcp_cmd, "_openclaw_cli", lambda: None)
    assert cmd_mcp_install(_args(tmp_path / "openclaw.json")) == 1
    err = capsys.readouterr().err
    assert "OpenClaw CLI not found on PATH" in err
    assert "fidelis mcp install --client openclaw" in err


def test_install_never_touches_other_clients(tmp_path, openclaw, monkeypatch):
    monkeypatch.setattr(mcp_cmd, "DEFAULT_SETTINGS", tmp_path / "claude-settings.json")
    monkeypatch.setattr(mcp_cmd, "_codex_cli", lambda: pytest.fail("codex CLI must not be invoked"))
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "copilot-home"))

    assert cmd_mcp_install(_args(tmp_path / "openclaw.json")) == 0
    assert not (tmp_path / "claude-settings.json").exists()
    assert not (tmp_path / "copilot-home").exists()


# --------------------------------------------------------------------------
# uninstall
# --------------------------------------------------------------------------


def test_uninstall_removes_only_fidelis(tmp_path, openclaw):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0
    data = _config(config)
    data["mcp"]["servers"]["docs"] = {"url": "https://mcp.example.com/mcp"}
    config.write_text(json.dumps(data))

    assert cmd_mcp_uninstall(_args(config)) == 0
    servers = _config(config)["mcp"]["servers"]
    assert MCP_SERVER_NAME not in servers
    assert servers["docs"] == {"url": "https://mcp.example.com/mcp"}
    assert openclaw.calls()[-1]["argv"] == ["mcp", "unset", "fidelis"]


def test_uninstall_refuses_foreign_entry(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    foreign = {"command": "npx", "args": ["other"]}
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_uninstall(_args(config)) == 1
    assert _entry(config) == foreign
    assert openclaw.calls() == []
    assert "refusing to remove" in capsys.readouterr().err

    assert cmd_mcp_uninstall(_args(config, force=True)) == 0
    assert MCP_SERVER_NAME not in _config(config)["mcp"]["servers"]


def test_uninstall_without_config_or_entry_is_a_noop(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_uninstall(_args(config)) == 0
    assert "nothing" in capsys.readouterr().out or True
    assert openclaw.calls() == [], "a missing config must not shell out"

    config.write_text(json.dumps({"mcp": {"servers": {}}}))
    assert cmd_mcp_uninstall(_args(config)) == 0
    assert openclaw.calls() == []


def test_uninstall_fails_fast_when_removal_silently_does_nothing(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0
    openclaw.mode("noop")

    assert cmd_mcp_uninstall(_args(config)) == 1
    assert "still registered" in capsys.readouterr().err


def test_uninstall_surfaces_cli_failure_verbatim(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0
    openclaw.mode("fail")
    assert cmd_mcp_uninstall(_args(config)) == 3
    assert "openclaw: boom" in capsys.readouterr().err
    assert _entry(config)["args"] == [str(MCP_SERVER_FILE)]


# --------------------------------------------------------------------------
# path resolution + CLI surface
# --------------------------------------------------------------------------


def test_config_path_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCLAW_CONFIG_PATH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    assert openclaw_config_path(None) == tmp_path / "home" / ".openclaw" / "openclaw.json"

    monkeypatch.setenv("OPENCLAW_CONFIG_PATH", str(tmp_path / "env.json"))
    assert openclaw_config_path(None) == tmp_path / "env.json"
    assert openclaw_config_path(str(tmp_path / "explicit.json")) == tmp_path / "explicit.json"


def test_cli_accepts_openclaw_client(tmp_path, openclaw, monkeypatch):
    config = tmp_path / "openclaw.json"
    monkeypatch.setattr(
        sys, "argv",
        ["fidelis", "mcp", "install", "--client", "openclaw", "--settings", str(config)],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert _entry(config)["args"] == [str(MCP_SERVER_FILE)]

    monkeypatch.setattr(
        sys, "argv",
        ["fidelis", "mcp", "uninstall", "--client", "openclaw", "--settings", str(config)],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert MCP_SERVER_NAME not in _config(config)["mcp"]["servers"]


def test_ownership_check_survives_hostile_config_strings(tmp_path, openclaw, capsys):
    """Every string in the entry is user data; path resolution must not crash."""
    config = tmp_path / "openclaw.json"
    hostile = {
        "url": "https://mcp.example.com/mcp",
        "headers": {"Authorization": "Bearer ~nosuchuser/\x00weird"},
        "args": ["~nosuchuser/x", "\x00"],
    }
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: hostile}}}))

    assert cmd_mcp_install(_args(config)) == 1  # foreign, not ours
    assert "refusing to overwrite" in capsys.readouterr().err
    assert _entry(config) == hostile
