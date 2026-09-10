"""Regression tests for `fidelis mcp install|uninstall --client openclaw`.

OpenClaw reads an optional JSON5 config from ``~/.openclaw/openclaw.json`` and
keeps outbound MCP servers under ``mcp.servers.<name>``. Because that file is
JSON5, Fidelis neither writes it nor parses it: the documented ``openclaw mcp
add`` / ``mcp unset`` CLI owns every write, the documented ``openclaw mcp show
--json`` / ``mcp list --json`` CLI answers every question about what is in it,
and ``$OPENCLAW_CONFIG_PATH`` pins which file that is for both.

These tests stand up a fake ``openclaw`` executable on PATH that implements
that contract, including reading JSON5 the way OpenClaw does. A config Fidelis
could not parse for itself is therefore fully visible here -- which is what
makes the JSON5 cases below real tests rather than tests of a blind spot.
Nothing here touches the real home directory or requires OpenClaw installed.
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
    openclaw_list_arguments,
    openclaw_show_arguments,
)


# A stand-in for the OpenClaw CLI. It honours exactly the documented surface
# Fidelis delegates to -- `mcp add --command/--arg`, `mcp unset`, `mcp show
# --json`, `mcp list --json`, and the `{"ok": false, "error": {...}}` failure
# envelope -- records every invocation, and can be told to misbehave.
#
# `FAKE_OPENCLAW_MODE` only ever changes the *write* subcommands, except for
# "unreadable", which is the config-is-broken case and only changes the reads.
# The reads are how Fidelis learns the truth; a shim that broke them everywhere
# could not tell a caught failure apart from a blind one.
FAKE_OPENCLAW = r'''#!/usr/bin/env python3
import json, os, re, sys

argv = sys.argv[1:]
record = os.environ["FAKE_OPENCLAW_LOG"]
with open(record, "a") as handle:
    handle.write(json.dumps({"argv": argv, "config": os.environ.get("OPENCLAW_CONFIG_PATH")}) + "\n")

path = os.environ["OPENCLAW_CONFIG_PATH"]
mode = os.environ.get("FAKE_OPENCLAW_MODE", "ok")
is_write = argv[:2] in (["mcp", "add"], ["mcp", "unset"])


def strip_json5(text):
    """Drop // and /* */ comments and trailing commas, respecting strings."""
    out, index, size = [], 0, len(text)
    while index < size:
        char = text[index]
        if char == '"':
            end = index + 1
            while end < size:
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == '"':
                    end += 1
                    break
                end += 1
            out.append(text[index:end])
            index = end
            continue
        if text.startswith("//", index):
            newline = text.find("\n", index)
            index = size if newline == -1 else newline
            continue
        if text.startswith("/*", index):
            close = text.find("*/", index)
            index = size if close == -1 else close + 2
            continue
        out.append(char)
        index += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def fail(message):
    print(json.dumps({"ok": False, "error": {"type": "cli_error", "message": message}}))
    sys.exit(1)


if mode == "unreadable" and not is_write:
    fail("Config file is invalid; fix it before using MCP config commands.")

if mode == "legacy" and not is_write:
    # An openclaw that writes but has no read-only surface to confirm with:
    # the answer is unknown, and unknown is not "nothing there".
    sys.stderr.write("error: unknown command '" + " ".join(argv[1:2]) + "'\n")
    sys.exit(2)

if is_write and mode == "fail":
    sys.stderr.write("openclaw: boom\n")
    sys.exit(3)

if is_write and mode == "noop":
    # Accept the call but leave the config alone. Models a write that silently
    # did not land -- Fidelis must catch that by read-back, not by trusting the
    # exit code.
    print("ok")
    sys.exit(0)

try:
    with open(path) as handle:
        config = json.loads(strip_json5(handle.read()))
except FileNotFoundError:
    config = {}
servers = config.setdefault("mcp", {}).setdefault("servers", {})

if argv[:2] == ["mcp", "list"]:
    print(json.dumps(servers))
    sys.exit(0)

if argv[:2] == ["mcp", "show"]:
    rest = [value for value in argv[2:] if not value.startswith("--")]
    if not rest:
        print(json.dumps(servers))
        sys.exit(0)
    name = rest[0]
    if name not in servers:
        fail('No MCP server named "' + name + '" in ' + path + ".")
    print(json.dumps(servers[name]))
    sys.exit(0)

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
    if argv[2] not in servers:
        fail('No MCP server named "' + argv[2] + '" in ' + path + ".")
    servers.pop(argv[2])
    print("unset " + argv[2])
else:
    sys.stderr.write("openclaw: unsupported: " + " ".join(argv) + "\n")
    sys.exit(2)

os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
with open(path, "w") as handle:
    json.dump(config, handle, indent=2)
'''

WRITE_ARGV_PREFIXES = (["mcp", "add"], ["mcp", "unset"])

# A JSON5 config: comments and a trailing comma. `json.loads` rejects every
# one of these files; OpenClaw reads them all.
JSON5_EMPTY = """{
  // hand-written note
  "mcp": {
    "servers": {},
  },
}
"""


def _json5_with_entry(entry: dict) -> str:
    return (
        "{\n"
        "  // hand-written note\n"
        '  "mcp": {\n'
        '    "servers": {\n'
        f'      "{MCP_SERVER_NAME}": {json.dumps(entry)},  /* mine, not yours */\n'
        "    },\n"
        "  },\n"
        "}\n"
    )


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
        def argvs() -> list[list[str]]:
            return [call["argv"] for call in Handle.calls()]

        @staticmethod
        def writes() -> list[list[str]]:
            return [argv for argv in Handle.argvs() if argv[:2] in WRITE_ARGV_PREFIXES]

        @staticmethod
        def mode(value: str) -> None:
            monkeypatch.setenv("FAKE_OPENCLAW_MODE", value)

    return Handle()


@pytest.fixture
def no_fidelis_writes(monkeypatch):
    """Fail the test if Fidelis writes an OpenClaw config itself."""

    def forbidden(*args, **kwargs):
        pytest.fail("Fidelis must delegate every OpenClaw config write to the CLI")

    monkeypatch.setattr(mcp_cmd, "_atomic_write_json", forbidden)
    monkeypatch.setattr(mcp_cmd, "_backup", forbidden)


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

    writes = openclaw.writes()
    assert writes == [
        ["mcp", "add", "fidelis", "--command", sys.executable, "--arg", str(MCP_SERVER_FILE)],
    ]
    assert writes[0] == openclaw_add_arguments()
    # Every delegated call -- the write and the read-back -- is pinned to the
    # file Fidelis reports on.
    assert {call["config"] for call in openclaw.calls()} == {str(config)}

    entry = _entry(config)
    assert entry["command"] == sys.executable
    assert entry["args"] == [str(MCP_SERVER_FILE)]
    assert Path(entry["args"][0]).is_file()

    out = capsys.readouterr().out
    assert str(config) in out
    assert "openclaw mcp reload" in out
    assert "openclaw mcp doctor fidelis --probe" in out


def test_install_confirms_the_write_through_the_openclaw_cli(tmp_path, openclaw):
    """The read-back is OpenClaw's own read-only surface, not a file parse."""
    config = tmp_path / "openclaw.json"
    config.write_text(JSON5_EMPTY)

    assert cmd_mcp_install(_args(config)) == 0
    argvs = openclaw.argvs()
    add = argvs.index(openclaw_add_arguments())
    assert openclaw_show_arguments() in argvs[add + 1 :], "must read back after writing"


def test_install_and_uninstall_never_write_the_config_themselves(
    tmp_path, openclaw, no_fidelis_writes
):
    """A JSON5 config is OpenClaw's to write; Fidelis only ever delegates."""
    config = tmp_path / "openclaw.json"
    config.write_text(JSON5_EMPTY)

    assert cmd_mcp_install(_args(config)) == 0
    assert cmd_mcp_uninstall(_args(config)) == 0


def test_install_leaves_a_json5_config_untouched_when_the_cli_does_nothing(tmp_path, openclaw):
    config = tmp_path / "openclaw.json"
    config.write_text(JSON5_EMPTY)
    before = config.read_text()

    openclaw.mode("noop")  # CLI accepts the call but leaves the file alone
    assert cmd_mcp_install(_args(config)) == 1, "an unconfirmed write is not a success"
    assert config.read_text() == before, "Fidelis must not rewrite a JSON5 config"


def test_install_fails_when_a_json5_registration_does_not_land(tmp_path, openclaw, capsys):
    """JSON5 is not a licence to claim success without a read-back."""
    config = tmp_path / "openclaw.json"
    config.write_text(JSON5_EMPTY)
    openclaw.mode("noop")

    assert cmd_mcp_install(_args(config)) == 1
    err = capsys.readouterr().err
    assert "reported success but no 'fidelis' server" in err


def test_install_refuses_a_foreign_json5_entry_without_force(tmp_path, openclaw, capsys):
    """The pre-flight sees a foreign entry even when the config is JSON5."""
    config = tmp_path / "openclaw.json"
    foreign = {"command": "npx", "args": ["some-other-fidelis"], "enabled": True}
    config.write_text(_json5_with_entry(foreign))
    before = config.read_text()

    assert cmd_mcp_install(_args(config)) == 1
    assert config.read_text() == before
    assert openclaw.writes() == [], "must not shell out to write before refusing"
    assert "refusing to overwrite" in capsys.readouterr().err

    assert cmd_mcp_install(_args(config, force=True)) == 0
    assert _entry(config)["args"] == [str(MCP_SERVER_FILE)]


def test_install_refuses_when_openclaw_cannot_report_the_state(tmp_path, openclaw, capsys):
    """An unreadable config is 'unknown', never 'nothing there'."""
    config = tmp_path / "openclaw.json"
    config.write_text("{ this is not a config at all")
    openclaw.mode("unreadable")

    assert cmd_mcp_install(_args(config)) == 1
    err = capsys.readouterr().err
    assert "could not confirm what 'fidelis' is" in err
    assert "Config file is invalid" in err
    assert openclaw.writes() == [], "must not shell out to write before refusing"


def test_install_refuses_when_the_cli_has_no_read_only_surface(tmp_path, openclaw, capsys):
    """An openclaw that cannot be asked is 'unknown', even though it can write."""
    config = tmp_path / "openclaw.json"
    config.write_text(JSON5_EMPTY)
    openclaw.mode("legacy")

    assert cmd_mcp_install(_args(config)) == 1
    err = capsys.readouterr().err
    assert "could not confirm what 'fidelis' is" in err
    assert openclaw.writes() == [], "must not shell out to write before refusing"


def test_install_force_past_an_unknown_state_still_needs_a_read_back(tmp_path, openclaw, capsys):
    """--force skips the pre-flight refusal; it does not buy a claimed success."""
    config = tmp_path / "openclaw.json"
    config.write_text(JSON5_EMPTY)
    openclaw.mode("legacy")

    assert cmd_mcp_install(_args(config, force=True)) == 1
    assert openclaw.writes() == [openclaw_add_arguments()], "the write was attempted"
    err = capsys.readouterr().err
    assert "could not be confirmed" in err
    assert "openclaw mcp show fidelis --json" in err


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
    assert openclaw.writes() == [], "must not shell out to write before refusing"
    assert "refusing to overwrite" in capsys.readouterr().err

    assert cmd_mcp_install(_args(config, force=True)) == 0
    assert _entry(config)["args"] == [str(MCP_SERVER_FILE)]


def test_install_refusal_does_not_leak_the_foreign_entrys_credentials(tmp_path, openclaw, capsys):
    """The refused entry is echoed for recognition, but never its secrets."""
    config = tmp_path / "openclaw.json"
    foreign = {
        "command": "npx",
        "args": ["some-other-fidelis"],
        "env": {"API_TOKEN": "sk-live-should-not-appear"},
        "headers": {"Authorization": "Bearer should-not-appear-either"},
    }
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_install(_args(config)) == 1
    err = capsys.readouterr().err
    assert "refusing to overwrite" in err
    assert "sk-live-should-not-appear" not in err
    assert "should-not-appear-either" not in err
    assert '"command": "npx"' in err
    assert "withheld: env, headers" in err


def test_install_refusal_does_not_leak_a_credential_passed_as_a_launch_argument(
    tmp_path, openclaw, capsys
):
    """A credential can arrive as a bare launch argument, not just env/headers
    -- being a launch field does not make a value safe to print."""
    config = tmp_path / "openclaw.json"
    foreign = {
        "command": "npx",
        "args": ["--api-key", "sk-live-should-not-appear-in-args"],
    }
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_install(_args(config)) == 1
    err = capsys.readouterr().err
    assert "refusing to overwrite" in err
    assert "sk-live-should-not-appear-in-args" not in err
    assert "--api-key" not in err
    assert "2 argument(s) withheld" in err


def test_install_refusal_does_not_leak_a_malformed_non_object_entry(tmp_path, openclaw, capsys):
    """Neither reader validates an entry's shape -- a hand-edited config can
    put a credential directly where an object is expected."""
    config = tmp_path / "openclaw.json"
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: "sk-live-should-not-appear"}}}))

    assert cmd_mcp_install(_args(config)) == 1
    err = capsys.readouterr().err
    assert "sk-live-should-not-appear" not in err
    assert "non-object entry" in err


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


def test_install_fails_when_a_stale_owned_entry_is_not_actually_refreshed(tmp_path, openclaw, capsys):
    """Ownership already matching is not enough to claim success.

    The pre-existing entry already names our script (so it already reads
    back as ours), but is stale -- a different interpreter, left disabled.
    If `openclaw mcp add` silently no-ops, ownership alone would still read
    back as `_OC_OURS` and the install would wrongly report success."""
    config = tmp_path / "openclaw.json"
    stale = {"command": "/old/python", "args": [str(MCP_SERVER_FILE)], "enabled": False}
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: stale}}}))
    openclaw.mode("noop")

    assert cmd_mcp_install(_args(config)) == 1
    err = capsys.readouterr().err
    assert "does not match what Fidelis requested" in err
    assert _entry(config) == stale, "the stale entry must be left exactly alone"


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
    assert openclaw.writes()[-1] == ["mcp", "unset", "fidelis"]


def test_uninstall_confirms_the_removal_through_the_openclaw_cli(tmp_path, openclaw):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0

    assert cmd_mcp_uninstall(_args(config)) == 0
    argvs = openclaw.argvs()
    unset = argvs.index(["mcp", "unset", MCP_SERVER_NAME])
    assert openclaw_show_arguments() in argvs[unset + 1 :], "must read back after removing"


def test_uninstall_refuses_foreign_entry(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    foreign = {"command": "npx", "args": ["other"]}
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_uninstall(_args(config)) == 1
    assert _entry(config) == foreign
    assert openclaw.writes() == []
    assert "refusing to remove" in capsys.readouterr().err

    assert cmd_mcp_uninstall(_args(config, force=True)) == 0
    assert MCP_SERVER_NAME not in _config(config)["mcp"]["servers"]


def test_uninstall_refusal_does_not_leak_the_foreign_entrys_credentials(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    foreign = {
        "command": "npx",
        "args": ["other"],
        "env": {"API_TOKEN": "sk-live-should-not-appear"},
    }
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_uninstall(_args(config)) == 1
    err = capsys.readouterr().err
    assert "refusing to remove" in err
    assert "sk-live-should-not-appear" not in err
    assert "withheld: env" in err


def test_uninstall_refuses_a_foreign_json5_entry_without_force(tmp_path, openclaw, capsys):
    """The pre-flight sees a foreign entry even when the config is JSON5."""
    config = tmp_path / "openclaw.json"
    foreign = {"command": "npx", "args": ["some-other-fidelis"], "enabled": True}
    config.write_text(_json5_with_entry(foreign))
    before = config.read_text()

    assert cmd_mcp_uninstall(_args(config)) == 1
    assert config.read_text() == before
    assert openclaw.writes() == []
    assert "refusing to remove" in capsys.readouterr().err

    assert cmd_mcp_uninstall(_args(config, force=True)) == 0
    assert MCP_SERVER_NAME not in _config(config)["mcp"]["servers"]


def test_uninstall_refuses_when_openclaw_cannot_report_the_state(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    config.write_text("{ this is not a config at all")
    openclaw.mode("unreadable")

    assert cmd_mcp_uninstall(_args(config)) == 1
    err = capsys.readouterr().err
    assert "could not confirm what 'fidelis' is" in err
    assert openclaw.writes() == []


def test_uninstall_without_config_or_entry_is_a_noop(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_uninstall(_args(config)) == 0
    assert "nothing to uninstall" in capsys.readouterr().out
    assert openclaw.calls() == [], "a missing config must not shell out at all"

    config.write_text(json.dumps({"mcp": {"servers": {}}}))
    assert cmd_mcp_uninstall(_args(config)) == 0
    assert "no 'fidelis' MCP server registered" in capsys.readouterr().out
    assert openclaw.writes() == [], "nothing to remove must not shell out to write"


def test_uninstall_fails_fast_when_removal_silently_does_nothing(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0
    openclaw.mode("noop")

    assert cmd_mcp_uninstall(_args(config)) == 1
    assert "still registered" in capsys.readouterr().err


def test_uninstall_fails_when_a_json5_removal_does_not_land(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    entry = {"command": sys.executable, "args": [str(MCP_SERVER_FILE)], "enabled": True}
    config.write_text(_json5_with_entry(entry))
    before = config.read_text()
    openclaw.mode("noop")

    assert cmd_mcp_uninstall(_args(config)) == 1
    assert "still registered" in capsys.readouterr().err
    assert config.read_text() == before


def test_uninstall_surfaces_cli_failure_verbatim(tmp_path, openclaw, capsys):
    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0
    openclaw.mode("fail")
    assert cmd_mcp_uninstall(_args(config)) == 3
    assert "openclaw: boom" in capsys.readouterr().err
    assert _entry(config)["args"] == [str(MCP_SERVER_FILE)]


def test_uninstall_fails_fast_without_the_openclaw_cli(tmp_path, monkeypatch, capsys):
    config = tmp_path / "openclaw.json"
    config.write_text(JSON5_EMPTY)
    monkeypatch.setattr(mcp_cmd, "_openclaw_cli", lambda: None)

    assert cmd_mcp_uninstall(_args(config)) == 1
    err = capsys.readouterr().err
    assert "OpenClaw CLI not found on PATH" in err
    assert "fidelis mcp uninstall --client openclaw" in err


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


def test_delegated_read_only_surface_is_the_documented_one():
    assert openclaw_show_arguments() == ["mcp", "show", MCP_SERVER_NAME, "--json"]
    assert openclaw_list_arguments() == ["mcp", "list", "--json"]


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


def test_ownership_check_ignores_our_path_outside_the_launch_fields(tmp_path, openclaw, capsys):
    """Our script path as *data* in env/headers/url does not make an entry
    ours -- only its own command/args do.

    A foreign server is free to reference the Fidelis script path in
    metadata unrelated to what it launches (an env var passed through to a
    child process, a header, a URL). Recognizing that as ownership would let
    Fidelis silently overwrite -- or, on uninstall, delete -- a server that
    was never its own."""
    config = tmp_path / "openclaw.json"
    foreign = {
        "command": "node",
        "args": ["./dist/mcp-server.js"],
        "env": {"FIDELIS_SCRIPT_HINT": str(MCP_SERVER_FILE)},
        "headers": {"X-Origin": str(MCP_SERVER_FILE)},
    }
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_install(_args(config)) == 1  # foreign, not ours
    assert _entry(config) == foreign
    assert openclaw.writes() == [], "must not shell out to write before refusing"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_ownership_check_ignores_our_path_used_as_input_not_as_the_launch(
    tmp_path, openclaw, capsys
):
    """Our script path as a *later* argument does not make an entry ours --
    only the command and the first argument, the actual launch position,
    do.

    A foreign server can take our script path as its own input (e.g. an
    ``--input-file`` argument) while launching something else entirely at
    position 0. Matching anywhere in ``args`` would recognize that as
    ownership and let Fidelis overwrite or delete a server that never was
    its own."""
    config = tmp_path / "openclaw.json"
    foreign = {
        "command": "python3",
        "args": ["some_other_script.py", "--input-file", str(MCP_SERVER_FILE)],
    }
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_install(_args(config)) == 1  # foreign, not ours
    assert _entry(config) == foreign
    assert openclaw.writes() == [], "must not shell out to write before refusing"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_ownership_check_recognizes_the_current_interpreter_however_it_is_named(
    tmp_path, openclaw, monkeypatch
):
    """Whatever this process's own interpreter is named, it is what
    `openclaw mcp add --command` actually writes -- so it must always be
    recognized as ours, even when its name doesn't look Python-ish at all
    (a custom build, a wrapper script)."""
    fake_interpreter = tmp_path / "my-custom-interpreter"
    fake_interpreter.write_text("#!/bin/sh\n")
    fake_interpreter.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(fake_interpreter))

    config = tmp_path / "openclaw.json"
    assert cmd_mcp_install(_args(config)) == 0
    assert _entry(config)["command"] == str(fake_interpreter)


def test_ownership_check_recognizes_a_stale_pypy_entry_by_name(tmp_path, openclaw):
    """A prior install under PyPy, from a since-removed venv, is still ours
    to refresh -- recognized by name, not by an exact path match."""
    config = tmp_path / "openclaw.json"
    stale = {"command": "/old/venv/bin/pypy3", "args": [str(MCP_SERVER_FILE)], "enabled": False}
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: stale}}}))

    assert cmd_mcp_install(_args(config)) == 0  # no --force needed
    assert _entry(config)["command"] == sys.executable


def test_ownership_check_requires_a_python_command_not_just_a_matching_argument(
    tmp_path, openclaw, capsys
):
    """A single argument naming our script is not ownership on its own --
    only a Python interpreter can actually execute it. An unrelated tool
    that merely accepts the path as its one input (reads it, rather than
    running it) must not be recognized as Fidelis's own."""
    config = tmp_path / "openclaw.json"
    foreign = {"command": "cat", "args": [str(MCP_SERVER_FILE)]}
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_install(_args(config)) == 1  # foreign, not ours
    assert _entry(config) == foreign
    assert openclaw.writes() == [], "must not shell out to write before refusing"
    assert "refusing to overwrite" in capsys.readouterr().err


def test_ownership_check_requires_an_exact_interpreter_name_not_a_substring(
    tmp_path, openclaw, capsys
):
    """`python-config` (a real tool bundled with a Python install, not an
    interpreter) must not be recognized just because its name contains
    "python"."""
    config = tmp_path / "openclaw.json"
    foreign = {"command": "/usr/bin/python3-config", "args": [str(MCP_SERVER_FILE)]}
    config.write_text(json.dumps({"mcp": {"servers": {MCP_SERVER_NAME: foreign}}}))

    assert cmd_mcp_install(_args(config)) == 1  # foreign, not ours
    assert _entry(config) == foreign
    assert openclaw.writes() == [], "must not shell out to write before refusing"
    assert "refusing to overwrite" in capsys.readouterr().err
