"""Regression tests for `fidelis mcp install|uninstall --client gemini`.

Gemini CLI owns the write: `gemini mcp add|remove` is the only thing that
touches settings.json, because Gemini reads that file as JSON-with-comments
and round-trips a user's comments through its own writer. Fidelis only reads
it back, to prove ownership before replacing or removing an entry and to prove
what a run actually changed.

The `gemini` binary is faked here. The fake reproduces the behaviours these
tests exist to guard, all confirmed against Gemini CLI 0.32.1 and the sources
in google-gemini/gemini-cli:

  - `mcp add` silently overwrites an entry of the same name (add.ts)
  - `mcp remove` exits 0 when the name is absent (remove.ts)
  - both leave unrelated servers, unrelated settings keys, and comments alone
"""

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fidelis import cli, mcp_cmd
from fidelis.mcp_cmd import (
    MCP_SERVER_FILE,
    _gemini_entry_matches,
    _is_fidelis_gemini_entry,
    _read_gemini_servers,
    _strip_json_comments,
    cmd_mcp_install,
    cmd_mcp_uninstall,
    gemini_server_entry,
    gemini_settings_path,
)


def _args(scope: str | None = "user", force: bool = False, client: str = "gemini",
          settings: str | None = None) -> Namespace:
    return Namespace(client=client, settings=settings, scope=scope, force=force)


def _settings(home: Path) -> Path:
    return home / ".gemini" / "settings.json"


def _read(path: Path) -> dict:
    return json.loads(_strip_json_comments(path.read_text()))


class FakeGemini:
    """Stand-in for the `gemini` binary, driven through subprocess.run.

    Records every argv it is handed so a test can assert the exact native
    command Fidelis issued, and mutates a settings.json the way Gemini does.
    """

    def __init__(self, path: Path, version: str = "0.32.1"):
        self.path = path
        self.version = version
        self.calls: list[list[str]] = []
        self.add_returncode = 0
        self.remove_returncode = 0
        self.stderr = ""
        # When set, `mcp add`/`mcp remove` report success without writing —
        # the no-op that no exit code reveals.
        self.silently_skip_write = False

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(_strip_json_comments(self.path.read_text()))

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    def run(self, argv, capture_output=True, text=True, check=False):
        self.calls.append(list(argv))
        if argv[1:2] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, f"{self.version}\n", "")

        assert argv[1] == "mcp", argv
        action, name = argv[2], argv[3]

        if action == "add":
            if self.add_returncode != 0:
                return subprocess.CompletedProcess(argv, self.add_returncode, "", self.stderr)
            if not self.silently_skip_write:
                data = self._load()
                servers = data.setdefault("mcpServers", {})
                servers[name] = {"command": argv[4], "args": list(argv[5:argv.index("--scope")])}
                self._save(data)
            out = f'MCP server "{name}" added to settings. (stdio)\n'
            return subprocess.CompletedProcess(argv, 0, out, "")

        if action == "remove":
            if self.remove_returncode != 0:
                return subprocess.CompletedProcess(argv, self.remove_returncode, "", self.stderr)
            data = self._load()
            servers = data.get("mcpServers", {})
            if name in servers and not self.silently_skip_write:
                del servers[name]
                self._save(data)
                out = f'Server "{name}" removed from settings.\n'
            else:
                # Gemini exits 0 here too.
                out = f'Server "{name}" not found in settings.\n'
            return subprocess.CompletedProcess(argv, 0, out, "")

        raise AssertionError(f"unexpected gemini invocation: {argv}")


@pytest.fixture
def gemini(tmp_path, monkeypatch):
    """A fake `gemini` on PATH, with HOME pointed at a scratch directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    fake = FakeGemini(_settings(home))
    monkeypatch.setattr(mcp_cmd.shutil, "which", lambda name: "/fake/bin/gemini" if name == "gemini" else None)
    monkeypatch.setattr(mcp_cmd.subprocess, "run", fake.run)
    fake.home = home
    return fake


# --------------------------------------------------------------------------
# absent → installed
# --------------------------------------------------------------------------


def test_install_delegates_to_native_cli_and_verifies(gemini, capsys):
    assert cmd_mcp_install(_args()) == 0

    add = [call for call in gemini.calls if call[1:3] == ["mcp", "add"]]
    assert add == [[
        "/fake/bin/gemini", "mcp", "add", "fidelis",
        sys.executable, str(MCP_SERVER_FILE),
        "--scope", "user",
        "--transport", "stdio",
    ]]

    entry = _read(gemini.path)["mcpServers"]["fidelis"]
    assert entry == gemini_server_entry()
    assert Path(entry["args"][0]).is_file()

    out = capsys.readouterr().out
    assert f"verified 'fidelis' in {gemini.path} (user scope)" in out
    assert "/mcp reload" in out


def test_install_project_scope_targets_cwd(tmp_path, gemini, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.chdir(project)
    gemini.path = _settings(project)

    assert cmd_mcp_install(_args(scope="project")) == 0
    assert "--scope" in gemini.calls[-1]
    assert gemini.calls[-1][gemini.calls[-1].index("--scope") + 1] == "project"
    assert gemini_settings_path("project") == project / ".gemini" / "settings.json"
    assert _read(gemini.path)["mcpServers"]["fidelis"] == gemini_server_entry()


# --------------------------------------------------------------------------
# owned → idempotent refresh
# --------------------------------------------------------------------------


def test_install_is_idempotent_without_calling_add(gemini, capsys):
    assert cmd_mcp_install(_args()) == 0
    capsys.readouterr()
    before = len([c for c in gemini.calls if c[1:3] == ["mcp", "add"]])

    assert cmd_mcp_install(_args()) == 0
    after = len([c for c in gemini.calls if c[1:3] == ["mcp", "add"]])

    assert after == before, "an already-correct entry must not be rewritten"
    assert "nothing to change" in capsys.readouterr().out


def test_install_refreshes_a_fidelis_entry_from_another_environment(gemini):
    """A Fidelis server registered from a different Python is ours to replace."""
    stale = {"command": "/other/venv/bin/python", "args": ["/other/venv/lib/fidelis/mcp_server.py"]}
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": stale}}))
    assert _is_fidelis_gemini_entry(stale)

    assert cmd_mcp_install(_args()) == 0  # no --force needed
    assert _read(gemini.path)["mcpServers"]["fidelis"] == gemini_server_entry()


# --------------------------------------------------------------------------
# foreign → refused, and unrelated state preserved
# --------------------------------------------------------------------------


def test_install_refuses_a_foreign_entry_of_the_same_name(gemini, capsys):
    foreign = {"command": "npx", "args": ["-y", "@someone/fidelis-mcp"], "env": {"TOKEN": "secret"}}
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))

    assert cmd_mcp_install(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "add"]], "must not write"
    assert _read(gemini.path)["mcpServers"]["fidelis"] == foreign

    err = capsys.readouterr().err
    assert "refusing to overwrite" in err
    assert "--force" in err


def test_install_refusal_does_not_leak_the_foreign_entrys_credentials(gemini, capsys):
    """The refused entry is echoed for recognition, but never its secrets."""
    foreign = {
        "command": "npx",
        "args": ["-y", "@someone/fidelis-mcp"],
        "env": {"API_TOKEN": "sk-live-should-not-appear"},
        "headers": {"Authorization": "Bearer should-not-appear-either"},
    }
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))

    assert cmd_mcp_install(_args()) == 1
    err = capsys.readouterr().err
    assert "refusing to overwrite" in err
    assert "sk-live-should-not-appear" not in err
    assert "should-not-appear-either" not in err
    # The user can still recognize what was refused and why.
    assert '"command": "npx"' in err
    assert "withheld: env, headers" in err


def test_install_refusal_does_not_leak_a_credential_passed_as_a_launch_argument(gemini, capsys):
    """A credential can arrive as a bare launch argument, not just env/headers
    -- being a launch field does not make a value safe to print."""
    foreign = {"command": "npx", "args": ["--api-key", "sk-live-should-not-appear-in-args"]}
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))

    assert cmd_mcp_install(_args()) == 1
    err = capsys.readouterr().err
    assert "refusing to overwrite" in err
    assert "sk-live-should-not-appear-in-args" not in err
    assert "--api-key" not in err
    assert "2 argument(s) withheld" in err


def test_install_refusal_does_not_leak_a_malformed_non_object_entry(gemini, capsys):
    """The Gemini settings.json reader does not validate an entry's shape --
    a hand-edited config can put a credential directly where an object is
    expected, and the diagnostic must not echo it just because it isn't a
    dict Fidelis knows how to check ownership on."""
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": "sk-live-should-not-appear"}}))

    assert cmd_mcp_install(_args()) == 1
    err = capsys.readouterr().err
    assert "sk-live-should-not-appear" not in err
    assert "non-object entry" in err


def test_install_refusal_withholds_a_non_scalar_safe_field(gemini, capsys):
    """A "safe" field is only safe once its value is actually confirmed to be
    a scalar -- a malformed config can nest a credential under `command`."""
    foreign = {"command": {"nested": "sk-live-should-not-appear"}, "args": []}
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))

    assert cmd_mcp_install(_args()) == 1
    err = capsys.readouterr().err
    assert "sk-live-should-not-appear" not in err
    assert "withheld: command" in err


def test_install_force_replaces_a_foreign_entry(gemini):
    foreign = {"command": "npx", "args": ["-y", "@someone/fidelis-mcp"]}
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))

    assert cmd_mcp_install(_args(force=True)) == 0
    assert _read(gemini.path)["mcpServers"]["fidelis"] == gemini_server_entry()


def test_install_preserves_foreign_servers_settings_and_comments(gemini, monkeypatch):
    """The native CLI owns the write precisely so this survives."""
    original = """{
  // my own note, which a Fidelis rewrite would delete
  "ui": { "theme": "Default" },
  "mcpServers": {
    /* keep this one */
    "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": { "GITHUB_TOKEN": "ghp_secret" } }
  }
}"""
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(original)

    # Stand in for Gemini's comment-preserving writer: splice the entry in
    # without disturbing the rest of the document.
    def preserving_run(argv, **kwargs):
        gemini.calls.append(list(argv))
        if argv[1:2] == ["--version"]:
            return subprocess.CompletedProcess(argv, 0, "0.32.1\n", "")
        entry = json.dumps({"command": argv[4], "args": [argv[5]]})
        text = gemini.path.read_text().replace(
            '"mcpServers": {', '"mcpServers": {\n    "fidelis": ' + entry + ",", 1
        )
        gemini.path.write_text(text)
        return subprocess.CompletedProcess(argv, 0, "added\n", "")

    monkeypatch.setattr(mcp_cmd.subprocess, "run", preserving_run)
    assert cmd_mcp_install(_args()) == 0

    text = gemini.path.read_text()
    assert "// my own note" in text
    assert "/* keep this one */" in text
    data = _read(gemini.path)
    assert data["ui"] == {"theme": "Default"}
    assert data["mcpServers"]["github"]["env"] == {"GITHUB_TOKEN": "ghp_secret"}
    assert data["mcpServers"]["fidelis"] == gemini_server_entry()


# --------------------------------------------------------------------------
# no-op / ambiguous states must fail, never be reported as success
# --------------------------------------------------------------------------


def test_install_fails_when_add_exits_zero_but_writes_nothing(gemini, capsys):
    gemini.silently_skip_write = True

    assert cmd_mcp_install(_args()) == 1
    assert "nothing was installed" in capsys.readouterr().err


def test_install_fails_when_add_writes_a_different_entry(gemini, capsys, monkeypatch):
    real_run = gemini.run

    def divert(argv, **kwargs):
        result = real_run(argv, **kwargs)
        if argv[1:3] == ["mcp", "add"]:
            data = _read(gemini.path)
            data["mcpServers"]["fidelis"]["command"] = "/somewhere/else/python"
            gemini.path.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(mcp_cmd.subprocess, "run", divert)

    assert cmd_mcp_install(_args()) == 1
    assert "wrote an unexpected 'fidelis' entry" in capsys.readouterr().err


def test_uninstall_fails_when_remove_exits_zero_but_entry_remains(gemini, capsys):
    assert cmd_mcp_install(_args()) == 0
    capsys.readouterr()
    gemini.silently_skip_write = True

    assert cmd_mcp_uninstall(_args()) == 1
    assert "still present" in capsys.readouterr().err


def test_project_scope_in_home_directory_is_refused_as_ambiguous(gemini, monkeypatch, capsys):
    monkeypatch.chdir(gemini.home)

    assert cmd_mcp_install(_args(scope="project")) == 1
    err = capsys.readouterr().err
    assert "resolves to the user settings file" in err
    assert "--scope user" in err
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "add"]]


def test_install_surfaces_the_auth_refusal(gemini, capsys):
    gemini.add_returncode = 41
    gemini.stderr = "Please set an Auth method in your settings.json"

    assert cmd_mcp_install(_args()) == 41
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err


# --------------------------------------------------------------------------
# malformed config
# --------------------------------------------------------------------------


def test_install_refuses_a_malformed_settings_file(gemini, capsys):
    gemini.path.parent.mkdir(parents=True)
    # A trailing comma: legal JSON5, rejected by Gemini's own parser.
    gemini.path.write_text('{"mcpServers": {"a": {"command": "x"},},}')

    assert cmd_mcp_install(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "add"]]
    err = capsys.readouterr().err
    assert "not valid Gemini settings JSON" in err
    assert "trailing comma" in err


def test_install_refuses_a_settings_file_that_is_not_valid_text(gemini, capsys):
    gemini.path.parent.mkdir(parents=True)
    # Invalid UTF-8: ``Path.read_text()`` raises ``UnicodeDecodeError`` (a
    # ``ValueError``), which must be reported, not escape as a traceback.
    gemini.path.write_bytes(b'{"mcpServers": {"a": {"command": "\xff\xfe"}}}')

    assert cmd_mcp_install(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "add"]]
    assert "is not valid text" in capsys.readouterr().err


def test_uninstall_refuses_a_settings_file_that_is_not_valid_text(gemini, capsys):
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_bytes(b'{"mcpServers": {"fidelis": {"command": "\xff\xfe"}}}')

    assert cmd_mcp_uninstall(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "remove"]]
    assert "is not valid text" in capsys.readouterr().err
    assert gemini.path.read_bytes() == b'{"mcpServers": {"fidelis": {"command": "\xff\xfe"}}}'


def test_install_refuses_a_non_object_mcp_servers_block(gemini, capsys):
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text('{"mcpServers": ["not", "an", "object"]}')

    assert cmd_mcp_install(_args()) == 1
    assert "is not a JSON object" in capsys.readouterr().err


def test_uninstall_refuses_a_malformed_settings_file(gemini, capsys):
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text("{ not json at all }")

    assert cmd_mcp_uninstall(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "remove"]]
    assert "not valid Gemini settings JSON" in capsys.readouterr().err


# --------------------------------------------------------------------------
# uninstall
# --------------------------------------------------------------------------


def test_uninstall_removes_only_the_fidelis_entry(gemini, capsys):
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({
        "mcpServers": {"github": {"command": "npx", "args": ["-y", "server-github"]}},
        "ui": {"theme": "Default"},
    }))
    assert cmd_mcp_install(_args()) == 0
    capsys.readouterr()

    assert cmd_mcp_uninstall(_args()) == 0

    assert gemini.calls[-1] == ["/fake/bin/gemini", "mcp", "remove", "fidelis", "--scope", "user"]
    data = _read(gemini.path)
    assert "fidelis" not in data["mcpServers"]
    assert data["mcpServers"]["github"] == {"command": "npx", "args": ["-y", "server-github"]}
    assert data["ui"] == {"theme": "Default"}
    assert "verified 'fidelis' is gone" in capsys.readouterr().out


def test_uninstall_is_a_clean_noop_when_absent(gemini, capsys):
    assert cmd_mcp_uninstall(_args()) == 0
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "remove"]]
    assert "nothing to uninstall" in capsys.readouterr().out


def test_uninstall_refuses_a_foreign_entry(gemini, capsys):
    foreign = {"command": "npx", "args": ["-y", "@someone/fidelis-mcp"]}
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))

    assert cmd_mcp_uninstall(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "remove"]]
    assert _read(gemini.path)["mcpServers"]["fidelis"] == foreign
    assert "refusing to remove it" in capsys.readouterr().err


def test_uninstall_refusal_does_not_leak_the_foreign_entrys_credentials(gemini, capsys):
    foreign = {
        "command": "npx",
        "args": ["-y", "@someone/fidelis-mcp"],
        "env": {"API_TOKEN": "sk-live-should-not-appear"},
    }
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": foreign}}))

    assert cmd_mcp_uninstall(_args()) == 1
    err = capsys.readouterr().err
    assert "refusing to remove it" in err
    assert "sk-live-should-not-appear" not in err
    assert "withheld: env" in err


def test_uninstall_force_removes_a_foreign_entry(gemini):
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps(
        {"mcpServers": {"fidelis": {"command": "npx", "args": ["-y", "@someone/fidelis-mcp"]}}}
    ))

    assert cmd_mcp_uninstall(_args(force=True)) == 0
    assert "fidelis" not in _read(gemini.path)["mcpServers"]


@pytest.mark.parametrize("hostile_arg", [
    "a\x00b",                        # JSON \u0000 decodes to a NUL: Path.resolve raises
    "~nosuchuser000/mcp_server.py",  # unresolvable ~user: Path.expanduser raises
])
def test_ownership_check_survives_hostile_settings_strings(gemini, capsys, hostile_arg):
    """Every string in settings.json is user data; path resolution must not crash.

    Both inputs are legal Gemini settings (``\u0000`` is valid JSON), and both
    used to escape install and uninstall as a traceback instead of the
    ownership refusal -- so the file was neither refused nor protected."""
    hostile = {"command": "python", "args": [hostile_arg]}
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text(json.dumps({"mcpServers": {"fidelis": hostile}}))

    assert cmd_mcp_install(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "add"]], "must not write"
    assert "refusing to overwrite" in capsys.readouterr().err

    assert cmd_mcp_uninstall(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "remove"]], "must not write"
    assert "refusing to remove it" in capsys.readouterr().err

    assert _read(gemini.path)["mcpServers"]["fidelis"] == hostile


# --------------------------------------------------------------------------
# preflight: binary, version boundary, flag routing
# --------------------------------------------------------------------------


def test_missing_gemini_binary_is_reported_not_worked_around(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(mcp_cmd.shutil, "which", lambda name: None)

    assert cmd_mcp_install(_args()) == 1
    assert not _settings(tmp_path).exists(), "must not hand-edit settings.json"
    assert "Gemini CLI not found on PATH" in capsys.readouterr().err


def test_version_below_the_mcp_subcommand_boundary_is_refused(gemini, capsys):
    # `gemini mcp add|remove|list` first shipped in v0.1.19 (gemini-cli#5481).
    gemini.version = "0.1.18"

    assert cmd_mcp_install(_args()) == 1
    assert not [c for c in gemini.calls if c[1:3] == ["mcp", "add"]]
    err = capsys.readouterr().err
    assert "no `gemini mcp` subcommand" in err
    assert "v0.1.19" in err


def test_version_at_the_boundary_is_accepted(gemini):
    gemini.version = "0.1.19"
    assert cmd_mcp_install(_args()) == 0


def test_unparseable_version_warns_but_proceeds(gemini, capsys):
    gemini.version = "built from source"

    assert cmd_mcp_install(_args()) == 0
    assert "could not read `gemini --version`" in capsys.readouterr().err


def test_a_failing_version_probe_does_not_bury_the_real_error(gemini, monkeypatch, capsys):
    """A broken settings.json makes `gemini --version` exit non-zero too.

    Warning about the version there would put a guess in front of the exact
    diagnosis the read-back is about to produce."""
    gemini.path.parent.mkdir(parents=True)
    gemini.path.write_text('{"mcpServers": {"a": {"command": "x"},},}')

    def failing_probe(argv, **kwargs):
        if argv[1:2] == ["--version"]:
            return subprocess.CompletedProcess(argv, 52, "", "Error in settings.json")
        return gemini.run(argv, **kwargs)

    monkeypatch.setattr(mcp_cmd.subprocess, "run", failing_probe)

    assert cmd_mcp_install(_args()) == 1
    err = capsys.readouterr().err
    assert "could not read `gemini --version`" not in err
    assert "not valid Gemini settings JSON" in err


def test_settings_flag_is_rejected_for_gemini(gemini, capsys):
    assert cmd_mcp_install(_args(settings="/tmp/whatever.json")) == 1
    assert "--settings is not supported for the Gemini CLI client" in capsys.readouterr().err


def test_scope_flag_is_rejected_for_other_clients(tmp_path, capsys):
    args = _args(scope="user", client="copilot", settings=str(tmp_path / "mcp-config.json"))
    assert cmd_mcp_install(args) == 1
    assert "--scope is only supported for the Gemini CLI client" in capsys.readouterr().err


@pytest.mark.parametrize("scope", [None, object(), MagicMock(), "", "workspace"])
def test_a_scope_that_was_never_passed_does_not_block_other_clients(scope, tmp_path):
    """Only argparse's own two choices count as a deliberate --scope.

    An attribute that merely exists — an absent flag, a hand-built Namespace,
    a MagicMock whose attributes autovivify — must not be read as one."""
    args = _args(scope=scope, client="copilot", settings=str(tmp_path / "mcp-config.json"))
    assert cmd_mcp_install(args) == 0


@pytest.mark.parametrize("subcommand", ["install", "uninstall"])
def test_cli_exposes_gemini_client_and_scope(subcommand, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fidelis", "mcp", subcommand, "--help"])
    with pytest.raises(SystemExit):
        cli.main()

    help_text = capsys.readouterr().out
    assert "gemini" in help_text
    assert "--scope" in help_text
    assert "user" in help_text and "project" in help_text


# --------------------------------------------------------------------------
# untrusted-input handling in the read-back parser
# --------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1} // trailing', {"a": 1}),
    ('{"a": 1 /* mid */ }', {"a": 1}),
    ('{"url": "http://x/y"}', {"url": "http://x/y"}),          # // inside a string
    ('{"note": "a /* not */ comment"}', {"note": "a /* not */ comment"}),
    ('{"esc": "quote \\" then // safe"}', {"esc": 'quote " then // safe'}),
    ('{\n// one\n"a": 1,\n/* two\nlines */\n"b": 2\n}', {"a": 1, "b": 2}),
])
def test_comment_stripper_matches_gemini_parser(text, expected):
    assert json.loads(_strip_json_comments(text)) == expected


def test_comment_stripper_keeps_line_numbers_for_error_messages():
    stripped = _strip_json_comments('{\n/* a\nb\nc */\n"bad"\n}')
    assert stripped.count("\n") == 5


@pytest.mark.parametrize("entry", [
    None, "a string", 42, [],
    {"command": "python"},                                   # no args
    {"command": "python", "args": "mcp_server.py"},          # args not a list
    {"command": "python", "args": []},                       # empty args
    {"command": "python", "args": ["a.py", "b.py"]},         # more than one arg
    {"command": "python", "args": ["/tmp/not-mcp_server.py-backup"]},
    {"command": "python", "args": ["\x00"]},                   # resolve() raises
    {"command": "python", "args": ["~nosuchuser000/mcp_server.py"]},  # expanduser() raises
])
def test_foreign_shaped_entries_are_never_claimed_as_ours(entry):
    assert not _is_fidelis_gemini_entry(entry)
    assert not _gemini_entry_matches(entry)


def test_missing_settings_file_reads_as_empty(tmp_path):
    servers, error = _read_gemini_servers(tmp_path / "nope" / "settings.json")
    assert servers == {}
    assert error is None


def test_settings_without_mcp_servers_reads_as_empty(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"ui": {"theme": "Default"}}')
    servers, error = _read_gemini_servers(path)
    assert servers == {}
    assert error is None
