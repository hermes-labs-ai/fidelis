"""Wire fidelis into a supported agent client's MCP configuration.

Claude Code configuration is edited atomically with a backup. Codex
configuration is delegated to the supported ``codex mcp`` CLI so the desktop
app, CLI, and IDE extension share the same registered server. GitHub Copilot
CLI configuration is edited atomically with a backup in the documented
``mcp-config.json`` file (``~/.copilot`` by default, or ``$COPILOT_HOME``).
Google Gemini CLI configuration is delegated to the native ``gemini mcp``
subcommands, which round-trip the comments in a user's ``settings.json``.
OpenClaw configuration is delegated to the supported ``openclaw mcp`` CLI, in
both directions: OpenClaw's config file is JSON5 (comments, trailing commas),
so a strict-JSON rewrite of it would destroy user content and a strict-JSON
read of it cannot be trusted to say what is there.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# An agent MCP config routinely carries an `env` block with API tokens, so a
# file we create for the first time is owner-only.
NEW_CONFIG_MODE = 0o600


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: temp file + os.replace. Prevents corruption if
    Claude Code (or any reader) is reading the settings file concurrently.

    os.replace swaps in the temp file's metadata, so the destination's
    permission bits are read first and reapplied. Without that, rewriting a
    0600 mcp-config.json would silently widen it to the umask default. A file
    that does not exist yet is created NEW_CONFIG_MODE.

    Uses parent/(name+".tmp") instead of with_suffix to be safe on Python
    3.10/3.11 where with_suffix raised ValueError on multi-dot suffixes."""
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = NEW_CONFIG_MODE
    tmp = path.parent / (path.name + ".tmp")
    # Create restrictively, then widen to the destination mode: the contents
    # must never pass through a file more readable than the destination.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(data, indent=2))
        os.chmod(tmp, mode)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _backup(path: Path) -> Path:
    """Copy `path` aside and return a restore point that never overwrites an
    earlier one.

    A whole-second timestamp alone collides when two mutations land in the same
    second — an install followed straight away by an uninstall — which would
    destroy the older restore point. O_EXCL plus a counter keeps both."""
    stamp = int(time.time())
    for attempt in range(1000):
        suffix = f".bak.{stamp}" if attempt == 0 else f".bak.{stamp}.{attempt}"
        backup = path.parent / (path.name + suffix)
        try:
            os.close(os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
        except FileExistsError:
            continue
        shutil.copy(path, backup)
        return backup
    raise OSError(f"could not allocate an unused backup name for {path}")


DEFAULT_SETTINGS = Path.home() / ".claude" / "settings.local.json"
MCP_SERVER_NAME = "fidelis"
COPILOT_MCP_CONFIG_NAME = "mcp-config.json"
OPENCLAW_CONFIG_NAME = "openclaw.json"
CURSOR_SETTINGS = Path.home() / ".cursor" / "mcp.json"

# Bundled MCP server file lives alongside this module
PACKAGE_DIR = Path(__file__).resolve().parent
MCP_SERVER_FILE = PACKAGE_DIR / "mcp_server.py"


def _cursor_entry() -> dict:
    """Use the currently installed Fidelis environment, without another fetch."""
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(MCP_SERVER_FILE)],
    }


def _cmd_cursor_install(args) -> int:
    path = Path(args.settings).expanduser() if args.settings else CURSOR_SETTINGS
    if not MCP_SERVER_FILE.is_file():
        print(f"error: bundled MCP server not found at {MCP_SERVER_FILE}", file=sys.stderr)
        return 1
    try:
        config = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read Cursor MCP config at {path}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(config, dict) or not isinstance(config.get("mcpServers", {}), dict):
        print(f"error: invalid Cursor MCP config at {path}", file=sys.stderr)
        return 1
    servers = config.setdefault("mcpServers", {})
    entry = _cursor_entry()
    existing = servers.get(MCP_SERVER_NAME)
    if existing == entry:
        print(f"Cursor MCP server '{MCP_SERVER_NAME}' is already configured")
        return 0
    if existing is not None and not args.force:
        print(
            f"error: a different Cursor MCP server named '{MCP_SERVER_NAME}' exists in {path}; "
            "use --force only if you intend to replace it",
            file=sys.stderr,
        )
        return 1
    servers[MCP_SERVER_NAME] = entry
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"backed up existing settings to {_backup(path)}")
        _atomic_write_json(path, config)
    except OSError as exc:
        print(f"error: cannot write Cursor MCP config at {path}: {exc}", file=sys.stderr)
        return 1
    print(f"registered Cursor MCP server '{MCP_SERVER_NAME}' in {path}")
    print("next: restart Cursor, open Customize > MCP, and confirm Fidelis has six tools")
    return 0


def _cmd_cursor_uninstall(args) -> int:
    path = Path(args.settings).expanduser() if args.settings else CURSOR_SETTINGS
    if not path.exists():
        print(f"no Cursor MCP config at {path}; nothing to uninstall")
        return 0
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read Cursor MCP config at {path}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(config, dict) or not isinstance(config.get("mcpServers", {}), dict):
        print(f"error: invalid Cursor MCP config at {path}", file=sys.stderr)
        return 1
    servers = config["mcpServers"]
    existing = servers.get(MCP_SERVER_NAME)
    if existing is None:
        print(f"Cursor MCP server '{MCP_SERVER_NAME}' is not configured")
        return 0
    if existing != _cursor_entry() and not args.force:
        print(
            f"error: Cursor MCP server '{MCP_SERVER_NAME}' is not this installation; "
            "refusing to remove it without --force",
            file=sys.stderr,
        )
        return 1
    del servers[MCP_SERVER_NAME]
    try:
        print(f"backed up existing settings to {_backup(path)}")
        _atomic_write_json(path, config)
    except OSError as exc:
        print(f"error: cannot write Cursor MCP config at {path}: {exc}", file=sys.stderr)
        return 1
    print(f"removed Cursor MCP server '{MCP_SERVER_NAME}' from {path}")
    return 0


def _is_fidelis_codex_entry(entry: dict) -> bool:
    """Return whether a Codex MCP entry launches this packaged server."""
    transport = entry.get("transport", entry)
    command = str(transport.get("command", ""))
    args = [str(value) for value in transport.get("args", [])]
    return (
        Path(command).expanduser().resolve() == Path(sys.executable).resolve()
        and len(args) == 1
        and Path(args[0]).expanduser().resolve() == MCP_SERVER_FILE.resolve()
    )


def _codex_cli() -> str | None:
    return shutil.which("codex")


def _codex_get(codex_bin: str) -> tuple[int, dict | None, str]:
    result = subprocess.run(
        [codex_bin, "mcp", "get", MCP_SERVER_NAME, "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return result.returncode, None, result.stderr.strip()
    try:
        return 0, json.loads(result.stdout), ""
    except json.JSONDecodeError as exc:
        return 1, None, f"Codex returned invalid JSON for '{MCP_SERVER_NAME}': {exc}"


def _cmd_codex_install(args) -> int:
    if getattr(args, "settings", None):
        print(
            "error: --settings is only supported for the Claude Code, Copilot CLI, and OpenClaw clients; "
            "Codex uses its shared config through the codex mcp CLI",
            file=sys.stderr,
        )
        return 1
    codex_bin = _codex_cli()
    if not codex_bin:
        print(
            "error: Codex CLI not found on PATH\n"
            "  install Codex, then rerun: fidelis mcp install --client codex",
            file=sys.stderr,
        )
        return 1

    rc, existing, error = _codex_get(codex_bin)
    if existing is not None:
        if _is_fidelis_codex_entry(existing):
            print("Codex MCP server 'fidelis' is already configured; nothing to change")
            return 0
        if not args.force:
            print(
                "error: a non-fidelis Codex MCP server named 'fidelis' already exists\n"
                "  refusing to overwrite. Use --force to replace it.",
                file=sys.stderr,
            )
            return 1
        # Let the supported Codex CLI replace the entry atomically. Do not
        # remove first: if registration fails, the user's prior entry remains.
    elif rc != 0 and "No MCP server named" not in error:
        print(error or "error: could not inspect Codex MCP configuration", file=sys.stderr)
        return rc

    result = subprocess.run(
        [codex_bin, "mcp", "add", MCP_SERVER_NAME, "--", sys.executable, str(MCP_SERVER_FILE)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip() or "error: Codex MCP registration failed", file=sys.stderr)
        return result.returncode

    print(result.stdout.strip() or "registered Codex MCP server 'fidelis'")
    print("next: restart Codex, then use /mcp to confirm the fidelis tools")
    return 0


def _cmd_codex_uninstall() -> int:
    codex_bin = _codex_cli()
    if not codex_bin:
        print("error: Codex CLI not found on PATH", file=sys.stderr)
        return 1
    rc, existing, error = _codex_get(codex_bin)
    if existing is None:
        if rc != 0 and "No MCP server named" not in error:
            print(error or "error: could not inspect Codex MCP configuration", file=sys.stderr)
            return rc
        print("no 'fidelis' MCP server registered in Codex; nothing to uninstall")
        return 0
    if not _is_fidelis_codex_entry(existing):
        print(
            "error: Codex MCP server 'fidelis' does not appear to belong to Fidelis; refusing to remove it",
            file=sys.stderr,
        )
        return 1
    result = subprocess.run(
        [codex_bin, "mcp", "remove", MCP_SERVER_NAME],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stderr.strip() or "error: Codex MCP removal failed", file=sys.stderr)
        return result.returncode
    print(result.stdout.strip() or "removed Codex MCP server 'fidelis'")
    return 0


# ---------------------------------------------------------------------------
# GitHub Copilot CLI
#
# Copilot CLI stores MCP servers in ``mcp-config.json`` under its configuration
# directory (``~/.copilot`` by default; ``COPILOT_HOME`` overrides it). GitHub
# documents editing that file directly as a supported alternative to
# ``copilot mcp add``. Fidelis edits it atomically with a backup so the host CLI
# does not need to be installed at configuration time and no account state is
# touched.
# ---------------------------------------------------------------------------


def copilot_config_path(settings: str | None = None) -> Path:
    """Resolve the Copilot CLI ``mcp-config.json`` path.

    Explicit ``settings`` wins, then ``$COPILOT_HOME/mcp-config.json``, then
    the documented default ``~/.copilot/mcp-config.json``."""
    if settings:
        return Path(settings).expanduser()
    home = os.environ.get("COPILOT_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".copilot"
    return base / COPILOT_MCP_CONFIG_NAME


def copilot_server_entry() -> dict:
    """The exact Copilot CLI stdio entry Fidelis registers."""
    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(MCP_SERVER_FILE)],
        "tools": ["*"],
    }


def _is_fidelis_server_path(value: object) -> bool:
    """Return whether a configured argument points at a Fidelis MCP server.

    An exact match against this install is the common case. But a user who
    registered Fidelis from another environment — a venv since rebuilt, pipx,
    uvx, a different Python prefix — has the same package laid out under a
    different root, and refusing to recognize that entry would leave them
    unable to refresh or remove their own server without --force. Requiring
    both the packaged file name and its `fidelis` package directory keeps
    foreign servers out: a near-collision such as
    ``/tmp/not-mcp_server.py-backup`` still fails.

    The value is user data read from a hand-editable config. Resolving it can
    raise -- an embedded NUL, an unresolvable ``~user`` -- and that only means
    "not our path": the caller must still refuse the entry, not crash."""
    try:
        candidate = Path(str(value)).expanduser()
        if candidate.resolve() == MCP_SERVER_FILE.resolve():
            return True
    except (OSError, ValueError, RuntimeError):
        return False
    return candidate.name == MCP_SERVER_FILE.name and candidate.parent.name == PACKAGE_DIR.name


def _is_fidelis_copilot_entry(entry: object) -> bool:
    """Return whether a Copilot MCP entry launches this packaged server."""
    if not isinstance(entry, dict):
        return False
    args = [str(value) for value in entry.get("args", [])]
    return len(args) == 1 and _is_fidelis_server_path(args[0])


def _load_json_object(path: Path) -> tuple[dict | None, str | None]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"error: {path} is not valid JSON: {exc}"
    if not isinstance(data, dict):
        return None, f"error: {path} must contain a JSON object at the top level"
    return data, None


def _cmd_copilot_install(args) -> int:
    if not MCP_SERVER_FILE.exists():
        print(
            f"error: bundled MCP server not found at {MCP_SERVER_FILE}\n"
            "  this install appears incomplete; reinstall Hermes Labs Fidelis "
            "from its tagged GitHub source (see README)",
            file=sys.stderr,
        )
        return 1

    config_path = copilot_config_path(getattr(args, "settings", None))
    if config_path.exists():
        config, error = _load_json_object(config_path)
        if error:
            print(error, file=sys.stderr)
            return 1
    else:
        config = {}

    mcp_servers = config.setdefault("mcpServers", {})
    if not isinstance(mcp_servers, dict):
        print(f"error: 'mcpServers' in {config_path} is not a JSON object", file=sys.stderr)
        return 1

    existing = mcp_servers.get(MCP_SERVER_NAME)
    entry = copilot_server_entry()
    if existing == entry:
        print(f"Copilot CLI MCP server '{MCP_SERVER_NAME}' is already configured in {config_path}; nothing to change")
        return 0
    if existing is not None and not _is_fidelis_copilot_entry(existing) and not args.force:
        print(
            f"error: a non-fidelis Copilot CLI MCP server named '{MCP_SERVER_NAME}' already exists in {config_path}\n"
            f"  entry: {json.dumps(existing)}\n"
            "  refusing to overwrite. Use --force to replace it.",
            file=sys.stderr,
        )
        return 1

    if config_path.exists():
        print(f"backed up existing config to {_backup(config_path)}")
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)

    mcp_servers[MCP_SERVER_NAME] = entry
    _atomic_write_json(config_path, config)
    print(f"wrote MCP server '{MCP_SERVER_NAME}' to {config_path}")
    print()
    print("next: restart Copilot CLI, then run /mcp list to see the fidelis server")
    print("  /mcp show fidelis shows its status and the tools it exposes")
    print("  Copilot may ask you to approve the fidelis tools on first use")
    return 0


def _cmd_copilot_uninstall(args) -> int:
    config_path = copilot_config_path(getattr(args, "settings", None))
    if not config_path.exists():
        print(f"no Copilot CLI config at {config_path}; nothing to uninstall")
        return 0

    config, error = _load_json_object(config_path)
    if error:
        print(error, file=sys.stderr)
        return 1

    mcp_servers = config.get("mcpServers")
    if not isinstance(mcp_servers, dict) or MCP_SERVER_NAME not in mcp_servers:
        print(f"no '{MCP_SERVER_NAME}' MCP server registered in {config_path}")
        return 0
    if not _is_fidelis_copilot_entry(mcp_servers[MCP_SERVER_NAME]) and not getattr(args, "force", False):
        print(
            f"error: Copilot CLI MCP server '{MCP_SERVER_NAME}' in {config_path} does not appear "
            "to belong to Fidelis; refusing to remove it\n"
            f"  entry: {json.dumps(mcp_servers[MCP_SERVER_NAME])}\n"
            "  use --force to remove it anyway.",
            file=sys.stderr,
        )
        return 1

    print(f"backed up to {_backup(config_path)}")
    del mcp_servers[MCP_SERVER_NAME]
    _atomic_write_json(config_path, config)
    print(f"removed '{MCP_SERVER_NAME}' MCP server from {config_path}")
    return 0


# ---------------------------------------------------------------------------
# Google Gemini CLI
#
# Gemini CLI has a native MCP management surface — `gemini mcp add|remove|list`,
# shipped since v0.1.19 (google-gemini/gemini-cli#5481). Every write goes
# through it, for one concrete reason: Gemini reads settings.json as JSON *with
# comments* (`JSON.parse(stripJsonComments(content))` in
# packages/cli/src/config/settings.ts) and its own writer round-trips a user's
# `//` and `/* */` comments. A rewrite of that file by Fidelis would silently
# delete them. `gemini mcp add` also leaves every unrelated server and settings
# key alone and preserves the file's permission bits.
#
# What the native surface does NOT give us:
#   - there is no `gemini mcp get`, and `gemini mcp list` has no `--json`
#   - `gemini mcp list` merges user + project + extension scopes into one
#     ANSI-coloured line per server, space-joining command and args (so an
#     argument containing a space is unrecoverable), and it opens a transport
#     to every configured server to report liveness
#   - `gemini mcp add` overwrites an existing entry of the same name without
#     being asked (add.ts), and `gemini mcp remove` exits 0 when the name is
#     absent (remove.ts) — neither exit code proves what actually changed
#
# So ownership, no-op detection, and post-write verification read the exact
# scope's settings.json directly, read-only, through a JSONC-tolerant parser
# that mirrors Gemini's own. Fidelis never writes that file itself.
# ---------------------------------------------------------------------------

GEMINI_DIR_NAME = ".gemini"
GEMINI_SETTINGS_NAME = "settings.json"
GEMINI_SCOPES = ("user", "project")

# `gemini mcp add|remove|list` first shipped in v0.1.19. Below that there is no
# native surface to delegate to, and we will not hand-edit settings.json.
GEMINI_MCP_MIN_VERSION = (0, 1, 19)

# Gemini CLI refuses every subcommand, `mcp` included, until an auth method is
# configured, and exits with this code.
GEMINI_AUTH_EXIT_CODE = 41

# Gemini's stdout/stderr and the entries inside settings.json are untrusted
# input. We echo them for diagnostics, so they are bounded first.
UNTRUSTED_ECHO_LIMIT = 2000

_GEMINI_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _trim(text: str) -> str:
    """Bound untrusted text before echoing it into our own diagnostics."""
    text = text.strip()
    if len(text) <= UNTRUSTED_ECHO_LIMIT:
        return text
    return text[:UNTRUSTED_ECHO_LIMIT] + " ...(truncated)"


# Scalar fields of an MCP client entry that are never a credential by
# themselves -- an executable name/path, a transport tag, a flag. Anything
# else, ``args``/``arguments`` included, is withheld: a launch argument is
# exactly where a bare ``--api-key <token>`` or a token-bearing URL lives, so
# being a launch field does not make a value safe to print.
_SAFE_ENTRY_SCALAR_FIELDS = ("command", "type", "transport", "enabled")
_ENTRY_ARGUMENT_LIST_FIELDS = ("args", "arguments")
# A value is only ever shown outright when it is one of these JSON scalar
# types. Neither Gemini's settings.json reader nor OpenClaw's own config
# validates what an entry (or a "safe" field within it) actually contains --
# a hand-edited or malformed config can put a credential-bearing string
# directly where a dict, or a nested object, is expected.
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def _safe_entry_summary(entry: object) -> str:
    """Render an untrusted client MCP entry for a diagnostic without leaking
    secrets.

    A refused or unexpected entry is echoed back so the user can recognize
    it, but the entry can carry a credential almost anywhere: an API token
    in ``env``, an ``Authorization`` header, a signed ``url``, a bare token
    passed as one of its own launch arguments -- or, since neither reader
    validates an entry's shape, a credential sitting directly where a dict
    was expected (the entry itself, or one of its "safe" fields). Only a
    fixed set of small, structural scalars is ever printed outright, and
    only once confirmed to actually be a scalar; every argument value --
    and every other field -- is named, so nothing is silently missing, but
    its value never is."""
    if not isinstance(entry, dict):
        return f"<non-object entry ({type(entry).__name__}), withheld>"
    safe: dict[str, object] = {}
    withheld: list[str] = []
    for key in _SAFE_ENTRY_SCALAR_FIELDS:
        if key not in entry:
            continue
        value = entry[key]
        if isinstance(value, _JSON_SCALAR_TYPES):
            safe[key] = value
        else:
            withheld.append(key)
    for list_field in _ENTRY_ARGUMENT_LIST_FIELDS:
        if list_field not in entry:
            continue
        values = entry[list_field]
        if isinstance(values, list):
            safe[list_field] = f"<{len(values)} argument(s) withheld>"
        else:
            withheld.append(list_field)
    shown = set(_SAFE_ENTRY_SCALAR_FIELDS) | set(_ENTRY_ARGUMENT_LIST_FIELDS)
    withheld.extend(key for key in entry if key not in shown)
    summary = _trim(json.dumps(safe))
    if withheld:
        summary += f" (withheld: {', '.join(sorted(set(withheld)))})"
    return summary


def _strip_json_comments(text: str) -> str:
    """Blank out `//` and `/* */` comments outside string literals.

    Mirrors the `strip-json-comments` pass Gemini CLI runs before
    `JSON.parse`, so we accept exactly the files Gemini accepts — including
    a commented settings.json, and excluding trailing commas, which Gemini
    rejects too. Newlines inside a block comment are kept so a decoder error
    still reports the line the user has to fix."""
    out: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    while index < length:
        char = text[index]
        if in_string:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "/":
            while index < length and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and index + 1 < length and text[index + 1] == "*":
            end = text.find("*/", index + 2)
            chunk = text[index:] if end == -1 else text[index : end + 2]
            out.append("\n" * chunk.count("\n"))
            index = length if end == -1 else end + 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def gemini_settings_path(scope: str) -> Path:
    """Resolve the settings.json that `gemini mcp --scope <scope>` writes.

    Mirrors Gemini's own Storage helpers: user scope is
    ``~/.gemini/settings.json`` (getGlobalSettingsPath), project scope is
    ``<cwd>/.gemini/settings.json`` (getWorkspaceSettingsPath, whose target
    directory is the process working directory)."""
    base = Path.home() if scope == "user" else Path.cwd()
    return base / GEMINI_DIR_NAME / GEMINI_SETTINGS_NAME


def _project_scope_collides_with_user(scope: str) -> bool:
    """Whether `--scope project` resolves to the user settings file.

    Gemini treats a workspace whose real path is the home directory as the
    user scope (Storage.isWorkspaceHomeDir). Running the project scope there
    would write one file while claiming to write another, so we refuse it
    rather than report an ambiguous result."""
    if scope != "project":
        return False
    try:
        return Path.cwd().resolve() == Path.home().resolve()
    except OSError:
        return False


def gemini_server_entry() -> dict:
    """The exact stdio entry `gemini mcp add` writes for Fidelis.

    Passing no --env/--timeout/--trust/--description flags makes Gemini write
    command and args and nothing else."""
    return {"command": sys.executable, "args": [str(MCP_SERVER_FILE)]}


def _read_gemini_servers(path: Path) -> tuple[dict | None, str | None]:
    """Read one scope's mcpServers block. Returns ``({}, None)`` when the file
    or the block is absent.

    The file is untrusted and hand-editable: a non-object document, a
    non-object mcpServers, or anything Gemini's own parser would reject is an
    error we surface, never something to write over."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return {}, None
    except OSError as exc:
        return None, f"error: could not read {path}: {exc}"
    except UnicodeDecodeError as exc:
        # ``UnicodeDecodeError`` is a ``ValueError``, not an ``OSError``: a
        # settings file that is not valid text must surface like any other
        # unreadable file instead of escaping install/uninstall as a traceback.
        return None, f"error: {path} is not valid text: {exc}"
    try:
        data = json.loads(_strip_json_comments(raw))
    except json.JSONDecodeError as exc:
        return None, (
            f"error: {path} is not valid Gemini settings JSON: {exc}\n"
            "  Gemini CLI reads this file as JSON-with-comments and rejects it too\n"
            "  (a trailing comma is the usual cause). Fix it, then rerun."
        )
    if not isinstance(data, dict):
        return None, f"error: {path} must contain a JSON object at the top level"
    servers = data.get("mcpServers")
    if servers is None:
        return {}, None
    if not isinstance(servers, dict):
        return None, f"error: 'mcpServers' in {path} is not a JSON object"
    return servers, None


def _is_fidelis_gemini_entry(entry: object) -> bool:
    """Return whether a Gemini MCP entry launches a Fidelis MCP server."""
    if not isinstance(entry, dict):
        return False
    args = entry.get("args")
    if not isinstance(args, list) or len(args) != 1:
        return False
    return _is_fidelis_server_path(args[0])


def _gemini_entry_matches(entry: object) -> bool:
    """Return whether a read-back entry is exactly the launch we asked for.

    Stricter than ownership: this is what proves the write landed, so the
    interpreter path must match this install, not merely some Fidelis one."""
    if not isinstance(entry, dict):
        return False
    if str(entry.get("command", "")) != sys.executable:
        return False
    args = entry.get("args")
    if not isinstance(args, list):
        return False
    return [str(value) for value in args] == [str(MCP_SERVER_FILE)]


def _gemini_version(gemini_bin: str) -> tuple[tuple[int, ...] | None, bool]:
    """Read `gemini --version` as ``(version, answered)``.

    `--version` is answered before the auth check, so this works on a host
    that has never signed in. It does *not* survive a broken settings.json:
    Gemini exits non-zero there, which is why a failed probe is reported as
    ``answered=False`` and left alone — the caller reads the same file a
    moment later and can say what is actually wrong with it."""
    try:
        result = subprocess.run(
            [gemini_bin, "--version"], capture_output=True, text=True, check=False
        )
    except OSError:
        return None, False
    if result.returncode != 0:
        return None, False
    match = _GEMINI_VERSION_RE.search(result.stdout)
    if not match:
        return None, True
    return tuple(int(part) for part in match.groups()), True


def _gemini_failure(result) -> str:
    """Render a failed `gemini mcp` run, with the auth hint when that is why."""
    message = _trim(result.stderr) or _trim(result.stdout) or "error: gemini mcp command failed"
    blob = f"{result.stdout}\n{result.stderr}"
    if result.returncode == GEMINI_AUTH_EXIT_CODE or "set an Auth method" in blob:
        message += (
            "\n  Gemini CLI refuses every `gemini mcp` subcommand until an auth method is set.\n"
            "  Run `gemini` once and sign in, or export GEMINI_API_KEY, then rerun."
        )
    return message


def _explicit_gemini_scope(args) -> str | None:
    """The --scope the caller actually passed, or None.

    argparse hands us None or one of its two choices. Anything else — an
    absent attribute, a hand-built Namespace, a test double whose attributes
    autovivify — counts as not passed, so it can never be mistaken for a
    deliberate scope selection."""
    scope = getattr(args, "scope", None)
    return scope if scope in GEMINI_SCOPES else None


def _gemini_preflight(args) -> tuple[str, str, Path] | None:
    """Resolve the Gemini binary, scope, and settings path, or explain why not."""
    if getattr(args, "settings", None):
        print(
            "error: --settings is not supported for the Gemini CLI client; use "
            "--scope user|project, which selects the settings.json that "
            "`gemini mcp` itself writes",
            file=sys.stderr,
        )
        return None

    scope = _explicit_gemini_scope(args) or "user"

    gemini_bin = shutil.which("gemini")
    if not gemini_bin:
        print(
            "error: Gemini CLI not found on PATH\n"
            "  install it (npm install -g @google/gemini-cli), then rerun: "
            "fidelis mcp install --client gemini",
            file=sys.stderr,
        )
        return None

    version, answered = _gemini_version(gemini_bin)
    minimum = ".".join(str(part) for part in GEMINI_MCP_MIN_VERSION)
    if version is not None and version < GEMINI_MCP_MIN_VERSION:
        print(
            f"error: Gemini CLI {'.'.join(str(part) for part in version)} has no "
            "`gemini mcp` subcommand\n"
            f"  `gemini mcp add|remove|list` first shipped in v{minimum}; upgrade and rerun.",
            file=sys.stderr,
        )
        return None
    if version is None and answered:
        print(
            "warning: could not read `gemini --version`; continuing. "
            f"`gemini mcp` needs v{minimum} or newer and will fail loudly below if it is older.",
            file=sys.stderr,
        )

    if _project_scope_collides_with_user(scope):
        print(
            "error: --scope project resolves to the user settings file in your home "
            f"directory ({gemini_settings_path('user')})\n"
            "  rerun with --scope user, or change into a project directory first.",
            file=sys.stderr,
        )
        return None

    return gemini_bin, scope, gemini_settings_path(scope)


def _cmd_gemini_install(args) -> int:
    if not MCP_SERVER_FILE.exists():
        print(
            f"error: bundled MCP server not found at {MCP_SERVER_FILE}\n"
            "  this install appears incomplete; reinstall Hermes Labs Fidelis "
            "from its tagged GitHub source (see README)",
            file=sys.stderr,
        )
        return 1

    preflight = _gemini_preflight(args)
    if preflight is None:
        return 1
    gemini_bin, scope, config_path = preflight

    servers, error = _read_gemini_servers(config_path)
    if error:
        print(error, file=sys.stderr)
        return 1

    existing = servers.get(MCP_SERVER_NAME)
    if existing is not None:
        if _gemini_entry_matches(existing):
            print(
                f"Gemini CLI MCP server '{MCP_SERVER_NAME}' is already configured in "
                f"{config_path} ({scope} scope); nothing to change"
            )
            return 0
        if not _is_fidelis_gemini_entry(existing) and not args.force:
            print(
                f"error: a non-fidelis Gemini CLI MCP server named '{MCP_SERVER_NAME}' already "
                f"exists in {config_path} ({scope} scope)\n"
                f"  entry: {_safe_entry_summary(existing)}\n"
                "  refusing to overwrite. Use --force to replace it.",
                file=sys.stderr,
            )
            return 1

    result = subprocess.run(
        [
            gemini_bin, "mcp", "add", MCP_SERVER_NAME,
            sys.executable, str(MCP_SERVER_FILE),
            "--scope", scope,
            "--transport", "stdio",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(_gemini_failure(result), file=sys.stderr)
        return result.returncode

    # `gemini mcp add` exits 0 whether it created, replaced, or (in a future
    # version) declined to write. Only the file proves what happened.
    servers, error = _read_gemini_servers(config_path)
    if error:
        print(
            "error: gemini mcp add reported success but its config no longer reads back\n"
            f"{error}",
            file=sys.stderr,
        )
        return 1
    written = servers.get(MCP_SERVER_NAME)
    if written is None:
        print(
            f"error: gemini mcp add reported success but no '{MCP_SERVER_NAME}' server is "
            f"present in {config_path} ({scope} scope); nothing was installed",
            file=sys.stderr,
        )
        return 1
    if not _gemini_entry_matches(written):
        print(
            f"error: gemini mcp add wrote an unexpected '{MCP_SERVER_NAME}' entry to "
            f"{config_path} ({scope} scope)\n"
            f"  expected: {json.dumps(gemini_server_entry())}\n"
            f"  found:    {_safe_entry_summary(written)}",
            file=sys.stderr,
        )
        return 1

    print(_trim(result.stdout) or f"registered Gemini CLI MCP server '{MCP_SERVER_NAME}'")
    print(f"verified '{MCP_SERVER_NAME}' in {config_path} ({scope} scope)")
    print()
    print("next: restart Gemini CLI, or run /mcp reload in an open session")
    print("  gemini mcp list shows the server and whether it connects")
    return 0


def _cmd_gemini_uninstall(args) -> int:
    preflight = _gemini_preflight(args)
    if preflight is None:
        return 1
    gemini_bin, scope, config_path = preflight

    servers, error = _read_gemini_servers(config_path)
    if error:
        print(error, file=sys.stderr)
        return 1

    existing = servers.get(MCP_SERVER_NAME)
    if existing is None:
        print(
            f"no '{MCP_SERVER_NAME}' MCP server registered in {config_path} "
            f"({scope} scope); nothing to uninstall"
        )
        return 0
    if not _is_fidelis_gemini_entry(existing) and not getattr(args, "force", False):
        print(
            f"error: Gemini CLI MCP server '{MCP_SERVER_NAME}' in {config_path} "
            f"({scope} scope) does not appear to belong to Fidelis; refusing to remove it\n"
            f"  entry: {_safe_entry_summary(existing)}\n"
            "  use --force to remove it anyway.",
            file=sys.stderr,
        )
        return 1

    result = subprocess.run(
        [gemini_bin, "mcp", "remove", MCP_SERVER_NAME, "--scope", scope],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(_gemini_failure(result), file=sys.stderr)
        return result.returncode

    # `gemini mcp remove` exits 0 for an absent name too, so the exit code
    # cannot distinguish a removal from a no-op. Read the file back.
    servers, error = _read_gemini_servers(config_path)
    if error:
        print(
            "error: gemini mcp remove reported success but its config no longer reads back\n"
            f"{error}",
            file=sys.stderr,
        )
        return 1
    if MCP_SERVER_NAME in servers:
        print(
            f"error: gemini mcp remove exited 0 but '{MCP_SERVER_NAME}' is still present in "
            f"{config_path} ({scope} scope); nothing was removed",
            file=sys.stderr,
        )
        return 1

    print(_trim(result.stdout) or f"removed Gemini CLI MCP server '{MCP_SERVER_NAME}'")
    print(f"verified '{MCP_SERVER_NAME}' is gone from {config_path} ({scope} scope)")
    return 0


# ---------------------------------------------------------------------------
# OpenClaw
#
# OpenClaw reads an optional JSON5 config from ``~/.openclaw/openclaw.json``
# and keeps outbound MCP servers under ``mcp.servers.<name>``. Because that
# file is JSON5 -- comments and trailing commas are supported and common --
# Fidelis neither writes it nor parses it. OpenClaw's own read-only CLI is the
# authority on what is in it:
#
#     openclaw mcp show fidelis --json   # that server's definition, or exit 1
#     openclaw mcp list --json           # the whole mcp.servers map
#
# and its documented write path owns every mutation:
#
#     openclaw mcp add local-tools --command node --arg ./dist/mcp-server.js
#     openclaw mcp unset local-tools
#
# ``$OPENCLAW_CONFIG_PATH`` is the documented way to point OpenClaw at a
# specific config file. Fidelis pins it for every delegated call -- the reads
# as well as the write -- so the state it reads back is provably the state of
# the file the CLI just wrote.
# ---------------------------------------------------------------------------


def openclaw_config_path(settings: str | None = None) -> Path:
    """Resolve the OpenClaw config file.

    Explicit ``settings`` wins, then ``$OPENCLAW_CONFIG_PATH``, then the
    documented default ``~/.openclaw/openclaw.json``."""
    if settings:
        return Path(settings).expanduser()
    configured = os.environ.get("OPENCLAW_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".openclaw" / OPENCLAW_CONFIG_NAME


def openclaw_add_arguments() -> list[str]:
    """The exact documented ``openclaw mcp add`` arguments Fidelis delegates."""
    return [
        "mcp",
        "add",
        MCP_SERVER_NAME,
        "--command",
        sys.executable,
        "--arg",
        str(MCP_SERVER_FILE),
    ]


def openclaw_show_arguments() -> list[str]:
    """The documented read-only call that answers what our entry is."""
    return ["mcp", "show", MCP_SERVER_NAME, "--json"]


def openclaw_list_arguments() -> list[str]:
    """The documented read-only call that answers which entries exist."""
    return ["mcp", "list", "--json"]


def _openclaw_cli() -> str | None:
    return shutil.which("openclaw")


# The fields naming an OpenClaw entry's launch arguments, per
# ``openclaw mcp add --command/--arg`` (see ``openclaw_add_arguments``).
_OPENCLAW_LAUNCH_LIST_KEYS = ("args", "arguments")


# An interpreter's own basename, optionally versioned (`python3`,
# `pypy3.10`) and optionally free-threaded (`python3.14t`) or `.exe`-suffixed.
# Anchored on both ends so a real but unrelated tool that merely contains one
# of these names -- `python-config`, `python3-analyzer` -- does not match.
_INTERPRETER_NAME_RE = re.compile(r"^(python|pypy)(2|3)?(\.\d+){0,2}t?(\.exe)?$", re.IGNORECASE)


def _looks_like_python_interpreter(command: str) -> bool:
    """Whether `command` is -- or, by name, plausibly is -- a Python
    interpreter.

    A ``.py`` entry point has no other way to run, so this is what tells
    "an interpreter launching our script" apart from "an unrelated tool
    that merely accepts our script's path as its one input". This exact
    process's own interpreter always counts, whatever it is named --
    that is the one Fidelis itself has ever actually written via
    ``openclaw mcp add --command``. Beyond that, the executable's own name
    must match a known interpreter shape exactly, not merely contain one:
    a differently-rooted venv, a pyenv shim, or a pipx-managed Python from
    a past install are still recognized (CPython and PyPy binaries are
    named for it wherever they are installed), but a real tool that just
    happens to have "python" in its name, and is not itself one, is not."""
    try:
        if Path(command).expanduser().resolve() == Path(sys.executable).resolve():
            return True
    except (OSError, ValueError, RuntimeError):
        pass
    return bool(_INTERPRETER_NAME_RE.match(Path(command).name))


def _mentions_fidelis_server(node: object) -> bool:
    """Whether an entry's command and argument *layout* is our own launch,
    not merely a value somewhere that happens to match it.

    Fidelis's own ``openclaw mcp add`` writes exactly one shape: a Python
    interpreter in ``command`` and a single positional script argument.
    Checking the script argument alone is not enough to prove that -- an
    entry's metadata (``env``, ``headers``, a remote ``url``) is user data
    that can legitimately hold our script path as a *value* without the
    entry being ours to launch, and so can a single argument on its own:
    an unrelated executable can accept our script path as its one input
    (e.g. reading it, not running it) while ``command`` names something
    else entirely. Only that whole shape -- a Python-named ``command`` with
    exactly one argument naming our script -- is what Fidelis itself ever
    writes. Matching a looser one would let a foreign server be recognized
    as Fidelis's own, and then get installed over or uninstalled without
    ``--force``. A shape that doesn't match it is never ours -- it fails
    closed as foreign, not as a false match."""
    if not isinstance(node, dict):
        return False
    command = node.get("command")
    if not isinstance(command, str) or not _looks_like_python_interpreter(command):
        return False
    args: list | None = None
    for list_key in _OPENCLAW_LAUNCH_LIST_KEYS:
        values = node.get(list_key)
        if isinstance(values, list):
            args = values
            break
    if args is None or len(args) != 1 or not isinstance(args[0], str):
        return False
    # The value is still user data -- path resolution on it can raise
    # (embedded NUL, an unresolvable ``~user``); that just means "not our
    # path".
    try:
        return _is_fidelis_server_path(args[0])
    except (OSError, ValueError, RuntimeError):
        return False


# What OpenClaw says about ``mcp.servers.fidelis``.
_OC_ABSENT = "absent"        # no config file, or no entry under that name
_OC_OURS = "ours"            # an entry naming our packaged MCP server
_OC_FOREIGN = "foreign"      # an entry named 'fidelis' that is not ours
_OC_UNKNOWN = "unknown"      # OpenClaw could not tell us; never assume absent


def _openclaw_error_message(payload: object) -> str | None:
    """The message inside OpenClaw's documented CLI failure envelope.

    Every ``--json`` command prints ``{"ok": false, "error": {"type":
    "cli_error", "message": ...}}`` on stdout when it fails. Returns ``None``
    when ``payload`` is not such an envelope, which is what tells a real answer
    apart from a reported failure."""
    if not isinstance(payload, dict) or payload.get("ok") is not False:
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return ""
    return str(error.get("message", ""))


def _run_openclaw(openclaw_bin: str, arguments: list[str], config_path: Path):
    env = dict(os.environ)
    env["OPENCLAW_CONFIG_PATH"] = str(config_path)
    return subprocess.run(
        [openclaw_bin, *arguments],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _openclaw_json(openclaw_bin: str, arguments: list[str], config_path: Path):
    """Run a read-only openclaw command and parse its ``--json`` stdout."""
    result = _run_openclaw(openclaw_bin, arguments, config_path)
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        payload = None
    return result, payload


def _openclaw_state(openclaw_bin: str, config_path: Path) -> tuple[str, object, str]:
    """Ask OpenClaw what ``mcp.servers.fidelis`` currently is.

    Fidelis never parses the config itself. JSON5 is a superset of JSON, so a
    config carrying comments or a trailing comma is unreadable to
    ``json.loads`` while being perfectly readable to OpenClaw -- and a check
    that cannot read the file is exactly how a user's foreign ``fidelis`` entry
    gets silently overwritten, or a write that never landed gets reported as a
    success. OpenClaw's own read-only surface has no such blind spot.

    ``mcp show <name> --json`` prints the server definition and exits 0, but it
    exits 1 with the same failure envelope whether the server is merely not
    configured or the config could not be loaded at all. Those two answers are
    not interchangeable, so a non-zero ``show`` is resolved by ``mcp list
    --json``, which exits 0 with the whole ``mcp.servers`` map for any config
    OpenClaw can read. Whatever is left over is ``_OC_UNKNOWN`` -- never
    ``_OC_ABSENT``, and never a success.

    Returns the state, the entry behind it when there is one, and a diagnostic
    line to show the user when the state is ``_OC_UNKNOWN``."""
    if not config_path.exists():
        # No file, no saved servers. That is a fact about the filesystem rather
        # than a parse of a JSON5 document, and ``$OPENCLAW_CONFIG_PATH`` pins
        # this exact path for every delegated call, so it is the file OpenClaw
        # itself would read.
        return _OC_ABSENT, None, ""

    result, payload = _openclaw_json(openclaw_bin, openclaw_show_arguments(), config_path)
    if (
        result.returncode == 0
        and isinstance(payload, dict)
        and payload
        and _openclaw_error_message(payload) is None
    ):
        state = _OC_OURS if _mentions_fidelis_server(payload) else _OC_FOREIGN
        return state, payload, ""
    show_detail = _openclaw_error_message(payload) or result.stderr.strip()

    result, payload = _openclaw_json(openclaw_bin, openclaw_list_arguments(), config_path)
    if (
        result.returncode != 0
        or not isinstance(payload, dict)
        or _openclaw_error_message(payload) is not None
    ):
        detail = _openclaw_error_message(payload) or result.stderr.strip() or show_detail
        return _OC_UNKNOWN, None, detail
    if MCP_SERVER_NAME not in payload:
        return _OC_ABSENT, None, ""
    entry = payload[MCP_SERVER_NAME]
    return (_OC_OURS if _mentions_fidelis_server(entry) else _OC_FOREIGN), entry, ""


def _openclaw_entry_matches(entry: object) -> bool:
    """Whether a read-back OpenClaw entry is exactly the launch we asked for.

    Stricter than ``_mentions_fidelis_server``: ownership only proves the
    entry is Fidelis's, not that this particular install landed. A prior
    install's entry -- a stale interpreter from a since-rebuilt venv, or
    left ``enabled: false`` -- still names our script and so still reads
    back as ``_OC_OURS`` even when ``openclaw mcp add`` silently did
    nothing. This is what catches that: the interpreter, the script
    argument, and that the entry was not left disabled."""
    if not isinstance(entry, dict):
        return False
    if str(entry.get("command", "")) != sys.executable:
        return False
    args = entry.get("args")
    if not isinstance(args, list) or [str(value) for value in args] != [str(MCP_SERVER_FILE)]:
        return False
    return entry.get("enabled", True) is not False


def _openclaw_missing_cli(action: str) -> int:
    print(
        "error: OpenClaw CLI not found on PATH\n"
        "  OpenClaw owns every write to its JSON5 config, and is the only thing\n"
        "  that can read one back, so its CLI is required.\n"
        f"  install OpenClaw, then rerun: fidelis mcp {action} --client openclaw",
        file=sys.stderr,
    )
    return 1


def _openclaw_unknown_state(config_path: Path, detail: str, action: str) -> int:
    """Refuse to act on a config OpenClaw could not report on."""
    print(
        "\n".join(
            [
                f"error: could not confirm what '{MCP_SERVER_NAME}' is in {config_path}",
                f"  openclaw could not report it{': ' + detail if detail else '.'}",
                f"  an existing entry cannot be ruled out, so Fidelis will not {action} blindly.",
                "  fix the config (openclaw doctor), or pass --force to proceed anyway.",
            ]
        ),
        file=sys.stderr,
    )
    return 1


def _cmd_openclaw_install(args) -> int:
    if not MCP_SERVER_FILE.exists():
        print(
            f"error: bundled MCP server not found at {MCP_SERVER_FILE}\n"
            "  this install appears incomplete; reinstall Hermes Labs Fidelis "
            "from its tagged GitHub source (see README)",
            file=sys.stderr,
        )
        return 1

    openclaw_bin = _openclaw_cli()
    if not openclaw_bin:
        return _openclaw_missing_cli("install")

    config_path = openclaw_config_path(getattr(args, "settings", None))
    state, existing, detail = _openclaw_state(openclaw_bin, config_path)
    if state == _OC_FOREIGN and not args.force:
        print(
            f"error: a non-fidelis OpenClaw MCP server named '{MCP_SERVER_NAME}' already exists "
            f"in {config_path}\n"
            f"  entry: {_safe_entry_summary(existing)}\n"
            "  refusing to overwrite. Use --force to replace it.",
            file=sys.stderr,
        )
        return 1
    if state == _OC_UNKNOWN and not args.force:
        return _openclaw_unknown_state(config_path, detail, "register")

    result = _run_openclaw(openclaw_bin, openclaw_add_arguments(), config_path)
    if result.returncode != 0:
        print(
            result.stderr.strip() or "error: OpenClaw MCP registration failed",
            file=sys.stderr,
        )
        return result.returncode

    # A zero exit from the CLI is a claim, not a result. Only a read-back
    # through OpenClaw's own surface proves the entry landed.
    after, entry, detail = _openclaw_state(openclaw_bin, config_path)
    if after == _OC_UNKNOWN:
        print(
            "error: openclaw reported success but the registration could not be confirmed "
            f"in {config_path}\n"
            f"  openclaw could not report it{': ' + detail if detail else '.'}\n"
            f"  check with: openclaw mcp show {MCP_SERVER_NAME} --json",
            file=sys.stderr,
        )
        return 1
    if after != _OC_OURS:
        print(
            f"error: openclaw reported success but no '{MCP_SERVER_NAME}' server naming "
            f"{MCP_SERVER_FILE} is present in {config_path}",
            file=sys.stderr,
        )
        return 1
    if not _openclaw_entry_matches(entry):
        # Ownership alone is not enough: a pre-existing entry can already
        # name our script and so already read back as ours, while still
        # being the stale registration `add` was supposed to replace (a
        # silent no-op, per test_install_leaves_a_json5_config_untouched...
        # for the config-untouched case; this is the same failure for a
        # config that already has a stale entry under our name).
        print(
            f"error: openclaw reported success but the '{MCP_SERVER_NAME}' entry in "
            f"{config_path} does not match what Fidelis requested\n"
            f"  expected: {json.dumps({'command': sys.executable, 'args': [str(MCP_SERVER_FILE)]})}\n"
            f"  found:    {_safe_entry_summary(entry)}",
            file=sys.stderr,
        )
        return 1

    print(f"registered OpenClaw MCP server '{MCP_SERVER_NAME}' in {config_path}")
    print()
    print("next: openclaw mcp reload            # pick up the new server")
    print("  openclaw mcp status --verbose      # confirm the saved config")
    print(f"  openclaw mcp doctor {MCP_SERVER_NAME} --probe   # verify it connects")
    return 0


def _cmd_openclaw_uninstall(args) -> int:
    config_path = openclaw_config_path(getattr(args, "settings", None))
    if not config_path.exists():
        print(f"no OpenClaw config at {config_path}; nothing to uninstall")
        return 0

    openclaw_bin = _openclaw_cli()
    if not openclaw_bin:
        return _openclaw_missing_cli("uninstall")

    force = getattr(args, "force", False)
    state, existing, detail = _openclaw_state(openclaw_bin, config_path)
    if state == _OC_ABSENT:
        print(f"no '{MCP_SERVER_NAME}' MCP server registered in {config_path}")
        return 0
    if state == _OC_FOREIGN and not force:
        print(
            f"error: OpenClaw MCP server '{MCP_SERVER_NAME}' in {config_path} does not appear "
            "to belong to Fidelis; refusing to remove it\n"
            f"  entry: {_safe_entry_summary(existing)}\n"
            "  use --force to remove it anyway.",
            file=sys.stderr,
        )
        return 1
    if state == _OC_UNKNOWN and not force:
        return _openclaw_unknown_state(config_path, detail, "remove")

    # `unset` is the documented subcommand for removing an OpenClaw-managed
    # mcp.servers entry, and it fails when the named server does not exist. A
    # zero exit is still only a claim: the read-back below is what proves the
    # entry is gone.
    result = _run_openclaw(
        openclaw_bin, ["mcp", "unset", MCP_SERVER_NAME], config_path
    )
    if result.returncode != 0:
        print(
            result.stderr.strip() or "error: OpenClaw MCP removal failed",
            file=sys.stderr,
        )
        return result.returncode

    after, _, detail = _openclaw_state(openclaw_bin, config_path)
    if after == _OC_ABSENT:
        print(f"removed '{MCP_SERVER_NAME}' MCP server from {config_path}")
        return 0
    if after == _OC_UNKNOWN:
        print(
            "error: openclaw reported success but the removal could not be confirmed "
            f"in {config_path}\n"
            f"  openclaw could not report it{': ' + detail if detail else '.'}\n"
            f"  check with: openclaw mcp show {MCP_SERVER_NAME} --json",
            file=sys.stderr,
        )
        return 1
    print(
        f"error: openclaw reported success but '{MCP_SERVER_NAME}' is still registered in "
        f"{config_path}",
        file=sys.stderr,
    )
    return 1


def _reject_scope(client: str) -> bool:
    """--scope selects a Gemini settings file; it means nothing elsewhere."""
    if client == "gemini":
        return False
    print(
        "error: --scope is only supported for the Gemini CLI client; "
        "Claude Code, Cursor, Copilot CLI, and OpenClaw use --settings",
        file=sys.stderr,
    )
    return True


def cmd_mcp_install(args) -> int:
    client = getattr(args, "client", "claude")
    if _explicit_gemini_scope(args) and _reject_scope(client):
        return 1
    if client == "codex":
        return _cmd_codex_install(args)
    if client == "copilot":
        return _cmd_copilot_install(args)
    if client == "gemini":
        return _cmd_gemini_install(args)
    if client == "openclaw":
        return _cmd_openclaw_install(args)
    if client == "cursor":
        return _cmd_cursor_install(args)

    settings_path = Path(args.settings).expanduser() if args.settings else DEFAULT_SETTINGS

    if not MCP_SERVER_FILE.exists():
        print(
            f"error: bundled MCP server not found at {MCP_SERVER_FILE}\n"
            "  this install appears incomplete; reinstall Hermes Labs Fidelis "
            "from its tagged GitHub source (see README)",
            file=sys.stderr,
        )
        return 1

    # Load or initialize settings
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError as e:
            print(f"error: {settings_path} is not valid JSON: {e}", file=sys.stderr)
            return 1

        # Backup before edit
        print(f"backed up existing settings to {_backup(settings_path)}")
    else:
        settings = {}
        settings_path.parent.mkdir(parents=True, exist_ok=True)

    mcp_servers = settings.setdefault("mcpServers", {})

    # Refuse to overwrite a non-fidelis entry under the fidelis name. Look at
    # both command and args — our own previous install puts the fidelis path in
    # args, so we must inspect both to recognize ourselves.
    existing = mcp_servers.get(MCP_SERVER_NAME)
    if existing and not args.force:
        existing_cmd = existing.get("command", "")
        existing_args = " ".join(existing.get("args", []))
        existing_blob = f"{existing_cmd} {existing_args}"
        if "fidelis" not in existing_blob and "mcp_server.py" not in existing_blob:
            print(
                f"error: an entry named '{MCP_SERVER_NAME}' already exists in mcpServers\n"
                f"  command: {existing_cmd}\n"
                f"  args: {existing.get('args', [])}\n"
                f"  refusing to overwrite. Use --force to replace, or pick a different name.",
                file=sys.stderr,
            )
            return 1

    python_bin = sys.executable
    mcp_servers[MCP_SERVER_NAME] = {
        "command": python_bin,
        "args": [str(MCP_SERVER_FILE)],
    }
    _atomic_write_json(settings_path, settings)
    print(f"wrote MCP server '{MCP_SERVER_NAME}' to {settings_path}")
    print()
    print("next: restart Claude Code to pick up the new MCP server")
    print(f"  the fidelis tools will appear under the prefix mcp__{MCP_SERVER_NAME}__*")
    return 0


def cmd_mcp_uninstall(args) -> int:
    client = getattr(args, "client", "claude")
    if _explicit_gemini_scope(args) and _reject_scope(client):
        return 1
    if client == "gemini":
        return _cmd_gemini_uninstall(args)
    if client == "codex":
        if getattr(args, "settings", None):
            print(
                "error: --settings is only supported for the Claude Code, Copilot CLI, and OpenClaw clients; "
                "Codex uses its shared config through the codex mcp CLI",
                file=sys.stderr,
            )
            return 1
        return _cmd_codex_uninstall()
    if client == "copilot":
        return _cmd_copilot_uninstall(args)
    if client == "openclaw":
        return _cmd_openclaw_uninstall(args)
    if client == "cursor":
        return _cmd_cursor_uninstall(args)

    settings_path = Path(args.settings).expanduser() if args.settings else DEFAULT_SETTINGS

    if not settings_path.exists():
        print(f"no settings file at {settings_path}; nothing to uninstall")
        return 0

    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError as e:
        print(f"error: {settings_path} is not valid JSON: {e}", file=sys.stderr)
        return 1

    mcp_servers = settings.get("mcpServers", {})
    if MCP_SERVER_NAME not in mcp_servers:
        print(f"no '{MCP_SERVER_NAME}' MCP server registered in {settings_path}")
        return 0

    print(f"backed up to {_backup(settings_path)}")

    del mcp_servers[MCP_SERVER_NAME]
    _atomic_write_json(settings_path, settings)
    print(f"removed '{MCP_SERVER_NAME}' MCP server from {settings_path}")
    return 0
