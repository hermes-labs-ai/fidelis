"""Wire fidelis into a supported agent client's MCP configuration.

Claude Code configuration is edited atomically with a backup. Codex
configuration is delegated to the supported ``codex mcp`` CLI so the desktop
app, CLI, and IDE extension share the same registered server.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically: temp file + os.replace. Prevents corruption if
    Claude Code (or any reader) is reading the settings file concurrently.

    Uses parent/(name+".tmp") instead of with_suffix to be safe on Python
    3.10/3.11 where with_suffix raised ValueError on multi-dot suffixes."""
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)

DEFAULT_SETTINGS = Path.home() / ".claude" / "settings.local.json"
MCP_SERVER_NAME = "fidelis"

# Bundled MCP server file lives alongside this module
PACKAGE_DIR = Path(__file__).resolve().parent
MCP_SERVER_FILE = PACKAGE_DIR / "mcp_server.py"


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
            "error: --settings is only supported for the Claude Code client; "
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


def cmd_mcp_install(args) -> int:
    if getattr(args, "client", "claude") == "codex":
        return _cmd_codex_install(args)

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
        backup = settings_path.with_suffix(f".json.bak.{int(time.time())}")
        shutil.copy(settings_path, backup)
        print(f"backed up existing settings to {backup}")
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
    if getattr(args, "client", "claude") == "codex":
        if getattr(args, "settings", None):
            print(
                "error: --settings is only supported for the Claude Code client; "
                "Codex uses its shared config through the codex mcp CLI",
                file=sys.stderr,
            )
            return 1
        return _cmd_codex_uninstall()

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

    backup = settings_path.with_suffix(f".json.bak.{int(time.time())}")
    shutil.copy(settings_path, backup)
    print(f"backed up to {backup}")

    del mcp_servers[MCP_SERVER_NAME]
    _atomic_write_json(settings_path, settings)
    print(f"removed '{MCP_SERVER_NAME}' MCP server from {settings_path}")
    return 0
