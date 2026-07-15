"""fidelis mcp install — wire fidelis into Claude Code's MCP config.

Writes a fidelis MCP server entry into ~/.claude/settings.local.json. Idempotent;
backs up existing settings first; refuses to overwrite if a non-fidelis entry
named "fidelis" already exists.
"""

from __future__ import annotations

import json
import os
import shutil
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


def cmd_mcp_install(args) -> int:
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
