"""fidelis init — install + start the fidelis service so memory is "on" automatically.

Cross-platform:
- macOS: launchd plist at ~/Library/LaunchAgents/ai.hermeslabs.fidelis-server.plist
- Linux: systemd user unit at ~/.config/systemd/user/fidelis-server.service
- Other: fallback to nohup (best-effort, no auto-start on reboot)

Idempotent: re-running install upgrades the unit in place. Uninstall removes
the unit cleanly + stops the service.

Safe upgrades: detects existing services, offers --force to override, --dry-run
to preview, --port/--label to namespace multiple installs, and preserves existing
EnvironmentVariables and config keys on upgrade.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 19420
SERVICE_LABEL = "ai.hermeslabs.fidelis-server"

PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{server_bin}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_path}</string>
    <key>StandardErrorPath</key>
    <string>{log_path}</string>
    <key>ThrottleInterval</key>
    <integer>{throttle_interval}</integer>
    <key>EnvironmentVariables</key>
    <dict>
        <!--
            Telemetry kill. mem0 fires a posthog.capture on every memory
            operation; posthog calls platform.mac_ver() per event, which
            opens /System/Library/CoreServices/SystemVersion.plist. Under
            transient fd pressure (Ollama slowness → request pileup) every
            capture spams an EMFILE error against the plist. Disabling
            mem0's telemetry stops the calls.

            MEM0_TELEMETRY=False   → mem0 (the actual offender; sets
                                      self.posthog = None at import time).
            ANONYMIZED_TELEMETRY=False / CHROMA_TELEMETRY_DISABLED=True
                                   → chromadb (current version's posthog
                                      adapter is a no-op, but kept for
                                      forward compat; cheap).
            PYDANTIC_DISABLE_PLUGINS=__all__
                                   → pydantic v2 plugin system (security gate;
                                      disables untrusted plugin loads at
                                      import time).

            Note: there is no POSTHOG_DISABLED env var in the posthog
            Python SDK. Earlier versions of this template set it, but it
            was inert — see git history for the removal commit.

            The entries below are rendered dynamically from the merged
            env-var dict (existing plist's vars + template defaults) so
            that a re-init preserves customizations instead of silently
            reverting them — see _merge_plist_env_vars.
        -->
{env_vars_xml}
    </dict>
</dict>
</plist>
"""


def _render_env_vars_xml(env_vars: dict) -> str:
    """Render an EnvironmentVariables dict as indented plist <key>/<string> pairs."""
    lines = []
    for key, value in env_vars.items():
        lines.append(f"        <key>{key}</key>")
        lines.append(f"        <string>{value}</string>")
    return "\n".join(lines)

SYSTEMD_TEMPLATE = """[Unit]
Description=Fidelis agent memory server
After=network.target

[Service]
ExecStart={server_bin}
Restart=on-failure
RestartSec=3
StandardOutput=append:{log_path}
StandardError=append:{log_path}
WorkingDirectory={working_dir}
# Telemetry kill — mem0's per-operation posthog.capture opens
# SystemVersion.plist on every event, spamming EMFILE under fd pressure.
# MEM0_TELEMETRY=False is the only flag mem0 honors; the chromadb flags
# are belt-and-suspenders for forward compat.
Environment=MEM0_TELEMETRY=False
Environment=ANONYMIZED_TELEMETRY=False
Environment=CHROMA_TELEMETRY_DISABLED=True
Environment=PYDANTIC_DISABLE_PLUGINS=__all__

[Install]
WantedBy=default.target
"""


def _server_bin() -> str:
    """Locate the fidelis-server entry point installed by pip."""
    bin_path = shutil.which("fidelis-server")
    if bin_path:
        return bin_path
    # Fallback: look in the same dir as the active python
    candidate = Path(sys.executable).parent / "fidelis-server"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError(
        "fidelis-server entry point not found on PATH. "
        "Reinstall Hermes Labs Fidelis from its tagged GitHub source; "
        "see the README installation instructions."
    )


def _health_check(timeout_s: float = 10.0, port: int = None) -> bool:
    """Wait up to timeout_s for /health to return ok."""
    if port is None:
        port = PORT
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


_LEGACY_LABELS = ("ai.hermeslabs.cogito-server", "ai.cogito.server")


def _bootout_legacy_macos(force: bool = False) -> None:
    """Migrate from pre-rename launchd labels. Idempotent.

    Args:
        force: If True, actually unlink legacy plists. If False, only log.
    """
    for label in _LEGACY_LABELS:
        legacy_plist = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
        if legacy_plist.exists():
            subprocess.run(["launchctl", "unload", str(legacy_plist)], check=False)
            if force:
                try:
                    legacy_plist.unlink()
                    print(f"migrated: removed legacy plist {legacy_plist.name}")
                except OSError:  # noqa: silent — best-effort migration
                    pass
            else:
                print(f"note: legacy plist exists at {legacy_plist.name}; pass --migrate to remove it")


def _detect_existing_service(label: str, port: int) -> dict | None:
    """Check if a launchd service with this label or port is already running.

    Returns:
        dict with keys 'label', 'port', 'pid', 'binary' if found, None if not.
    """
    # Check if label is already loaded
    result = subprocess.run(
        ["launchctl", "list", label],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        # Label is loaded. `launchctl list <label>` output is NOT the
        # tabular "PID STATUS LABEL" format used by bare `launchctl list`;
        # for a single label it prints a property-list dict, e.g.:
        #   { "PID" = 74977; "Label" = "..."; "LastExitStatus" = 0; ... }
        # The previous parser assumed the tabular format and did
        # int(result.stdout.split('\n')[0].split()[0]), expecting an int
        # first token — against real launchd output that token is the
        # literal "{", so int() always raised ValueError and the label
        # collision was silently missed whenever the port-listener check
        # (the second, independent signal below) didn't also catch it.
        # Confirmed against the live production service on this host:
        #   launchctl list ai.hermeslabs.fidelis-server -> PID 74977
        # returncode == 0 alone already proves the label is loaded (launchd
        # sets this correctly); PID extraction below is best-effort, used
        # only to name the process in the refusal message — never a gate
        # on whether to refuse.
        pid = None
        match = re.search(r'"PID"\s*=\s*(\d+)', result.stdout)
        if match:
            pid = int(match.group(1))
        return {
            "label": label,
            "port": port,
            "pid": pid,
            "binary": _server_bin(),
            "found_by": "launchctl_list",
        }

    # Check if port is already in use
    result = subprocess.run(
        ["lsof", "-i", f":{port}", "-sTCP:LISTEN"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        # Port is in use. Extract PID.
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:  # Skip header line
            parts = lines[1].split()
            if len(parts) > 1:
                try:
                    pid = int(parts[1])
                    return {
                        "label": label,
                        "port": port,
                        "pid": pid,
                        "binary": "unknown (via lsof)",
                        "found_by": "lsof",
                    }
                except ValueError:
                    pass

    return None


def _merge_plist_env_vars(existing_plist_path: Path, new_env_vars: dict) -> dict:
    """Merge new env vars with existing plist's env vars.

    Preserves existing values, overlays new ones.

    Args:
        existing_plist_path: Path to existing plist file
        new_env_vars: Dict of new env vars from template

    Returns:
        Merged dict (existing + new, with new taking precedence)
    """
    if not existing_plist_path.exists():
        return new_env_vars

    try:
        import plistlib
        existing_data = plistlib.loads(existing_plist_path.read_bytes())
        existing_env = existing_data.get("EnvironmentVariables", {})

        # Merge: start with existing, overlay new (new takes precedence)
        merged = dict(existing_env)
        merged.update(new_env_vars)
        return merged
    except Exception as e:
        print(
            f"warning: could not parse existing plist ({e}); using template env vars",
            file=sys.stderr,
        )
        return new_env_vars


def _install_macos(
    uninstall: bool = False,
    force: bool = False,
    dry_run: bool = False,
    port: int | None = None,
    label: str | None = None,
    migrate_legacy: bool = False,
) -> int:
    if port is None:
        port = PORT
    if label is None:
        label = SERVICE_LABEL

    plist_path = Path.home() / "Library/LaunchAgents" / f"{label}.plist"

    if uninstall:
        if plist_path.exists():
            subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
            plist_path.unlink()
            print(f"removed {plist_path}")
        else:
            print(f"no service installed at {plist_path}")
        _bootout_legacy_macos(force=migrate_legacy)
        return 0

    # Collision detection on install
    if not force and not dry_run:
        collision = _detect_existing_service(label, port)
        if collision:
            print(
                f"ERROR: fidelis-server already running from {collision['binary']} "
                f"(PID {collision['pid']}, {collision['found_by']})",
                file=sys.stderr,
            )
            print(
                f"  Existing service on port {collision['port']} (label: {collision['label']})",
                file=sys.stderr,
            )
            print(
                "  To upgrade this installation, run: fidelis init --force",
                file=sys.stderr,
            )
            if label == SERVICE_LABEL and port == PORT:
                print(
                    "  To run multiple fidelis instances, use: "
                    "fidelis init --label <custom-label> --port <port>",
                    file=sys.stderr,
                )
            return 1

    # Dry-run: print what would happen
    if dry_run:
        server_bin = _server_bin()
        log_path = Path.home() / ".fidelis" / "server.log"
        print("[DRY RUN] Would perform:")
        print(f"  - Label: {label}")
        print(f"  - Port: {port}")
        print(f"  - Binary: {server_bin}")
        print(f"  - Plist: {plist_path}")
        print(f"  - Log: {log_path}")
        return 0

    # Real install
    _bootout_legacy_macos(force=migrate_legacy)
    server_bin = _server_bin()
    log_path = Path.home() / ".fidelis" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare env vars
    base_env_vars = {
        "MEM0_TELEMETRY": "False",
        "ANONYMIZED_TELEMETRY": "False",
        "CHROMA_TELEMETRY_DISABLED": "True",
        "PYDANTIC_DISABLE_PLUGINS": "__all__",
        "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
    }

    # Merge with existing plist's env vars (preserve customizations)
    env_vars = _merge_plist_env_vars(plist_path, base_env_vars)

    # Preserve ThrottleInterval from an existing plist, if any
    throttle_interval = 15
    if plist_path.exists():
        try:
            import plistlib

            existing_data = plistlib.loads(plist_path.read_bytes())
            throttle_interval = existing_data.get("ThrottleInterval", 15)
        except Exception:
            pass

    # Format plist (preserve ThrottleInterval + EnvironmentVariables if they exist)
    plist = PLIST_TEMPLATE.format(
        label=label,
        server_bin=server_bin,
        working_dir=str(Path.home()),
        log_path=str(log_path),
        throttle_interval=throttle_interval,
        env_vars_xml=_render_env_vars_xml(env_vars),
    )

    # Backup existing plist before overwrite
    if plist_path.exists():
        backup = plist_path.with_suffix(f".plist.bak.{int(time.time())}")
        shutil.copy(plist_path, backup)
        print(f"backed up existing plist to {backup}")
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)

    plist_path.write_text(plist)
    print(f"wrote {plist_path}")
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"launchctl load failed: {result.stderr}", file=sys.stderr)
        return 1
    print(f"loaded service {label}")
    return 0


def _install_linux(
    uninstall: bool = False,
    force: bool = False,
    dry_run: bool = False,
    port: int | None = None,
    label: str | None = None,
    migrate_legacy: bool = False,
) -> int:
    if port is None:
        port = PORT
    if label is None:
        label = "fidelis-server"  # systemd uses .service suffix, not a separate label concept

    unit_path = Path.home() / ".config/systemd/user" / f"{label}.service"

    if uninstall:
        if unit_path.exists():
            subprocess.run(["systemctl", "--user", "stop", f"{label}.service"], check=False)
            subprocess.run(["systemctl", "--user", "disable", f"{label}.service"], check=False)
            unit_path.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            print(f"removed {unit_path}")
        else:
            print(f"no service installed at {unit_path}")
        return 0

    if dry_run:
        server_bin = _server_bin()
        log_path = Path.home() / ".fidelis" / "server.log"
        print("[DRY RUN] Would perform:")
        print(f"  - Service: {label}")
        print(f"  - Port: {port}")
        print(f"  - Binary: {server_bin}")
        print(f"  - Unit: {unit_path}")
        print(f"  - Log: {log_path}")
        return 0

    server_bin = _server_bin()
    log_path = Path.home() / ".fidelis" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.parent.mkdir(parents=True, exist_ok=True)

    unit = SYSTEMD_TEMPLATE.format(
        server_bin=server_bin,
        working_dir=str(Path.home()),
        log_path=str(log_path),
    )
    if unit_path.exists():
        backup = unit_path.with_suffix(f".service.bak.{int(time.time())}")
        shutil.copy(unit_path, backup)
        print(f"backed up existing unit to {backup}")

    unit_path.write_text(unit)
    print(f"wrote {unit_path}")
    for cmd in (
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", f"{label}.service"],
        ["systemctl", "--user", "start", f"{label}.service"],
    ):
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"systemctl command failed: {' '.join(cmd)}\n"
                f"  stdout: {result.stdout.strip()}\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  hint: ensure systemd --user is enabled (`loginctl enable-linger $USER`)\n"
                f"        or fall back to running `fidelis-server` manually under your process manager",
                file=sys.stderr,
            )
            return 1
    print(f"started {label}.service")
    return 0


def _install_fallback(uninstall: bool = False, **kwargs) -> int:
    """nohup-based fallback for unsupported platforms. No auto-start on reboot."""
    if uninstall:
        # Best-effort: kill any running fidelis-server
        subprocess.run(["pkill", "-f", "fidelis-server"], check=False)
        print("attempted to stop any running fidelis-server (no auto-start was configured)")
        return 0

    if kwargs.get("dry_run"):
        server_bin = _server_bin()
        log_path = Path.home() / ".fidelis" / "server.log"
        print("[DRY RUN] Would perform:")
        print(f"  - Start under nohup (no auto-restart): {server_bin}")
        print(f"  - Log: {log_path}")
        return 0

    server_bin = _server_bin()
    log_path = Path.home() / ".fidelis" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"WARNING: platform '{platform.system()}' has no auto-start support; "
        "starting under nohup. Will NOT survive reboot."
    )
    # Open + close in parent; pass fd duplicate to Popen so parent doesn't leak fd.
    log_fh = open(log_path, "ab")
    try:
        subprocess.Popen(
            [server_bin],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_fh.close()
    return 0


def _ollama_preflight() -> int:
    """Verify Ollama is reachable and the embed model is pulled.

    Returns 0 on success, non-zero with a one-line user-facing error otherwise.
    Failing here saves the user from a launchd service that boots, crashes on
    first /store, and keeps restart-looping silently. The fix is always one
    shell command; we name it instead of leaving them to debug a daemon.
    """
    ollama_url = os.environ.get("COGITO_OLLAMA_URL", "http://localhost:11434")
    embed_model = os.environ.get("COGITO_EMBED_MODEL", "nomic-embed-text")
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=3) as resp:
            if resp.status != 200:
                print(
                    f"ERROR: Ollama at {ollama_url} returned HTTP {resp.status}. "
                    f"Start it with `ollama serve &` and retry.",
                    file=sys.stderr,
                )
                return 2
            payload = resp.read()
    except (urllib.error.URLError, OSError) as e:
        print(
            f"ERROR: Ollama not reachable at {ollama_url} ({e}). "
            f"Install with `brew install ollama` (macOS) or see https://ollama.com, "
            f"then `ollama serve &` and retry.",
            file=sys.stderr,
        )
        return 2

    # Confirm the embed model is pulled. Cheap to check, painful to debug otherwise.
    try:
        import json as _json

        models = _json.loads(payload).get("models", [])
        names = {m.get("name", "").split(":")[0] for m in models}
        if embed_model.split(":")[0] not in names:
            print(
                f"ERROR: Ollama is up but the embed model '{embed_model}' is not pulled. "
                f"Run `ollama pull {embed_model}` (~280 MB, one-time) and retry.",
                file=sys.stderr,
            )
            return 2
    except (ValueError, KeyError) as e:  # noqa: silent — best-effort tag parsing
        print(f"warning: could not parse Ollama /api/tags response ({e}); proceeding", file=sys.stderr)
    return 0


def cmd_init(args) -> int:
    """Install + start fidelis-server as a system service.

    --uninstall: stop service + remove the unit/plist.
    --force: override collision detection and proceed with install.
    --dry-run: print what would change without modifying anything.
    --port: override the default port (for multiple instances).
    --label: override the default label (for multiple instances, macOS only).
    --migrate: remove legacy launchd labels (ai.hermeslabs.cogito-server, etc).
    """
    system = platform.system()

    if args.uninstall:
        if system == "Darwin":
            return _install_macos(uninstall=True, migrate_legacy=args.migrate)
        elif system == "Linux":
            return _install_linux(uninstall=True)
        else:
            return _install_fallback(uninstall=True)

    # Preflight: refuse to install a service that we know will crash at first
    # write because Ollama isn't running or the embed model isn't pulled.
    rc = _ollama_preflight()
    if rc != 0:
        return rc

    port = getattr(args, "port", None) or PORT
    label = getattr(args, "label", None) or SERVICE_LABEL
    force = getattr(args, "force", False)
    dry_run = getattr(args, "dry_run", False)
    migrate = getattr(args, "migrate", False)

    print(f"installing fidelis-server as a {system} service...")
    if system == "Darwin":
        rc = _install_macos(
            force=force,
            dry_run=dry_run,
            port=port,
            label=label,
            migrate_legacy=migrate,
        )
    elif system == "Linux":
        rc = _install_linux(
            force=force,
            dry_run=dry_run,
            port=port,
            label=label,
            migrate_legacy=migrate,
        )
    else:
        rc = _install_fallback(dry_run=dry_run, port=port, label=label)

    if rc != 0:
        return rc

    if dry_run:
        print("[DRY RUN] No changes were made.")
        return 0

    print("waiting for service to come up...")
    check_port = port if port else PORT
    if _health_check(timeout_s=10.0, port=check_port):
        print(f"✓ fidelis-server is up at http://127.0.0.1:{check_port}")
        print(f"  log: {Path.home() / '.fidelis' / 'server.log'}")
        print()
        print("next steps:")
        print("  fidelis health                  # confirm")
        print("  fidelis watch ~/notes           # auto-ingest a directory")
        print("  fidelis mcp install             # wire up Claude Code")
        return 0
    else:
        print("✗ service installed but /health did not respond within 10s")
        print(f"  check log: {Path.home() / '.fidelis' / 'server.log'}")
        return 2
