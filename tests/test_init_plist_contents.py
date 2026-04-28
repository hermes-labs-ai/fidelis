"""Pin the launchd plist + systemd unit content.

These tests prevent silent regressions of two specific production-tested
requirements:

1. **Telemetry kill** — chromadb pulls posthog at import. In a launchd /
   systemd restart loop, posthog leaks file descriptors until the service
   trips EMFILE and dies silently. The kill flags must be present.
2. **Server binary path** — the unit must point to the `fidelis-server`
   console script that pip installs, not a hardcoded source path.

If either of these regresses, friends running `fidelis init` get a
service that either silently crashes or never starts. The tests are
text-level — no real launchctl / systemctl invocations.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def fake_home(monkeypatch):
    """Isolate Path.home() to a tmpdir so install writes don't touch the user's
    real ~/Library/LaunchAgents or ~/.config/systemd/user."""
    tmp = Path(tempfile.mkdtemp(prefix="fidelis-plist-test-"))
    monkeypatch.setenv("HOME", str(tmp))
    return tmp


def test_macos_plist_includes_telemetry_kill(fake_home):
    """ANONYMIZED_TELEMETRY=False, CHROMA_TELEMETRY_DISABLED=True, POSTHOG_DISABLED=1
    must all be in the launchd plist EnvironmentVariables block. Otherwise
    chromadb's posthog import will leak fds across launchd's restart loop and
    eventually EMFILE-crash the user's service silently."""
    from fidelis.init_cmd import _install_macos

    with patch("subprocess.run") as fake_run:
        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""
        fake_run.return_value = _Result()
        rc = _install_macos(uninstall=False)
        assert rc == 0

    plist_path = fake_home / "Library/LaunchAgents/ai.hermeslabs.fidelis-server.plist"
    assert plist_path.exists(), f"plist not written at {plist_path}"
    content = plist_path.read_text()

    # Each of the three flags must be present. If any drops out of the
    # template, chromadb/posthog will leak fds and crash on restart.
    for required in (
        "<key>ANONYMIZED_TELEMETRY</key>",
        "<string>False</string>",
        "<key>CHROMA_TELEMETRY_DISABLED</key>",
        "<string>True</string>",
        "<key>POSTHOG_DISABLED</key>",
        "<string>1</string>",
    ):
        assert required in content, (
            f"plist missing telemetry-kill marker {required!r}; "
            f"chromadb posthog import will EMFILE-crash launchd restarts. "
            f"See feedback_disable_chromadb_posthog_telemetry.md."
        )


def test_macos_plist_uses_console_script(fake_home):
    """ProgramArguments must reference the `fidelis-server` entry point pip
    installs, not a hardcoded source path. Otherwise users on different
    machines/venvs hit a path that doesn't exist."""
    from fidelis.init_cmd import _install_macos

    with patch("subprocess.run") as fake_run:
        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""
        fake_run.return_value = _Result()
        _install_macos(uninstall=False)

    plist_path = fake_home / "Library/LaunchAgents/ai.hermeslabs.fidelis-server.plist"
    content = plist_path.read_text()

    # Path should end in the console-script name. Don't assert an absolute
    # path — that varies per env. Do assert it's not the source dir.
    assert "fidelis-server" in content
    assert "src/fidelis/server.py" not in content, (
        "plist references the source file directly; should be the pip-installed "
        "console script `fidelis-server` so it works from any venv."
    )


def test_systemd_unit_includes_telemetry_kill(fake_home):
    """Same EMFILE concern, Linux flavor — Environment= lines in the unit."""
    # Bypass the systemctl + daemon-reload subprocess calls; we only care
    # about what's written to disk.
    from fidelis.init_cmd import SYSTEMD_TEMPLATE

    rendered = SYSTEMD_TEMPLATE.format(
        server_bin="/fake/venv/bin/fidelis-server",
        log_path="/tmp/fidelis-test.log",
        working_dir="/tmp",
    )

    for required in (
        "Environment=ANONYMIZED_TELEMETRY=False",
        "Environment=CHROMA_TELEMETRY_DISABLED=True",
        "Environment=POSTHOG_DISABLED=1",
    ):
        assert required in rendered, (
            f"systemd unit missing telemetry-kill {required!r}; "
            f"chromadb posthog import will EMFILE-crash on restarts."
        )


def test_legacy_label_bootout_is_idempotent(fake_home):
    """`_bootout_legacy_macos` must work even when no legacy plist exists.
    Idempotency matters — the function runs on every `fidelis init`."""
    from fidelis.init_cmd import _bootout_legacy_macos

    # No legacy plist on disk; must not raise.
    _bootout_legacy_macos()

    # Plant a fake legacy plist; verify it gets removed.
    legacy_dir = fake_home / "Library/LaunchAgents"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_plist = legacy_dir / "ai.hermeslabs.cogito-server.plist"
    legacy_plist.write_text("<?xml version='1.0'?><plist></plist>")

    with patch("subprocess.run") as fake_run:
        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""
        fake_run.return_value = _Result()
        _bootout_legacy_macos()

    assert not legacy_plist.exists(), "legacy plist not removed by migration"
