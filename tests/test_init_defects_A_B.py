"""Regression tests for DEFECT A and DEFECT B in fidelis init.

DEFECT A (P0 - GLOBAL SINGLETON COLLISION):
    Module-level constants PORT=19420 and SERVICE_LABEL="ai.hermeslabs.fidelis-server".
    fidelis init unconditionally writes ~/Library/LaunchAgents/<label>.plist and
    force-restarts that launchd service with NO check for an already-running instance.
    This destroys concurrent installations and orphans memories.

DEFECT B (SILENT CONFIG DEGRADATION):
    The PLIST_TEMPLATE in init_cmd.py does NOT contain PYDANTIC_DISABLE_PLUGINS or
    ThrottleInterval, but live production plists DO. A re-init silently downgrades
    working configs even when it 'succeeds'.

Tests use monkeypatch for subprocess/launchctl and tmp_path for HOME isolation.
No actual destructive operations are executed.
"""

from __future__ import annotations

import plistlib
import subprocess
from unittest.mock import Mock


# Import the functions we're testing
from fidelis.init_cmd import (
    PLIST_TEMPLATE,
    SERVICE_LABEL,
)


class TestDefectA_CollisionDetection:
    """Tests for DEFECT A: Global singleton collision (P0).

    Assertions:
    1. fidelis init should detect when a service with the same label is already loaded
    2. fidelis init should detect when a service is listening on the same port
    3. fidelis init should refuse to proceed by default, with a clear error message
    4. fidelis init should NOT unload/reload a running service without explicit --force
    5. The production service MUST remain untouched after a collision attempt
    """

    def test_collision_detection_rejects_already_loaded_service(
        self, tmp_path, monkeypatch
    ):
        """DEFECT A: fidelis init should REFUSE when a service with this label
        is already loaded (even if from a different binary).

        Setup: A launchd service labeled ai.hermeslabs.fidelis-server is running
               from /some/other/venv/bin/fidelis-server (older or different version).

        Expected behavior: fidelis init detects the label collision and prints
                          a clear error message naming the conflict, then REFUSES
                          to proceed (returns 1, touches nothing).

        This test FAILS against current main because main has zero collision detection.
        """
        # Setup: fake HOME pointing to tmp_path
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        # Setup: fake LaunchAgents directory
        launch_agents = fake_home / "Library" / "LaunchAgents"
        launch_agents.mkdir(parents=True)

        # Setup: pre-existing plist from a different install (collision scenario)
        collision_plist_path = launch_agents / f"{SERVICE_LABEL}.plist"
        old_plist_data = PLIST_TEMPLATE.format(
            label=SERVICE_LABEL,
            server_bin="/other/venv/bin/fidelis-server",
            working_dir=str(fake_home),
            log_path=str(fake_home / ".fidelis" / "server.log"),
            throttle_interval=15,
            env_vars_xml='        <key>MEM0_TELEMETRY</key>\n        <string>False</string>',
        )
        collision_plist_path.write_text(old_plist_data)

        # Mock: subprocess.run for launchctl list (service is loaded with PID 12345)
        def mock_subprocess_run(cmd, *args, **kwargs):
            result = Mock()
            result.returncode = 0  # Service exists
            result.stdout = "12345"
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

        # Mock: _server_bin() returns current binary
        monkeypatch.setattr(
            "fidelis.init_cmd._server_bin",
            lambda: "/current/venv/bin/fidelis-server",
        )

        # Mock: _health_check to avoid waiting
        monkeypatch.setattr("fidelis.init_cmd._health_check", lambda timeout_s=10.0, port=None: False)

        # Import fresh to get our mocked functions
        from fidelis import init_cmd

        # ACTION: try to install
        # ASSERTION: Should return non-zero (refuse to proceed)
        result = init_cmd._install_macos()

        # EXPECTED: detects collision and returns 1 (REFUSES)
        # This test FAILS on current main — init_cmd._install_macos has no
        # collision detection and proceeds to overwrite.
        # After our fix, this should return 1 and not touch the plist.
        assert result != 0, (
            "fidelis init should refuse when a service is already loaded with that label"
        )

        # ASSERTION: plist should NOT be modified
        assert collision_plist_path.read_text() == old_plist_data, (
            "plist should not be overwritten when collision is detected"
        )

    def test_collision_detection_rejects_port_in_use(self, tmp_path, monkeypatch):
        """DEFECT A: fidelis init should REFUSE when the port is already in use
        (even if the service label is different).

        Setup: Some service is listening on 127.0.0.1:19420 (collision by port).

        Expected behavior: fidelis init detects the port collision and refuses.
        """
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        launch_agents = fake_home / "Library" / "LaunchAgents"
        launch_agents.mkdir(parents=True)

        # Mock: subprocess.run — launchctl list returns 1, lsof finds the port
        def mock_subprocess_run(cmd, *args, **kwargs):
            result = Mock()
            if "lsof" in cmd:
                result.returncode = 0
                result.stdout = "COMMAND   PID  USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME\nOTHER  12345  user    4u  IPv4 0xabcdef       0t0  tcp 127.0.0.1:19420 (LISTEN)"
            else:
                result.returncode = 1  # Label not loaded
                result.stdout = ""
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
        monkeypatch.setattr(
            "fidelis.init_cmd._server_bin",
            lambda: "/current/venv/bin/fidelis-server",
        )
        monkeypatch.setattr("fidelis.init_cmd._health_check", lambda timeout_s=10.0, port=None: False)

        from fidelis import init_cmd

        result = init_cmd._install_macos()

        # EXPECTED: Should refuse because port is in use
        assert result != 0, (
            "fidelis init should refuse when the port is already in use"
        )

    def test_collision_detection_parses_real_launchctl_dict_output(
        self, tmp_path, monkeypatch
    ):
        """ADVERSARIAL — reproduces the EXACT failure mode from
        FLAGS.md 2026-09-15-011: `launchctl list <label>` on a real macOS box
        does NOT print the tabular "PID STATUS LABEL" format assumed by the
        two tests above (which use a fabricated stdout of "12345"). For a
        single label, real launchd prints a property-list dict, e.g.:

            {
                "PID" = 74977;
                "Label" = "ai.hermeslabs.fidelis-server";
                "LastExitStatus" = 0;
                ...
            };

        A parser doing int(result.stdout.split('\\n')[0].split()[0]) chokes on
        the literal "{" token and raises ValueError — which, before this fix,
        made _detect_existing_service() silently return None for the label
        check even though the label WAS loaded, relying entirely on the
        independent port-listener check to catch the collision. This test
        uses the exact real launchctl output captured from this machine's
        live production fidelis-server (147,996-memory instance, PID 74977)
        to prove refusal-not-overwrite against the real shape, and that the
        PID is correctly extracted for the user-facing message.
        """
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        launch_agents = fake_home / "Library" / "LaunchAgents"
        launch_agents.mkdir(parents=True)

        collision_plist_path = launch_agents / f"{SERVICE_LABEL}.plist"
        old_plist_data = PLIST_TEMPLATE.format(
            label=SERVICE_LABEL,
            server_bin="/Users/rbr_lpci/hermes-venv/bin/fidelis-server",
            working_dir=str(fake_home),
            log_path=str(fake_home / ".fidelis" / "server.log"),
            throttle_interval=15,
            env_vars_xml='        <key>MEM0_TELEMETRY</key>\n        <string>False</string>',
        )
        collision_plist_path.write_text(old_plist_data)

        real_launchctl_list_output = (
            '{\n'
            '\t"StandardOutPath" = "/Users/rbr_lpci/.fidelis/server.log";\n'
            '\t"LimitLoadToSessionType" = "Aqua";\n'
            '\t"StandardErrorPath" = "/Users/rbr_lpci/.fidelis/server.log";\n'
            '\t"Label" = "ai.hermeslabs.fidelis-server";\n'
            '\t"OnDemand" = true;\n'
            '\t"LastExitStatus" = 0;\n'
            '\t"PID" = 74977;\n'
            '\t"Program" = "/Users/rbr_lpci/hermes-venv/bin/fidelis-server";\n'
            '\t"ProgramArguments" = (\n'
            '\t\t"/Users/rbr_lpci/hermes-venv/bin/fidelis-server";\n'
            '\t);\n'
            '};'
        )

        launchctl_calls = []

        def mock_subprocess_run(cmd, *args, **kwargs):
            result = Mock()
            if cmd[:2] == ["launchctl", "list"]:
                launchctl_calls.append(cmd)
                result.returncode = 0
                result.stdout = real_launchctl_list_output
            else:
                result.returncode = 1
                result.stdout = ""
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
        monkeypatch.setattr(
            "fidelis.init_cmd._server_bin",
            lambda: "/current/venv/bin/fidelis-server",
        )
        monkeypatch.setattr("fidelis.init_cmd._health_check", lambda timeout_s=10.0, port=None: False)

        from fidelis import init_cmd

        collision = init_cmd._detect_existing_service(SERVICE_LABEL, 19420)
        assert collision is not None, (
            "collision detection must recognize the label is loaded from real "
            "launchctl dict-shaped output, not just a fabricated tabular one"
        )
        assert collision["pid"] == 74977, (
            "PID must be correctly extracted from the real property-list "
            "output so the refusal message can name the live process"
        )

        # ACTION: try to install — must refuse, must NOT touch the plist or
        # call launchctl unload/load on it.
        result = init_cmd._install_macos()

        assert result != 0, (
            "fidelis init must refuse when launchctl reports the label loaded "
            "via real dict-shaped output — this is the exact P0 regression"
        )
        assert collision_plist_path.read_text() == old_plist_data, (
            "plist must be byte-identical after a refused install — no "
            "overwrite, no unload/reload of a possibly-live service"
        )
        for call_args in launchctl_calls:
            assert call_args[:2] != ["launchctl", "unload"], (
                "must never unload the colliding service without --force"
            )


class TestDefectB_ConfigDegradation:
    """Tests for DEFECT B: Silent config degradation.

    Assertions:
    1. When re-running fidelis init, pre-existing EnvironmentVariables should be preserved
    2. Specifically, PYDANTIC_DISABLE_PLUGINS and ThrottleInterval should survive a re-init
    3. The new plist should be a merge (old + new), not a replacement
    """

    def test_plist_template_lacks_required_env_vars(self):
        """DEFECT B documentation: Confirm that the template NOW CONTAINS
        PYDANTIC_DISABLE_PLUGINS and ThrottleInterval.

        This test documents that the defect is FIXED. After our fix, this
        test should PASS.
        """
        # Check PLIST_TEMPLATE for required keys
        template_has_pydantic = "PYDANTIC_DISABLE_PLUGINS" in PLIST_TEMPLATE
        template_has_throttle = "ThrottleInterval" in PLIST_TEMPLATE

        # FIXED: template now contains these
        assert template_has_pydantic, (
            "PLIST_TEMPLATE should now contain PYDANTIC_DISABLE_PLUGINS; "
            "the defect is fixed."
        )
        assert template_has_throttle, (
            "PLIST_TEMPLATE should now contain ThrottleInterval; "
            "the defect is fixed."
        )

    def test_reinit_preserves_existing_environment_variables(self, tmp_path, monkeypatch):
        """DEFECT B: When re-running fidelis init on an existing install,
        pre-existing EnvironmentVariables should be preserved.

        Setup: An existing plist at ~/Library/LaunchAgents/ai.hermeslabs.fidelis-server.plist
               with custom env vars: PYDANTIC_DISABLE_PLUGINS=__all__ and ThrottleInterval=15.

        Expected behavior: After re-init, those vars are still present in the new plist.

        Current behavior (DEFECT B): The new plist lacks those vars, silently degrading
                                     the config.
        """
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        launch_agents = fake_home / "Library" / "LaunchAgents"
        launch_agents.mkdir(parents=True)

        plist_path = launch_agents / f"{SERVICE_LABEL}.plist"

        # Create a pre-existing plist with custom env vars
        # (simulating a production install that was customized)
        pre_existing_plist = PLIST_TEMPLATE.format(
            label=SERVICE_LABEL,
            server_bin="/usr/local/bin/fidelis-server",
            working_dir=str(fake_home),
            log_path=str(fake_home / ".fidelis" / "server.log"),
            throttle_interval=15,
            env_vars_xml='        <key>MEM0_TELEMETRY</key>\n        <string>False</string>',
        )

        # Parse and inject custom env vars
        plist_dict = plistlib.loads(pre_existing_plist.encode())
        plist_dict["EnvironmentVariables"]["PYDANTIC_DISABLE_PLUGINS"] = "__all__"
        plist_dict["ThrottleInterval"] = 15

        # Write it back
        plist_path.write_bytes(plistlib.dumps(plist_dict))

        # Verify pre-existing state
        pre_existing_data = plistlib.loads(plist_path.read_bytes())
        assert pre_existing_data["EnvironmentVariables"]["PYDANTIC_DISABLE_PLUGINS"] == "__all__"
        assert pre_existing_data.get("ThrottleInterval") == 15

        # Mock: subprocess for launchctl
        def mock_subprocess_run(cmd, *args, **kwargs):
            result = Mock()
            result.returncode = 1  # Service not loaded
            result.stdout = ""
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
        monkeypatch.setattr(
            "fidelis.init_cmd._server_bin",
            lambda: "/usr/local/bin/fidelis-server",
        )
        monkeypatch.setattr("fidelis.init_cmd._health_check", lambda timeout_s=10.0, port=None: True)

        from fidelis import init_cmd

        # ACTION: re-run fidelis init
        init_cmd._install_macos()

        # Read the new plist that was written
        new_plist_data = plistlib.loads(plist_path.read_bytes())

        # ASSERTION: Custom env vars should be preserved
        assert (
            new_plist_data["EnvironmentVariables"].get("PYDANTIC_DISABLE_PLUGINS") == "__all__"
        ), (
            "Custom EnvironmentVariable PYDANTIC_DISABLE_PLUGINS was lost "
            "during re-init. Pre-existing config should be merged, not replaced."
        )

        assert new_plist_data.get("ThrottleInterval") == 15, (
            "Custom plist key ThrottleInterval was lost during re-init. "
            "Pre-existing config should be merged, not replaced."
        )

    def test_reinit_idempotent_on_working_install(self, tmp_path, monkeypatch):
        """Idempotency test: running fidelis init twice should produce
        the same plist both times (no silent degradation).
        """
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        launch_agents = fake_home / "Library" / "LaunchAgents"
        launch_agents.mkdir(parents=True)

        plist_path = launch_agents / f"{SERVICE_LABEL}.plist"

        def mock_subprocess_run(cmd, *args, **kwargs):
            result = Mock()
            result.returncode = 1  # Service not loaded
            result.stdout = ""
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
        monkeypatch.setattr(
            "fidelis.init_cmd._server_bin",
            lambda: "/usr/local/bin/fidelis-server",
        )
        monkeypatch.setattr("fidelis.init_cmd._health_check", lambda timeout_s=10.0, port=None: True)

        from fidelis import init_cmd

        # First init
        init_cmd._install_macos()
        first_plist = plist_path.read_bytes()

        # Second init
        init_cmd._install_macos()
        second_plist = plist_path.read_bytes()

        # ASSERTION: plists should be identical (idempotent)
        assert first_plist == second_plist, (
            "DEFECT B: Running fidelis init twice produced different plists. "
            "The operation should be idempotent."
        )


class TestLegacyLabelGating:
    """Tests for legacy label unlink gating.

    The code calls _bootout_legacy_macos() unconditionally, which unlinks
    plists for ai.hermeslabs.cogito-server and ai.cogito.server.
    This should be gated behind an explicit flag (e.g., --migrate or only on first install).
    """

    def test_legacy_bootout_should_not_run_on_reinit(self, tmp_path, monkeypatch):
        """Legacy label cleanup should only run on explicit opt-in,
        not silently on every reinit.
        """
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        legacy_agents = fake_home / "Library" / "LaunchAgents"
        legacy_agents.mkdir(parents=True)

        legacy_plist = legacy_agents / "ai.hermeslabs.cogito-server.plist"
        legacy_plist.write_text("<xml>legacy</xml>")

        def mock_subprocess_run(cmd, *args, **kwargs):
            result = Mock()
            result.returncode = 1  # Service not loaded
            result.stdout = ""
            result.stderr = ""
            return result

        monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
        monkeypatch.setattr(
            "fidelis.init_cmd._server_bin",
            lambda: "/usr/local/bin/fidelis-server",
        )
        monkeypatch.setattr("fidelis.init_cmd._health_check", lambda timeout_s=10.0, port=None: True)

        from fidelis import init_cmd

        # Re-init should NOT delete legacy plists without explicit opt-in
        init_cmd._install_macos()

        # ASSERTION: legacy plist should still exist (not deleted)
        # CURRENT MAIN: _bootout_legacy_macos runs unconditionally, deleting it
        assert legacy_plist.exists(), (
            "Legacy plist should not be unlinked on routine reinit. "
            "It should only be removed on explicit --migrate or first install."
        )
