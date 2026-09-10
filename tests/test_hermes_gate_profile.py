"""Regression tests for the repository gate profile under ``.hermes/``.

``hermes-gate fast`` spawns each step's argv verbatim, so ``python3`` is
whatever PATH resolves. The copied runner refuses Python older than 3.11 (the
CLI's floor), while this package supports 3.10, so the ``diff-check`` step used
to exit inside that guard on a 3.10 host before checking a single file. The
step now goes through ``.hermes/run_gate_runner.py``. These tests run it the
way the gate does -- as a subprocess, in a throwaway Git repository, with
``python3`` and the ``hermes-gate`` launcher on a PATH the test controls.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE_PROFILE = REPO / ".hermes" / "gate.toml"
RUNNER_FLOOR = (3, 11)


def _diff_check_argv() -> list[str]:
    """The ``diff-check`` fast step exactly as ``hermes-gate`` reads it."""
    text = GATE_PROFILE.read_text(encoding="utf-8")
    try:
        import tomllib
    except ImportError:  # Python 3.10: read the one array the test needs
        tomllib = None
    if tomllib is not None:
        steps = [s for s in tomllib.loads(text)["fast"] if s["name"] == "diff-check"]
        assert len(steps) == 1
        return list(steps[0]["argv"])
    match = re.search(r'name = "diff-check"\nargv = (\[.*?\])\n', text)
    assert match, "diff-check step not found in gate.toml"
    return ast.literal_eval(match.group(1))


def _version_of(interpreter: str) -> tuple[int, int] | None:
    """The interpreter's (major, minor), or None when it cannot even start up
    with what the runner imports (a broken site-packages, say)."""
    probe = subprocess.run(
        [interpreter, "-c", "import subprocess, sys; print(*sys.version_info[:2])"],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0 or probe.stderr.strip():
        return None
    major, minor = probe.stdout.split()
    return int(major), int(minor)


def _usable(interpreter: str, *, below_floor: bool) -> bool:
    version = _version_of(interpreter)
    return version is not None and (version < RUNNER_FLOOR) == below_floor


def _old_python() -> str | None:
    """An interpreter below the runner's floor, or None when the host has none."""
    if sys.version_info < RUNNER_FLOOR:
        return sys.executable
    for name in ("python3.10",):
        found = shutil.which(name)
        if found and _usable(found, below_floor=True):
            return found
    uv = shutil.which("uv")
    if uv:
        probe = subprocess.run(
            [uv, "python", "find", "3.10"], capture_output=True, text=True, check=False,
            cwd=Path(os.devnull).parent,
        )
        found = probe.stdout.strip()
        if probe.returncode == 0 and found and _usable(found, below_floor=True):
            return found
    return None


def _new_python() -> str | None:
    """An interpreter at or above the runner's floor, or None."""
    if sys.version_info >= RUNNER_FLOOR:
        return sys.executable
    for name in ("python3.14", "python3.13", "python3.12", "python3.11"):
        found = shutil.which(name)
        if found and _usable(found, below_floor=False):
            return found
    return None


@pytest.fixture
def gate_repo(tmp_path):
    """A throwaway Git repository carrying this repository's ``.hermes/`` profile."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    shutil.copytree(REPO / ".hermes", root / ".hermes")
    (root / "dirty.txt").write_text("trailing space \n")
    (root / "clean.txt").write_text("no trailing space\n")
    return root


def _run_step(root: Path, *, python3: str, launcher_python: str | None, files: list[str]):
    """Spawn the diff-check step as ``hermes-gate fast`` does, with PATH pinned."""
    bin_dir = root.parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python3").unlink(missing_ok=True)
    (bin_dir / "python3").symlink_to(python3)
    launcher = bin_dir / "hermes-gate"
    launcher.unlink(missing_ok=True)
    if launcher_python is not None:
        launcher.write_text(f"#!{launcher_python}\nraise SystemExit('launcher stub')\n")
        launcher.chmod(0o755)
    # Only the pinned bin dir and Git's own directory: the real ``hermes-gate``
    # launcher of the host must not leak into a case that says there is none.
    git_dir = os.path.dirname(shutil.which("git") or "/usr/bin/git")
    path = os.pathsep.join([str(bin_dir), git_dir, "/usr/bin", "/bin"])
    assert shutil.which("hermes-gate", path=path) == (str(launcher) if launcher_python else None)
    env = dict(os.environ, PATH=path)
    argv = [part for step in _diff_check_argv() for part in (files if step == "{files}" else [step])]
    return subprocess.run(argv, cwd=root, env=env, capture_output=True, text=True, check=False)


def test_runner_alone_refuses_an_old_python(gate_repo):
    """The failure the shim exists for: the copied runner cannot run on 3.10."""
    old = _old_python()
    if old is None:
        pytest.skip("no Python below 3.11 on this host")
    result = subprocess.run(
        [old, ".hermes/hermes_gate_runner.py", "diff-check", "dirty.txt"],
        cwd=gate_repo, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1
    assert "requires Python 3.11" in result.stderr
    assert "trailing whitespace" not in result.stdout  # it never looked


@pytest.mark.skipif(sys.version_info < RUNNER_FLOOR, reason="needs Python 3.11+ as python3")
def test_diff_check_step_runs_under_a_current_python3(gate_repo):
    dirty = _run_step(gate_repo, python3=sys.executable, launcher_python=None, files=["dirty.txt"])
    assert dirty.returncode != 0, dirty
    assert "trailing whitespace" in dirty.stdout

    clean = _run_step(gate_repo, python3=sys.executable, launcher_python=None, files=["clean.txt"])
    assert clean.returncode == 0, clean


def test_diff_check_step_borrows_the_launcher_interpreter_on_an_old_python3(gate_repo):
    old, new = _old_python(), _new_python()
    if old is None or new is None:
        pytest.skip("needs one Python below 3.11 and one at or above it")

    dirty = _run_step(gate_repo, python3=old, launcher_python=new, files=["dirty.txt"])
    assert "requires Python 3.11" not in dirty.stderr, dirty
    assert dirty.returncode != 0, dirty
    assert "trailing whitespace" in dirty.stdout

    clean = _run_step(gate_repo, python3=old, launcher_python=new, files=["clean.txt"])
    assert clean.returncode == 0, clean


def test_diff_check_step_names_the_gap_when_no_launcher_can_help(gate_repo):
    old = _old_python()
    if old is None:
        pytest.skip("no Python below 3.11 on this host")
    result = _run_step(gate_repo, python3=old, launcher_python=None, files=["dirty.txt"])
    assert result.returncode == 1
    assert "needs 3.11+" in result.stderr
    assert "no hermes-gate launcher" in result.stderr
    assert "Traceback" not in result.stderr
