"""Run the copied Hermes Gate runner under an interpreter that satisfies its floor.

``hermes-gate init`` writes a profile whose ``diff-check`` step is
``python3 .hermes/hermes_gate_runner.py`` and a runner that refuses anything
older than Python 3.11 -- the floor of the hermes-gate CLI, not of this
package, which supports 3.10. On a host whose ``python3`` is 3.10 that step
exited before checking a single file.

The runner is copied byte-for-byte and hash-verified by ``hermes-gate
doctor``, so its floor cannot be lowered here, and the profile cannot name the
CLI's interpreter because that path is host-specific. What every host that
can run the gate does have is the ``hermes-gate`` launcher on PATH, and its
shebang is the 3.11+ interpreter the CLI runs under. This shim uses the
current interpreter when it is new enough and borrows the launcher's
otherwise; when neither is possible it says so instead of failing inside the
runner's version guard.
"""

from __future__ import annotations

import os
import shutil
import sys

RUNNER_FLOOR = (3, 11)
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hermes_gate_runner.py")


def launcher_interpreter(launcher: str) -> list[str] | None:
    """The interpreter argv a console-script launcher's shebang names, or None."""
    try:
        with open(launcher, "rb") as handle:
            first = handle.readline()
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    return first[2:].decode("utf-8", "replace").split() or None


def main(argv: list[str]) -> int:
    if sys.version_info >= RUNNER_FLOOR:
        prefix = [sys.executable]
    else:
        launcher = shutil.which("hermes-gate")
        prefix = launcher_interpreter(launcher) if launcher else None
        if not prefix:
            major, minor = sys.version_info[:2]
            sys.stderr.write(
                f"hermes-gate diff-check: python3 is {major}.{minor} but the runner needs "
                f"{RUNNER_FLOOR[0]}.{RUNNER_FLOOR[1]}+, and no hermes-gate launcher is on PATH "
                "to borrow an interpreter from; run the gate with Python 3.11+\n"
            )
            return 1
    os.execv(prefix[0], [*prefix, RUNNER, *argv])
    return 1  # unreachable: execv replaces the process


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
