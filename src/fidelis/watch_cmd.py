"""fidelis watch — auto-ingest new markdown/text files from a directory.

Behavior:
- Initial scan: ingests all matching files (capped by --max-files)
- Continuous: polls for new/modified files every --interval seconds
- Idempotent: tracks ingested file hashes in ~/.fidelis/watched.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_GLOB_PATTERNS = ("*.md", "*.txt")
DEFAULT_MAX_FILES = 500
DEFAULT_INTERVAL_S = 5.0
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB — skip files larger than this
LEDGER_PATH = Path.home() / ".fidelis" / "watched.json"

# Backpressure constants. Tuned so that dropping 10,000 small markdown files
# into a watched directory does not OOM the user's machine or hammer the
# server's embedding queue:
#   - INGEST_BATCH_SIZE: process in chunks of this many files between health
#     checks. After each batch we check /health; if the server reports
#     degraded or queued depth above the threshold, we pause the watcher.
#   - HEALTH_QUEUE_PAUSE: if /health reports more than this many items in
#     the graceful-degrade queue, the server is already behind — stop
#     adding more until it catches up.
#   - SESSION_BYTE_CAP: refuse to push more than this many total bytes in a
#     single watch session. Default 1 GB. Catches the runaway "I pointed
#     watch at /Users" pattern.
#   - PAUSE_S: how long to wait between batches when paused.
INGEST_BATCH_SIZE = 50
HEALTH_QUEUE_PAUSE = 200
SESSION_BYTE_CAP = 1 * 1024 * 1024 * 1024
PAUSE_S = 5.0


def _server_url() -> str:
    port = os.environ.get("FIDELIS_PORT", os.environ.get("COGITO_PORT", "19420"))
    return f"http://127.0.0.1:{port}"


def _post(path: str, payload: dict, timeout: float = 30.0) -> dict | None:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{_server_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError) as e:
        print(f"  [warn] {path} request failed: {e}", file=sys.stderr)
        return None


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ledger() -> dict:
    if LEDGER_PATH.exists():
        try:
            return json.loads(LEDGER_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_ledger(ledger: dict) -> None:
    """Atomic write: tempfile + os.replace prevents corruption if killed mid-write.

    Uses parent/(name+".tmp") instead of with_suffix to be safe on Python 3.10/3.11
    where with_suffix raised ValueError on multi-dot suffixes."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.parent / (LEDGER_PATH.name + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2))
    os.replace(tmp, LEDGER_PATH)


def _scan_files(root: Path, patterns: tuple, max_files: int) -> list[Path]:
    found: list[Path] = []
    for pat in patterns:
        for f in root.rglob(pat):
            if f.is_file():
                found.append(f)
                if len(found) >= max_files:
                    return found
    return found


def _ingest_file(path: Path, verbose: bool, max_bytes: int = DEFAULT_MAX_FILE_BYTES) -> tuple[bool, int]:
    """Returns (success, bytes_pushed). bytes_pushed is 0 if skipped.

    Caller uses bytes_pushed to enforce session-level byte caps.
    """
    try:
        size = path.stat().st_size
    except OSError as e:
        print(f"  [skip] {path}: {e}", file=sys.stderr)
        return False, 0
    if size > max_bytes:
        if verbose:
            print(f"  [skip-toobig] {path} ({size:,} bytes > {max_bytes:,} cap)", file=sys.stderr)
        return False, 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"  [skip] {path}: {e}", file=sys.stderr)
        return False, 0
    if not text.strip():
        return False, 0
    res = _post("/store", {"text": text, "metadata": {"source_path": str(path)}})
    if res and verbose:
        print(f"  [ingest] {path}")
    return res is not None, len(text.encode("utf-8")) if res else 0


def _server_pressure() -> tuple[bool, int]:
    """Check /health. Returns (under_pressure, queued_depth).

    under_pressure=True means the server is degraded or its graceful-degrade
    queue is deeper than HEALTH_QUEUE_PAUSE. Caller should pause.
    """
    try:
        with urllib.request.urlopen(f"{_server_url()}/health", timeout=3) as resp:
            data = json.loads(resp.read())
        queued = int(data.get("queued", 0))
        degraded = data.get("status") != "ok"
        return (degraded or queued >= HEALTH_QUEUE_PAUSE, queued)
    except (urllib.error.URLError, OSError, ValueError):
        # If we can't reach health, assume pressure — don't pile on.
        return (True, -1)


def _ingest_with_backpressure(
    files: list[Path],
    ledger: dict,
    verbose: bool,
    bytes_budget: int,
) -> tuple[int, int]:
    """Ingest files in batches, pausing when the server is under pressure
    or when bytes_budget is exhausted.

    Returns (count_ingested, bytes_pushed). The ledger is mutated in place
    so caller can _save_ledger periodically.
    """
    ingested = 0
    bytes_pushed = 0
    for i, f in enumerate(files):
        if bytes_pushed >= bytes_budget:
            print(
                f"  [backpressure] session byte cap reached "
                f"({bytes_pushed:,} >= {bytes_budget:,}); stopping ingestion",
                file=sys.stderr,
            )
            break
        # Check server pressure between batches
        if i > 0 and i % INGEST_BATCH_SIZE == 0:
            under, queued = _server_pressure()
            if under:
                print(
                    f"  [backpressure] server queued={queued} (>= {HEALTH_QUEUE_PAUSE}) — "
                    f"pausing {PAUSE_S}s before next batch",
                    file=sys.stderr,
                )
                time.sleep(PAUSE_S)
                # After the pause, recheck. If still pressured, abort the batch
                # rather than keep piling on. The user can retry later.
                under_again, queued_again = _server_pressure()
                if under_again:
                    print(
                        f"  [backpressure] server still pressured (queued={queued_again}); "
                        f"aborting batch, processed {ingested} files this run",
                        file=sys.stderr,
                    )
                    break

        h = _file_hash(f)
        if ledger.get(str(f)) == h:
            continue
        ok, pushed = _ingest_file(f, verbose)
        if ok:
            ledger[str(f)] = h
            ingested += 1
            bytes_pushed += pushed
    return ingested, bytes_pushed


def cmd_watch(args) -> int:
    root = Path(args.path).expanduser().resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    patterns = tuple(args.glob) if args.glob else DEFAULT_GLOB_PATTERNS
    max_files = args.max_files
    interval = args.interval
    verbose = args.verbose

    ledger = _load_ledger()
    print(f"watching {root} (patterns: {','.join(patterns)}, max-files: {max_files})")
    print(
        f"backpressure: batch={INGEST_BATCH_SIZE} files, queue-pause-at={HEALTH_QUEUE_PAUSE}, "
        f"session-byte-cap={SESSION_BYTE_CAP // (1024*1024)} MB"
    )

    bytes_remaining = SESSION_BYTE_CAP

    # Initial scan
    initial = _scan_files(root, patterns, max_files)
    print(f"initial scan: {len(initial)} files")
    ingested, pushed = _ingest_with_backpressure(initial, ledger, verbose, bytes_remaining)
    bytes_remaining -= pushed
    _save_ledger(ledger)
    print(f"initial: ingested={ingested} bytes_pushed={pushed:,}")

    if args.once:
        print(f"--once flag set, exiting after initial scan ({ingested} files processed)")
        return 0

    # Continuous polling
    print(f"polling every {interval}s (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(interval)
            current = _scan_files(root, patterns, max_files)
            new_or_changed, pushed = _ingest_with_backpressure(
                current, ledger, verbose, bytes_remaining,
            )
            bytes_remaining -= pushed
            if new_or_changed:
                print(f"[+{new_or_changed}] ingested new/changed files (bytes_remaining={bytes_remaining:,})")
                _save_ledger(ledger)
            if bytes_remaining <= 0:
                print(
                    "[backpressure] session byte cap exhausted — stopping watch loop. "
                    "Restart `fidelis watch` to begin a new session.",
                    file=sys.stderr,
                )
                return 0
    except KeyboardInterrupt:
        print("\nstopped")
        return 0
