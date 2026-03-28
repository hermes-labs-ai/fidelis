"""
cogito seed — bulk-seed the memory store from markdown/text files.

Reads source files, chunks them by section, and posts each chunk to the
running cogito server's /add endpoint. The extraction LLM inside mem0
decides which facts to store — this script never decides what's memorable.

Usage:
    cogito seed ~/memory/                          # seed all .md files
    cogito seed ~/memory/ ~/notes/sessions/        # multiple dirs
    cogito seed --file ~/notes/today.md            # single file
    cogito seed ~/memory/ --dry-run                # show what would be sent
    cogito seed ~/memory/ --force                  # re-seed even unchanged files
    cogito seed ~/memory/ --glob "*.md"            # filter by pattern

State is tracked in ~/.cogito/seeded.json (file path → mtime hash).
Re-run at any time — only changed or new files are seeded.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


# ── chunking ───────────────────────────────────────────────────────────────

def _chunks_from_file(path: Path, max_chars: int = 1500) -> list[str]:
    """
    Split a markdown file into chunks by ## heading.
    Each chunk is at most max_chars. Long chunks are split further by
    paragraph. Short consecutive sections may stay merged.
    """
    text = path.read_text(errors="replace")
    if not text.strip():
        return []

    # Split on markdown section headings (## or ###)
    import re
    sections = re.split(r"(?m)^(?=#{1,3} )", text)
    sections = [s.strip() for s in sections if s.strip()]

    chunks: list[str] = []
    for section in sections:
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Split long sections by paragraph
            paras = re.split(r"\n{2,}", section)
            buf = ""
            for para in paras:
                para = para.strip()
                if not para:
                    continue
                if buf and len(buf) + len(para) + 2 > max_chars:
                    chunks.append(buf)
                    buf = para
                else:
                    buf = (buf + "\n\n" + para).strip() if buf else para
            if buf:
                chunks.append(buf)

    return [c for c in chunks if len(c.strip()) >= 40]


# ── state tracking ──────────────────────────────────────────────────────────

def _state_path() -> Path:
    p = Path.home() / ".cogito" / "seeded.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_state() -> dict[str, str]:
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict[str, str]) -> None:
    _state_path().write_text(json.dumps(state, indent=2))


def _file_hash(path: Path) -> str:
    """Cheap hash: file size + mtime. Fast and sufficient for change detection."""
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


# ── HTTP ────────────────────────────────────────────────────────────────────

def _add(base_url: str, text: str, timeout: int = 120) -> tuple[int, list[str]]:
    """POST /add and return (count, memories)."""
    data = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        f"{base_url}/add",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        return result.get("count", 0), result.get("memories", [])
    except urllib.error.URLError as e:
        raise RuntimeError(f"Server not reachable at {base_url}: {e}") from e


def _check_server(base_url: str) -> int:
    """Return memory count from /health, or raise if unreachable."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            result = json.loads(resp.read())
            return result.get("count", 0)
    except urllib.error.URLError as e:
        raise RuntimeError(f"cogito server not reachable at {base_url}") from e


# ── core ────────────────────────────────────────────────────────────────────

def seed(
    sources: list[Path],
    base_url: str,
    glob_pattern: str = "*.md",
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
    delay_ms: int = 200,
) -> dict:
    """
    Seed the cogito store from source dirs/files.

    Returns a summary dict: {files_processed, files_skipped, chunks_sent,
                              facts_added, errors}.
    """
    state = _load_state()
    stats = {
        "files_processed": 0,
        "files_skipped": 0,
        "chunks_sent": 0,
        "facts_added": 0,
        "errors": 0,
    }

    # Collect files
    all_files: list[Path] = []
    for src in sources:
        src = src.expanduser().resolve()
        if src.is_file():
            all_files.append(src)
        elif src.is_dir():
            matched = sorted(src.rglob(glob_pattern))
            all_files.extend(matched)
        else:
            print(f"  [skip] not found: {src}", file=sys.stderr)

    if not all_files:
        print("No files found to seed.")
        return stats

    # Check server (unless dry run)
    if not dry_run:
        try:
            count_before = _check_server(base_url)
            print(f"[cogito seed] Server OK — {count_before} memories before seeding")
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Start with: cogito-server", file=sys.stderr)
            sys.exit(1)

    print(f"[cogito seed] {len(all_files)} file(s) to consider\n")

    for path in all_files:
        file_key = str(path)
        file_hash = _file_hash(path)

        if not force and state.get(file_key) == file_hash:
            stats["files_skipped"] += 1
            if verbose:
                print(f"  [=] {path.name}  (unchanged, skip)")
            continue

        chunks = _chunks_from_file(path)
        if not chunks:
            if verbose:
                print(f"  [0] {path.name}  (empty)")
            state[file_key] = file_hash
            continue

        rel = path.name
        print(f"  [→] {rel}  {len(chunks)} chunk(s)")

        file_facts = 0
        file_errors = 0

        for i, chunk in enumerate(chunks, 1):
            if dry_run:
                preview = chunk[:80].replace("\n", " ")
                print(f"      [{i}/{len(chunks)}] DRY: {preview!r}...")
                stats["chunks_sent"] += 1
                continue

            try:
                count, memories = _add(base_url, chunk)
                file_facts += count
                stats["chunks_sent"] += 1
                stats["facts_added"] += count
                if verbose and memories:
                    for m in memories:
                        print(f"      ✓ {m[:80]}")
                elif verbose:
                    print(f"      [{i}/{len(chunks)}] +{count} facts")
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000)
            except RuntimeError as e:
                print(f"      [!] chunk {i}: {e}", file=sys.stderr)
                file_errors += 1
                stats["errors"] += 1

        if not dry_run and file_errors == 0:
            state[file_key] = file_hash

        stats["files_processed"] += 1
        print(f"      done — +{file_facts} facts  ({file_errors} errors)")

    if not dry_run:
        _save_state(state)
        try:
            count_after = _check_server(base_url)
            print(f"\n[cogito seed] Done. {count_after} memories in store.")
        except Exception:
            pass

    print(f"\n  files processed : {stats['files_processed']}")
    print(f"  files skipped   : {stats['files_skipped']}  (unchanged)")
    print(f"  chunks sent     : {stats['chunks_sent']}")
    print(f"  facts added     : {stats['facts_added']}")
    if stats["errors"]:
        print(f"  errors          : {stats['errors']}")

    return stats
