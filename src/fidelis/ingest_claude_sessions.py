"""
cogito.ingest_claude_sessions — Claude Code session ingestion for cogito-ergo.

Reads ~/.claude/projects/*/SESSION_ID.jsonl, extracts role-structured user+assistant
turn pairs, and stores each session as ONE session-typed memory in cogito's ChromaDB.

Usage:
    python3 -m cogito.ingest_claude_sessions [--since YYYY-MM-DD] [--dry-run]

Privacy note:
    Claude Code sessions contain full conversation history. This script is EXPLICIT
    — nothing runs automatically. Run it intentionally. Data stays local (ChromaDB +
    Ollama embedding, no cloud calls by default).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
COGITO_STORE = Path.home() / ".cogito" / "store"
COGITO_SESSIONS_DIR = Path.home() / ".cogito" / "session_ingest"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_PREFIX = "search_document: "
USER_ID = "agent"
MAX_TEXT_CHARS = 2000          # matches cogito's safe truncation limit
MAX_TURNS_PER_SESSION = 100    # prevent runaway sessions dominating the index


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed(text: str) -> list[float]:
    """Embed via Ollama nomic-embed-text. Raises on failure."""
    payload = json.dumps({
        "model": EMBED_MODEL,
        "input": [EMBED_PREFIX + text[:MAX_TEXT_CHARS]],
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    embeddings = result.get("embeddings", [])
    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings")
    return embeddings[0]


# ── JSONL Parsing ─────────────────────────────────────────────────────────────

def _extract_text(content) -> str:
    """Extract plain text from a Claude Code message content field."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item["text"].strip())
        return " ".join(parts)
    return ""


def _parse_jsonl(path: Path) -> list[dict]:
    """Parse a .jsonl session file → list of {role, content, ts} dicts."""
    turns = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                t = obj.get("type")
                if t not in ("user", "assistant"):
                    continue

                msg = obj.get("message", {})
                role = msg.get("role", t)
                content = msg.get("content", "")
                text = _extract_text(content)

                # Skip empty, meta-only, or tool-use-only turns
                if not text or len(text) < 5:
                    continue

                # Skip side-chain messages (parallel tool results, not main thread)
                if obj.get("isSidechain"):
                    continue

                ts = obj.get("timestamp", "")
                turns.append({"role": role, "content": text, "ts": ts})

                if len(turns) >= MAX_TURNS_PER_SESSION:
                    break
    except (OSError, PermissionError):  # noqa: silent — unreadable session file → skip; partial turns are still useful
        pass
    return turns


# ── Session Discovery ─────────────────────────────────────────────────────────

def _iter_sessions(since: datetime | None = None) -> Iterator[tuple[str, Path, str]]:
    """
    Yield (session_id, jsonl_path, project_path) for all Claude Code sessions.
    Filters by since if given.
    """
    if not CLAUDE_PROJECTS.exists():
        return

    for project_dir in CLAUDE_PROJECTS.iterdir():
        if not project_dir.is_dir():
            continue
        project_path = project_dir.name  # e.g. '-Users-rbr-lpci'

        for jsonl_file in project_dir.glob("*.jsonl"):
            session_id = jsonl_file.stem

            if since is not None:
                mtime = datetime.fromtimestamp(
                    jsonl_file.stat().st_mtime, tz=timezone.utc
                )
                if mtime < since:
                    continue

            yield session_id, jsonl_file, project_path


# ── Dedup Ledger ──────────────────────────────────────────────────────────────

def _ledger_path() -> Path:
    COGITO_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return COGITO_SESSIONS_DIR / "ingested.json"


def _load_ledger() -> dict[str, str]:
    """Returns {ingest_hash: chroma_id}."""
    lp = _ledger_path()
    if lp.exists():
        try:
            return json.loads(lp.read_text())
        except Exception:  # noqa: silent — corrupted ledger → start fresh; partial re-ingest is fine
            pass
    return {}


def _save_ledger(ledger: dict[str, str]) -> None:
    _ledger_path().write_text(json.dumps(ledger, indent=2))


def _make_ingest_hash(session_id: str, turns: list[dict]) -> str:
    """Stable hash over session_id + turn contents for idempotency."""
    payload = session_id + json.dumps(
        [{"role": t["role"], "content": t["content"][:200]} for t in turns[:10]]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ── Session Text Representation ───────────────────────────────────────────────

def _session_to_text(turns: list[dict]) -> str:
    """
    Build a flat text representation of the session for embedding.
    Favours user turns (user intent is what we want to retrieve).
    """
    parts = []
    for turn in turns:
        prefix = "User" if turn["role"] == "user" else "Assistant"
        snippet = turn["content"][:400]
        parts.append(f"{prefix}: {snippet}")
    return "\n".join(parts)[:MAX_TEXT_CHARS]


# ── ChromaDB Interface ────────────────────────────────────────────────────────

def _get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(COGITO_STORE))
    return client.get_collection("cogito_main")


def _store_session(
    col,
    session_id: str,
    project_path: str,
    turns: list[dict],
    ingest_hash: str,
) -> str:
    """Store one session as a session-typed memory. Returns chroma ID."""
    text = _session_to_text(turns)
    embedding = _embed(text)

    start_ts = turns[0]["ts"] if turns else ""
    end_ts = turns[-1]["ts"] if turns else ""

    chroma_id = str(uuid.uuid4())
    col.add(
        ids=[chroma_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "user_id": USER_ID,
            "data": text,
            "mem_type": "session",
            "session_id": session_id,
            "project_path": project_path,
            "turns_json": json.dumps(turns),
            "turn_count": len(turns),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "ingest_hash": ingest_hash,
        }],
    )
    return chroma_id


# ── Main Ingestion Flow ───────────────────────────────────────────────────────

def ingest(
    since: datetime | None = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Ingest Claude Code sessions into cogito.

    Returns stats dict: {scanned, skipped_dedup, skipped_empty, stored, errors}
    """
    ledger = _load_ledger()
    col = None if dry_run else _get_collection()

    stats = {"scanned": 0, "skipped_dedup": 0, "skipped_empty": 0, "stored": 0, "errors": 0}

    for session_id, jsonl_path, project_path in _iter_sessions(since=since):
        stats["scanned"] += 1

        turns = _parse_jsonl(jsonl_path)
        if len(turns) < 2:
            stats["skipped_empty"] += 1
            if verbose:
                print(f"  SKIP (empty) {session_id[:8]}")
            continue

        ingest_hash = _make_ingest_hash(session_id, turns)
        if ingest_hash in ledger:
            stats["skipped_dedup"] += 1
            if verbose:
                print(f"  SKIP (dedup) {session_id[:8]}")
            continue

        if dry_run:
            user_snippet = next(
                (t["content"][:60] for t in turns if t["role"] == "user"), ""
            )
            print(
                f"  DRY-RUN  {session_id[:8]} | {len(turns)} turns | {project_path} | {user_snippet!r}"
            )
            stats["stored"] += 1
            continue

        try:
            chroma_id = _store_session(col, session_id, project_path, turns, ingest_hash)
            ledger[ingest_hash] = chroma_id
            stats["stored"] += 1
            if verbose:
                print(f"  STORED   {session_id[:8]} → {chroma_id[:8]} ({len(turns)} turns)")
        except Exception as exc:
            stats["errors"] += 1
            if verbose:
                print(f"  ERROR    {session_id[:8]}: {exc}")

    if not dry_run:
        _save_ledger(ledger)

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Ingest Claude Code sessions into cogito-ergo memory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Privacy: sessions contain full conversation history. Data stays local (ChromaDB +
Ollama). No cloud calls unless you enable flagship tier explicitly.

Examples:
    # Preview what would be ingested (last 7 days)
    python3 -m cogito.ingest_claude_sessions --since 2026-04-11 --dry-run

    # Actually ingest all sessions from last 7 days
    python3 -m cogito.ingest_claude_sessions --since 2026-04-11

    # Ingest everything (may take a while)
    python3 -m cogito.ingest_claude_sessions
""",
    )
    parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        help="Only ingest sessions modified after this date",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be ingested without writing anything",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-session status",
    )
    args = parser.parse_args(argv)

    since = None
    if args.since:
        try:
            since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            print(f"ERROR: --since must be YYYY-MM-DD, got: {args.since!r}", file=sys.stderr)
            sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"[cogito-ingest] {mode} — scanning {CLAUDE_PROJECTS}")
    if since:
        print(f"[cogito-ingest] Since: {args.since}")

    stats = ingest(since=since, dry_run=args.dry_run, verbose=args.verbose or args.dry_run)

    print("\n[cogito-ingest] Done.")
    print(f"  Scanned:       {stats['scanned']}")
    print(f"  Stored:        {stats['stored']}")
    print(f"  Skipped dedup: {stats['skipped_dedup']}")
    print(f"  Skipped empty: {stats['skipped_empty']}")
    print(f"  Errors:        {stats['errors']}")


if __name__ == "__main__":
    main()
