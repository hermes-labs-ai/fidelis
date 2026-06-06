"""
fidelis sessions — searchable, purge-able session memory (MVP-A).

Wraps existing capability (query_sessions, ingest) and adds the purge path.
Honesty rules (non-negotiable, carried from the v1 spec):
  • Never print retrieval-benchmark numbers in product output (label only:
    "searchable session history — best on multi-turn queries").
  • Deletion claims stop at the INDEX. Source files in ~/.claude/projects and
    backups in ~/Backups/claude-sessions are NEVER touched.
  • No redaction-as-privacy-control. The honest levers are: don't-index
    (deny-list / --dry-run) and delete-after (purge).

Pure helpers (_iso_before, _collect_purge_targets, _filter_before) are separated
so they can be tested without a live ChromaDB / Ollama — "experiential contact"
with the LOGIC, not the services.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

COGITO_SESSIONS_DIR = Path.home() / ".cogito" / "session_ingest"

BACKUP_BOUNDARY = (
    "Source files in ~/.claude/projects and backups in ~/Backups/claude-sessions "
    "are NOT touched by design."
)
PRIVACY_NOTICE = (
    "Note: sessions contain full conversation history, stored locally in ~/.cogito "
    "(readable only by your user). Run `fidelis sessions purge` to remove. " + BACKUP_BOUNDARY
)


# ── Pure helpers (testable without services) ──────────────────────────────────

def _iso_before(end_ts: str, before: str) -> bool:
    """True if end_ts is strictly before the `before` date (YYYY-MM-DD).
    ISO-8601 lexicographic comparison is valid for the stored `...Z` format,
    so we avoid ChromaDB's broken string `$lt` where-filter entirely."""
    if not end_ts:
        return False
    return end_ts[:10] < before[:10]


def _collect_purge_targets(
    metadatas: list[dict], ids: list[str], mode: str, before: str | None
) -> tuple[list[str], list[str]]:
    """Return (target_ids, target_hashes) for purge. Pure: takes already-fetched
    ChromaDB rows, never calls the DB. mode is 'all' or 'before'.
    Collects ingest_hash too — required for ledger cleanup (spec step 3)."""
    target_ids: list[str] = []
    target_hashes: list[str] = []
    for cid, meta in zip(ids, metadatas):
        if mode == "all":
            keep = True
        elif mode == "before":
            keep = _iso_before(meta.get("end_ts", ""), before or "")
        else:
            keep = False
        if keep:
            target_ids.append(cid)
            h = meta.get("ingest_hash")
            if h:
                target_hashes.append(h)
    return target_ids, target_hashes


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── purge (the locked contract — §4 of the v1 spec) ───────────────────────────

def cmd_purge(args) -> int:
    if bool(args.all) == bool(args.before):
        print("Error: pass exactly one of --all or --before YYYY-MM-DD")
        return 2

    from fidelis.ingest_claude_sessions import _get_collection  # lazy: avoids chromadb import at CLI start

    col = _get_collection()
    entries = col.get(where={"mem_type": "session"}, include=["metadatas"])
    ids = entries.get("ids", []) or []
    metas = entries.get("metadatas", []) or []

    mode = "all" if args.all else "before"
    target_ids, target_hashes = _collect_purge_targets(metas, ids, mode, args.before)

    if not target_ids:
        print("No matching sessions to purge.")
        return 0

    # oldest/newest for the confirmation summary
    ends = sorted(m.get("end_ts", "") for m in metas if m.get("end_ts"))
    span = f"{ends[0][:10]}..{ends[-1][:10]}" if ends else "unknown"

    if args.dry_run:
        print(f"DRY-RUN: would remove {len(target_ids)} sessions (span {span}). Nothing written.")
        print(BACKUP_BOUNDARY)
        return 0

    if not args.yes:
        print(f"About to remove {len(target_ids)} sessions from the search index (span {span}).")
        print(BACKUP_BOUNDARY)
        resp = input(f"Delete {len(target_ids)} sessions? [y/N] ").strip().lower()
        if resp != "y":  # default N — accidental Enter must not delete
            print("Aborted.")
            return 1

    col.delete(ids=target_ids)

    # ledger cleanup: ingested.json maps {ingest_hash: chroma_id}
    ledger_path = COGITO_SESSIONS_DIR / "ingested.json"
    if ledger_path.exists():
        try:
            ledger = json.loads(ledger_path.read_text())
            for h in target_hashes:
                ledger.pop(h, None)
            ledger_path.write_text(json.dumps(ledger, indent=2))
        except Exception:  # noqa: best-effort; index delete already happened
            pass

    # append purge log
    COGITO_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(COGITO_SESSIONS_DIR / "purge_log.jsonl", "a") as f:
        f.write(json.dumps({
            "ts": _now_iso(), "mode": mode, "count": len(target_ids),
            "session_ids": target_ids,
        }) + "\n")

    print(f"Removed {len(target_ids)} sessions from the search index. {BACKUP_BOUNDARY}")
    return 0


# ── ingest / search / list / stats (thin wrappers over existing code) ─────────

def cmd_ingest(args) -> int:
    from fidelis.ingest_claude_sessions import ingest
    from datetime import datetime as _dt, timezone as _tz
    since = None
    if not args.all:
        # default: last 7 days (spec) unless --since given
        if args.since:
            since = _dt.fromisoformat(args.since).replace(tzinfo=_tz.utc)
        else:
            from datetime import timedelta
            since = _dt.now(_tz.utc) - timedelta(days=7)
    stats = ingest(since=since, dry_run=args.dry_run, verbose=args.verbose)
    print(f"scanned={stats['scanned']} stored={stats['stored']} "
          f"skipped_dedup={stats['skipped_dedup']} skipped_empty={stats['skipped_empty']} "
          f"errors={stats['errors']}")
    if not args.dry_run:
        print(PRIVACY_NOTICE)
    return 0


def cmd_search(args) -> int:
    from fidelis.recall_sessions import query_sessions
    results = query_sessions(args.query, top_k=args.limit)
    if args.raw:
        print(json.dumps([r.to_dict() for r in results], indent=2))
        return 0
    if not results:
        print("No matching sessions. (searchable session history — best on multi-turn queries)")
        return 0
    for i, r in enumerate(results, 1):
        print(f"[{i}] score {r.score:.4f}  {r.start_ts[:10]}  {r.turn_count} turns  {r.project_path}")
        print(f"    {r.matched_chunk}")
    return 0


def cmd_list(args) -> int:
    from fidelis.ingest_claude_sessions import _get_collection
    col = _get_collection()
    entries = col.get(where={"mem_type": "session"}, include=["metadatas"])
    metas = entries.get("metadatas", []) or []
    rows = sorted(metas, key=lambda m: m.get("start_ts", ""))
    if args.since:
        rows = [m for m in rows if m.get("start_ts", "")[:10] >= args.since[:10]]
    rows = rows[: args.limit]
    for m in rows:
        print(f"{m.get('start_ts','')[:10]}  {m.get('session_id','')[:8]}  "
              f"{m.get('turn_count','?')} turns  {m.get('project_path','')}")
    print(f"\n{len(rows)} sessions.")
    return 0


def cmd_stats(args) -> int:
    from fidelis.ingest_claude_sessions import _get_collection
    col = _get_collection()
    entries = col.get(where={"mem_type": "session"}, include=["metadatas"])
    metas = entries.get("metadatas", []) or []
    if not metas:
        print("No sessions indexed. Run `fidelis sessions ingest`.")
        return 0
    total_turns = sum(int(m.get("turn_count", 0) or 0) for m in metas)
    ends = sorted(m.get("end_ts", "") for m in metas if m.get("end_ts"))
    projects: dict[str, int] = {}
    for m in metas:
        projects[m.get("project_path", "?")] = projects.get(m.get("project_path", "?"), 0) + 1
    print(f"indexed sessions: {len(metas)}")
    print(f"date range: {ends[0][:10] if ends else '?'} .. {ends[-1][:10] if ends else '?'}")
    print(f"total turns: {total_turns}  avg/session: {total_turns // max(len(metas),1)}")
    print("by project:")
    for p, n in sorted(projects.items(), key=lambda x: -x[1])[:10]:
        print(f"  {n:>5}  {p}")
    print("\nTool/skill call breakdown is not available in this version.")
    return 0
