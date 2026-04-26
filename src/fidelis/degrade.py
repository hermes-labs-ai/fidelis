"""Graceful-degradation layer for fidelis writes.

When the upstream LLM (Ollama / mem0) is unreachable, we MUST NOT lose the
write. Instead, queue it locally as JSONL and let a sync job replay later.

This module exists because of the 2026-04-19 incident: Ollama's socket layer
broke under Python 3.14, every `cogito add` returned HTTP 500, and a full
session of memory was silently lost. The Hermes Seal v1 made this gap explicit.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

MAX_ATTEMPTS = 5


def _queue_dir() -> Path:
    """Resolve queue directory from env at call time so tests can isolate."""
    env = os.environ.get("FIDELIS_QUEUE_DIR") or os.environ.get("COGITO_QUEUE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cogito" / "queue"


def _dead_dir() -> Path:
    return _queue_dir() / "dead"


# Backwards-compat module-level paths. These resolve at import time and are kept
# so existing callers / scripts that reference QUEUE_DIR still work, but the
# functions below always use _queue_dir() to honour env overrides set after
# import (test isolation).
QUEUE_DIR = _queue_dir()
DEAD_DIR = _dead_dir()


def _ensure_queue() -> Path:
    qdir = _queue_dir()
    qdir.mkdir(parents=True, exist_ok=True)
    return qdir


def _ensure_dead() -> Path:
    ddir = _dead_dir()
    ddir.mkdir(parents=True, exist_ok=True)
    return ddir


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically: write to sibling tmp then os.replace.

    SIGKILL between write and replace leaves either the old file (if it
    existed) or no file — never a half-written corrupt JSON.
    """
    tmp = path.parent / (path.name + ".tmp")
    data = json.dumps(payload).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def queue_write(text: str, user_id: str, kind: str = "add") -> str:
    """Append a memory write to the local queue and return its id."""
    qdir = _ensure_queue()
    mid = str(uuid.uuid4())
    rec = {
        "id": mid, "ts": time.time(), "kind": kind, "user_id": user_id,
        "text": text, "attempts": 0,
    }
    qfile = qdir / f"{int(rec['ts'])}-{mid}.json"
    _atomic_write_json(qfile, rec)
    return mid


def queued_count() -> int:
    qdir = _queue_dir()
    if not qdir.exists():
        return 0
    return len([p for p in qdir.glob("*.json") if p.is_file()])


def dead_count() -> int:
    ddir = _dead_dir()
    if not ddir.exists():
        return 0
    return len(list(ddir.glob("*.json")))


def safe_add(memory, text: str, user_id: str, kind: str = "add") -> dict:
    """Try to write through mem0; on dependency failure, queue locally.

    Returns:
      {"status": "stored", "extracted": [...]}  — happy path
      {"status": "queued", "id": "...", "reason": "..."}  — fallback path
    """
    try:
        if kind == "store":
            mid = str(uuid.uuid4())
            vector = memory.embedding_model.embed(text)
            memory.vector_store.insert(
                vectors=[vector],
                payloads=[{"data": text, "user_id": user_id}],
                ids=[mid],
            )
            return {"status": "stored", "id": mid, "extracted": [text]}
        result = memory.add(text, user_id=user_id)
        extracted = [m.get("memory", "") for m in result.get("results", [])]
        return {"status": "stored", "extracted": extracted}
    except Exception as e:
        mid = queue_write(text, user_id, kind=kind)
        return {"status": "queued", "id": mid, "reason": f"{type(e).__name__}: {e}"}


def replay_queue(memory, user_id: str) -> dict:
    """Replay queued writes through mem0. Removes successfully replayed records.

    Two-stage fallback for "add"-kind items: first try the full mem0.add() path
    (which uses the LLM for fact extraction). If that fails (most commonly
    because the configured LLM model is not available on Ollama), fall back to
    a direct verbatim vector insert — equivalent to /store. The user gets
    something recallable instead of a write that's stuck in the queue forever.

    Per-item retry budget: each failure increments rec["attempts"]. After
    MAX_ATTEMPTS the record is moved to the dead-letter queue so a permanently
    poisoned write cannot keep the replay loop hot forever.
    """
    qdir = _queue_dir()
    if not qdir.exists():
        return {"replayed": 0, "remaining": 0, "replayed_verbatim": 0, "failed": 0, "dead_lettered": 0}
    replayed = 0
    replayed_verbatim = 0
    failed = 0
    dead_lettered = 0
    ddir = _dead_dir()
    for qfile in sorted(qdir.glob("*.json")):
        if not qfile.is_file():
            continue
        try:
            rec = json.loads(qfile.read_text())
        except Exception:  # noqa: silent — corrupted queue file moved to dead-letter for inspection
            _move_to_dead(qfile)
            dead_lettered += 1
            continue
        try:
            if rec.get("kind") == "store":
                _replay_verbatim(memory, rec)
            else:
                try:
                    memory.add(rec["text"], user_id=rec["user_id"])
                except Exception:
                    _replay_verbatim(memory, rec)
                    replayed_verbatim += 1
            qfile.unlink()
            replayed += 1
        except Exception as e:
            attempts = int(rec.get("attempts", 0)) + 1
            rec["attempts"] = attempts
            rec["last_error"] = f"{type(e).__name__}: {e}"
            if attempts >= MAX_ATTEMPTS:
                _ensure_dead()
                _atomic_write_json(ddir / qfile.name, rec)
                try:
                    qfile.unlink()
                except OSError:  # noqa: silent — file already moved
                    pass
                dead_lettered += 1
            else:
                _atomic_write_json(qfile, rec)
                failed += 1
    remaining = len([p for p in qdir.glob("*.json") if p.is_file()])
    return {
        "replayed": replayed,
        "replayed_verbatim": replayed_verbatim,
        "failed": failed,
        "dead_lettered": dead_lettered,
        "remaining": remaining,
    }


def _move_to_dead(qfile: Path) -> None:
    _ensure_dead()
    target = _dead_dir() / qfile.name
    try:
        qfile.rename(target)
    except OSError:  # noqa: silent — best-effort dead-letter; if rename fails the file stays for next sweep
        pass


def _replay_verbatim(memory, rec: dict) -> None:
    """Insert a queued record directly into the vector store without LLM
    extraction. Used as the safe-add fast path AND as the LLM-fallback path
    inside replay_queue.

    Idempotent under retry: chroma's insert with a duplicate id raises; we
    catch and treat as success because the prior partial-attempt already
    persisted the vector.
    """
    mid = rec.get("id") or str(uuid.uuid4())
    vector = memory.embedding_model.embed(rec["text"])
    try:
        memory.vector_store.insert(
            vectors=[vector],
            payloads=[{"data": rec["text"], "user_id": rec["user_id"]}],
            ids=[mid],
        )
    except Exception as e:
        # chromadb raises on duplicate ids — that's a successful prior insert.
        # Any other error propagates so the caller can count it as a failure.
        if "exist" in str(e).lower() or "duplicate" in str(e).lower():
            return
        raise
