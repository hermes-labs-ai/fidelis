"""Graceful-degradation layer for cogito writes.

When the upstream LLM (Ollama / mem0) is unreachable, we MUST NOT lose the
write. Instead, queue it locally as JSONL and let a sync job replay later.

This module exists because of the 2026-04-19 incident: Ollama's socket layer
broke under Python 3.14, every `cogito add` returned HTTP 500, and a full
session of memory was silently lost. The Hermes Seal v1 made this gap explicit.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

QUEUE_DIR = Path.home() / ".cogito" / "queue"


def _ensure_queue() -> Path:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    return QUEUE_DIR


def queue_write(text: str, user_id: str, kind: str = "add") -> str:
    """Append a memory write to the local queue and return its id."""
    qdir = _ensure_queue()
    mid = str(uuid.uuid4())
    rec = {"id": mid, "ts": time.time(), "kind": kind, "user_id": user_id, "text": text}
    qfile = qdir / f"{int(rec['ts'])}-{mid}.json"
    qfile.write_text(json.dumps(rec))
    return mid


def queued_count() -> int:
    if not QUEUE_DIR.exists():
        return 0
    return len(list(QUEUE_DIR.glob("*.json")))


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
    """Replay queued writes through mem0. Removes successfully replayed records."""
    if not QUEUE_DIR.exists():
        return {"replayed": 0, "remaining": 0}
    replayed = 0
    failed = 0
    for qfile in sorted(QUEUE_DIR.glob("*.json")):
        try:
            rec = json.loads(qfile.read_text())
        except Exception:  # noqa: silent — corrupted queue file is logged below by 'failed' count
            failed += 1
            continue
        try:
            if rec.get("kind") == "store":
                mid = rec["id"]
                vector = memory.embedding_model.embed(rec["text"])
                memory.vector_store.insert(
                    vectors=[vector],
                    payloads=[{"data": rec["text"], "user_id": rec["user_id"]}],
                    ids=[mid],
                )
            else:
                memory.add(rec["text"], user_id=rec["user_id"])
            qfile.unlink()
            replayed += 1
        except Exception:
            failed += 1
    remaining = len(list(QUEUE_DIR.glob("*.json")))
    return {"replayed": replayed, "failed": failed, "remaining": remaining}
