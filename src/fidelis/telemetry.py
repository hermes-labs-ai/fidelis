"""Escalation telemetry — makes the 80.2% vs 10% escalation bug observable.

STATUS.md documents:
  "Escalation rate: 80.2% actual vs 10% intended. Threshold was calibrated
  on 65 SSU/multi-session questions; preference questions have different
  confidence distributions. In production this is 8x the planned cost."

This module is the observability layer. It does not fix the calibration —
it makes the rate measurable without reading benchmark logs, so cost drift
is visible from `/stats` and regressions are catchable in tests.

Design:
  - Append-only JSONL at `~/.cogito/escalation.log`
  - `record(route, escalated, confidence)` is cheap and crash-safe (best-effort)
  - `rate(window_n=100)` summarises the last N decisions
  - Integration into recall_hybrid.py is an additive call — if the log
    path is unwritable, record() is a no-op and retrieval is unaffected.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_LOG_PATH = Path(os.environ.get("COGITO_ESCALATION_LOG", str(Path.home() / ".cogito" / "escalation.log")))


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: silent — telemetry must never break the caller
        pass


def record(route: str, escalated: bool, top1_score: float | None = None, gap: float | None = None) -> None:
    """Append one escalation-decision record. Never raises."""
    rec = {
        "ts": time.time(),
        "route": route,
        "escalated": bool(escalated),
        "top1": top1_score,
        "gap": gap,
    }
    try:
        _ensure_parent(_LOG_PATH)
        with _LOG_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: silent — telemetry must never break the caller
        return


def rate(window_n: int = 100, path: Path | None = None) -> dict[str, Any]:
    """Summarise the last N decisions. Returns zero-filled dict if log missing."""
    p = path or _LOG_PATH
    if not p.exists():
        return {"n": 0, "escalated": 0, "rate": 0.0, "by_route": {}}
    lines = p.read_text().splitlines()[-window_n:]
    n = len(lines)
    esc = 0
    by_route: dict[str, dict[str, int]] = {}
    for line in lines:
        try:
            r = json.loads(line)
        except Exception:  # noqa: silent — a corrupted record is counted in n but not classified
            continue
        route = r.get("route", "unknown")
        entry = by_route.setdefault(route, {"n": 0, "escalated": 0})
        entry["n"] += 1
        if r.get("escalated"):
            entry["escalated"] += 1
            esc += 1
    return {
        "n": n,
        "escalated": esc,
        "rate": (esc / n) if n else 0.0,
        "by_route": by_route,
    }


def reset(path: Path | None = None) -> None:
    """Truncate the log. Intended for tests and manual resets."""
    p = path or _LOG_PATH
    if p.exists():
        p.unlink()
