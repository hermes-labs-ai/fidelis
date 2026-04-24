"""Tests for cogito.telemetry — the escalation-rate observability layer."""
from __future__ import annotations

import json

import pytest

from fidelis import telemetry


@pytest.fixture
def iso_log(tmp_path, monkeypatch):
    p = tmp_path / "escalation.log"
    monkeypatch.setattr(telemetry, "_LOG_PATH", p)
    return p


def test_rate_empty_when_no_log(iso_log):
    out = telemetry.rate(window_n=100)
    assert out == {"n": 0, "escalated": 0, "rate": 0.0, "by_route": {}}


def test_record_and_rate_roundtrip(iso_log):
    telemetry.record("llm", escalated=True, top1_score=0.5, gap=0.02)
    telemetry.record("llm", escalated=True)
    telemetry.record("default", escalated=False, top1_score=0.9, gap=0.2)
    telemetry.record("skip", escalated=False)
    out = telemetry.rate(window_n=100)
    assert out["n"] == 4
    assert out["escalated"] == 2
    assert out["rate"] == 0.5
    assert out["by_route"]["llm"] == {"n": 2, "escalated": 2}
    assert out["by_route"]["default"] == {"n": 1, "escalated": 0}


def test_record_never_raises_on_unwritable_path(monkeypatch, tmp_path):
    # Point the log at a path where mkdir will fail, verify no exception escapes.
    bad = tmp_path / "not" / "a" / "dir" / "escalation.log"
    monkeypatch.setattr(telemetry, "_LOG_PATH", bad)
    monkeypatch.setattr(telemetry, "_ensure_parent", lambda _p: (_ for _ in ()).throw(OSError("nope")))
    # Should not raise:
    telemetry.record("llm", escalated=True)


def test_window_n_truncates_to_last_n(iso_log):
    for i in range(10):
        telemetry.record("llm", escalated=(i % 2 == 0))
    out = telemetry.rate(window_n=4)
    assert out["n"] == 4
    # last 4 of 10: indices 6,7,8,9 → escalated=True,False,True,False → 2/4
    assert out["escalated"] == 2


def test_corrupt_line_counted_in_n_but_not_classified(iso_log):
    iso_log.parent.mkdir(parents=True, exist_ok=True)
    iso_log.write_text(
        json.dumps({"ts": 1, "route": "llm", "escalated": True}) + "\n"
        + "garbage line\n"
        + json.dumps({"ts": 2, "route": "llm", "escalated": False}) + "\n"
    )
    out = telemetry.rate(window_n=100)
    assert out["n"] == 3  # all three lines counted in n
    assert out["escalated"] == 1  # only the well-formed escalated record
    assert out["by_route"]["llm"]["n"] == 2  # only classified rows
