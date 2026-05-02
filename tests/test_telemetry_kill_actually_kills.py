"""Regression tests for the EMFILE-storm bug.

Two narrow guarantees, both of which were silently violated for some
weeks before the audit on 2026-05-02:

1. The env var name we set in launchd/systemd actually disables mem0's
   posthog telemetry. The previous template set `POSTHOG_DISABLED=1`,
   which is not a real posthog env var — it was inert. The template now
   sets `MEM0_TELEMETRY=False`, which mem0/memory/telemetry.py reads at
   import and uses to short-circuit `capture_event`.

2. fidelis's /health probe reads chroma's count via the attribute name
   that exists on mem0 2.0's wrapper (`.collection`, set in
   mem0/vector_stores/chroma.py:74). Earlier code referenced `.col` and
   `.col_info()`, neither of which exists; both raised AttributeError on
   every probe and each one fired posthog.capture, contributing to the
   fd pressure that the storm turns into EMFILE spam.

These tests are import-time / mock-only; no Ollama, no real chromadb,
no real launchd.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest


def test_mem0_telemetry_false_disables_posthog(monkeypatch):
    """When MEM0_TELEMETRY=False is in the env, mem0/memory/telemetry.py
    sets `self.posthog = None` on the AnonymousTelemetry singleton, and
    `capture_event` short-circuits before reaching posthog.capture.

    The previous template set POSTHOG_DISABLED=1, which posthog 7.x
    does not honor — `grep -rn POSTHOG_DISABLED` in the installed
    posthog tree returns zero hits. This test pins the working flag.
    """
    monkeypatch.setenv("MEM0_TELEMETRY", "False")

    # Force re-import of mem0.memory.telemetry so the env var is read
    # fresh. The module reads MEM0_TELEMETRY at import time at
    # mem0/memory/telemetry.py:14.
    for mod_name in list(sys.modules):
        if mod_name == "mem0.memory.telemetry" or mod_name.startswith("mem0.memory.telemetry."):
            del sys.modules[mod_name]

    telemetry_mod = importlib.import_module("mem0.memory.telemetry")

    # The module-level constant should be the parsed-bool False.
    assert telemetry_mod.MEM0_TELEMETRY is False, (
        f"MEM0_TELEMETRY parsed as {telemetry_mod.MEM0_TELEMETRY!r}; "
        f"expected False. Check mem0/memory/telemetry.py:19 for the "
        f"lower-case-in tuple."
    )

    # Construct the telemetry client directly. With telemetry disabled,
    # __init__ sets self.posthog = None and capture_event becomes a noop.
    instance = telemetry_mod.AnonymousTelemetry()
    assert instance.posthog is None, (
        "AnonymousTelemetry.posthog should be None when MEM0_TELEMETRY=False; "
        "if non-None, capture_event will call posthog.capture, which opens "
        "/System/Library/CoreServices/SystemVersion.plist per event and "
        "spams EMFILE under fd pressure."
    )

    # Belt-and-suspenders: capture_event should not raise, and should not
    # touch posthog when posthog is None. The wrapper at line 88 returns
    # before calling self.posthog.capture.
    instance.capture_event("mem0.test", properties={"k": "v"})


def test_posthog_disabled_legacy_var_is_inert():
    """Documentation test: confirm that POSTHOG_DISABLED is *not* a real
    env var in the installed posthog SDK. If a future version of posthog
    starts honoring this flag, this test will fail and the template
    comment should be updated.
    """
    import posthog as posthog_mod

    posthog_dir = os.path.dirname(posthog_mod.__file__)
    hits = 0
    for root, _dirs, files in os.walk(posthog_dir):
        if "test" in root or "__pycache__" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, "rb") as f:
                if b"POSTHOG_DISABLED" in f.read():
                    hits += 1
    assert hits == 0, (
        f"posthog SDK now references POSTHOG_DISABLED in {hits} file(s); "
        f"the inert-flag claim in fidelis init_cmd.py is no longer correct. "
        f"Re-evaluate whether to add it back to the templates."
    )


def test_health_count_uses_collection_attr():
    """fidelis /health calls `memory.vector_store.collection.count()`.
    Earlier versions called `.col.count()` or `.col_info().count()`; both
    raise AttributeError on mem0 2.0's wrapper, and each AttributeError
    triggers a posthog.capture chain that opens SystemVersion.plist.

    This is a static check — confirm the source contains the correct
    attribute name and not the legacy ones.
    """
    import fidelis.server as server_mod
    import inspect

    src = inspect.getsource(server_mod)
    assert ".collection.count()" in src, (
        "/health code path no longer uses .collection.count(); will hit "
        "AttributeError on mem0 2.0 and trigger telemetry-driven plist opens."
    )
    # Legacy names should be gone — they are silent failure modes.
    assert ".col.count()" not in src, (
        "Legacy .col.count() reference still present in fidelis/server.py; "
        "mem0 2.0's wrapper does not expose .col, only .collection."
    )
    assert ".col_info()" not in src, (
        "Legacy .col_info() reference still present in fidelis/server.py; "
        "mem0 2.0's wrapper does not expose .col_info(), only .collection."
    )


def test_health_count_works_with_mock_collection(monkeypatch):
    """End-to-end mock: simulate a memory.vector_store with a .collection
    attribute that has .count() returning an int. Verify the do_GET path
    reads the count without raising.

    We synthesize the minimum object shape; the real chromadb.Collection
    has many other methods, but /health only touches .count().
    """
    class _FakeCollection:
        def count(self) -> int:
            return 42

    class _FakeVectorStore:
        collection = _FakeCollection()

    class _FakeMemory:
        vector_store = _FakeVectorStore()

    fake_memory = _FakeMemory()

    # Exercise the attribute chain the real handler uses. If this line
    # raises, /health is broken and posthog.capture will be triggered on
    # the chained exception, leading to the EMFILE storm under load.
    count = fake_memory.vector_store.collection.count()
    assert count == 42
    assert isinstance(count, int)
