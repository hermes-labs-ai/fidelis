"""Gate tests for MVP-A `fidelis sessions`. Pure-logic — no ChromaDB/Ollama needed.
These are the acceptance criteria AS TESTS (the contact-gate): the build is not
done until these pass.
"""
import importlib

from fidelis import sessions_cmd as S
from fidelis import ingest_claude_sessions as I


# ── deny-list (privacy lever) ────────────────────────────────────────────────

def test_denylist_excludes_matching_project():
    assert I._excluded("-Users-rbr-client-secret", ["client-secret"]) is True

def test_denylist_keeps_nonmatching_project():
    assert I._excluded("-Users-rbr-lpci", ["client-secret"]) is False

def test_denylist_empty_excludes_nothing():
    assert I._excluded("-Users-rbr-anything", []) is False
    assert I._excluded("-Users-rbr-anything", None) is False

def test_denylist_env_parsing(monkeypatch):
    monkeypatch.setenv("FIDELIS_SESSIONS_EXCLUDE", " a , b ,, c ")
    assert I._load_exclude() == ["a", "b", "c"]


# ── allow-list + ingest MODE (opt-in / opt-out privacy posture) ───────────────

def test_allowlist_includes_matching_project():
    assert I._included("-Users-rbr-hermes-rubric", ["hermes"]) is True

def test_allowlist_excludes_nonmatching_project():
    assert I._included("-Users-rbr-lpci", ["hermes"]) is False

def test_allowlist_empty_includes_nothing():
    # opt-in mirror of the deny-list: empty/None matches nothing → indexes nothing
    assert I._included("-Users-rbr-anything", []) is False
    assert I._included("-Users-rbr-anything", None) is False

def test_allowlist_env_parsing(monkeypatch):
    monkeypatch.setenv("FIDELIS_SESSIONS_INCLUDE", " a , b ,, c ")
    assert I._load_include() == ["a", "b", "c"]


def test_mode_defaults_to_opt_out(monkeypatch, tmp_path):
    monkeypatch.delenv("FIDELIS_SESSIONS_MODE", raising=False)
    monkeypatch.setattr(I.Path, "home", staticmethod(lambda: tmp_path))  # no config.json
    assert I._load_mode() == "opt-out"

def test_mode_read_from_env(monkeypatch):
    monkeypatch.setenv("FIDELIS_SESSIONS_MODE", "opt-in")
    assert I._load_mode() == "opt-in"

def test_mode_env_invalid_falls_back_to_opt_out(monkeypatch, tmp_path):
    monkeypatch.setenv("FIDELIS_SESSIONS_MODE", "garbage")
    monkeypatch.setattr(I.Path, "home", staticmethod(lambda: tmp_path))
    assert I._load_mode() == "opt-out"

def test_mode_read_from_config_json(monkeypatch, tmp_path):
    monkeypatch.delenv("FIDELIS_SESSIONS_MODE", raising=False)
    cfg_dir = tmp_path / ".cogito"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(I.json.dumps({"sessions_mode": "opt-in"}))
    monkeypatch.setattr(I.Path, "home", staticmethod(lambda: tmp_path))
    assert I._load_mode() == "opt-in"

def test_mode_env_overrides_config(monkeypatch, tmp_path):
    # config says opt-in, env says opt-out → env wins (matches config.py semantics)
    cfg_dir = tmp_path / ".cogito"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(I.json.dumps({"sessions_mode": "opt-in"}))
    monkeypatch.setattr(I.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("FIDELIS_SESSIONS_MODE", "opt-out")
    assert I._load_mode() == "opt-out"

def test_mode_config_malformed_falls_back_to_opt_out(monkeypatch, tmp_path):
    monkeypatch.delenv("FIDELIS_SESSIONS_MODE", raising=False)
    cfg_dir = tmp_path / ".cogito"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text("{ not valid json ")
    monkeypatch.setattr(I.Path, "home", staticmethod(lambda: tmp_path))
    assert I._load_mode() == "opt-out"


# ── _iter_sessions honours the mode (allow-list vs deny-list selection) ────────

def _fake_projects(tmp_path, names):
    """Build a fake ~/.claude/projects tree with one empty *.jsonl per project."""
    root = tmp_path / "projects"
    root.mkdir()
    for n in names:
        d = root / n
        d.mkdir()
        (d / "sess.jsonl").write_text("")  # presence is enough; we don't parse here
    return root


def test_iter_opt_out_indexes_all_but_excluded(monkeypatch, tmp_path):
    root = _fake_projects(tmp_path, ["-Users-rbr-hermes", "-Users-rbr-client-secret", "-Users-rbr-lpci"])
    monkeypatch.setattr(I, "CLAUDE_PROJECTS", root)
    seen = {p for _, _, p in I._iter_sessions(mode="opt-out", exclude=["client-secret"])}
    assert seen == {"-Users-rbr-hermes", "-Users-rbr-lpci"}  # excluded one dropped

def test_iter_opt_in_indexes_only_included(monkeypatch, tmp_path):
    root = _fake_projects(tmp_path, ["-Users-rbr-hermes", "-Users-rbr-client-secret", "-Users-rbr-lpci"])
    monkeypatch.setattr(I, "CLAUDE_PROJECTS", root)
    seen = {p for _, _, p in I._iter_sessions(mode="opt-in", include=["hermes"])}
    assert seen == {"-Users-rbr-hermes"}  # ONLY the allow-listed project

def test_iter_opt_in_empty_include_yields_nothing(monkeypatch, tmp_path):
    root = _fake_projects(tmp_path, ["-Users-rbr-hermes", "-Users-rbr-lpci"])
    monkeypatch.setattr(I, "CLAUDE_PROJECTS", root)
    seen = list(I._iter_sessions(mode="opt-in", include=[]))
    assert seen == []  # opt-in + empty allow-list = index nothing


# ── ingest() wiring: opt-out preserves behaviour, opt-in onboards ─────────────

def test_ingest_opt_out_preserves_existing_behaviour(capsys, monkeypatch):
    # opt-out is the default: existing deny-list path is untouched. Two candidates
    # flow through (then skipped_empty since _parse_jsonl returns []), proving the
    # iterator was actually consumed — i.e. opt-out indexes-except-excluded intact.
    monkeypatch.setattr(I, "_load_mode", lambda: "opt-out")
    monkeypatch.setattr(I, "_iter_sessions",
                        lambda since=None, exclude=None, mode="opt-out", include=None: iter([
                            ("s1", I.Path("/x/s1.jsonl"), "-Users-rbr-proj"),
                            ("s2", I.Path("/x/s2.jsonl"), "-Users-rbr-proj"),
                        ]))
    monkeypatch.setattr(I, "_parse_jsonl", lambda path: [])
    monkeypatch.setattr(I, "_load_ledger", lambda: {"existing": "id"})
    stats = I.ingest(dry_run=True)
    out = capsys.readouterr().out
    assert "2 sessions" in out
    assert stats["scanned"] == 2

def test_ingest_opt_in_with_include_indexes_only_matching(capsys, monkeypatch):
    # opt-in + non-empty allow-list: iterator must be called with mode/include and
    # only yield the matching project. We assert the passthrough by inspecting args.
    captured = {}
    def fake_iter(since=None, exclude=None, mode="opt-out", include=None):
        captured["mode"] = mode
        captured["include"] = include
        return iter([("s1", I.Path("/x/s1.jsonl"), "-Users-rbr-hermes")])
    monkeypatch.setattr(I, "_iter_sessions", fake_iter)
    monkeypatch.setattr(I, "_parse_jsonl", lambda path: [])
    monkeypatch.setattr(I, "_load_ledger", lambda: {"existing": "id"})
    stats = I.ingest(dry_run=True, mode="opt-in", include=["hermes"])
    out = capsys.readouterr().out
    assert captured == {"mode": "opt-in", "include": ["hermes"]}
    assert "1 sessions" in out
    assert "no projects included yet" not in out
    assert stats["scanned"] == 1

def test_ingest_opt_in_empty_include_prints_onboarding_and_indexes_nothing(capsys, monkeypatch):
    # Setup-sensitive: empty allow-list in opt-in mode must NOT silently no-op.
    # It prints the onboarding message and indexes nothing, without walking the tree.
    def boom(*a, **k):
        raise AssertionError("_iter_sessions must not run when include is empty")
    monkeypatch.setattr(I, "_iter_sessions", boom)
    stats = I.ingest(dry_run=True, mode="opt-in", include=[])
    out = capsys.readouterr().out
    assert "opt-in mode: no projects included yet" in out
    assert "FIDELIS_SESSIONS_INCLUDE" in out
    assert stats == {"scanned": 0, "skipped_dedup": 0, "skipped_empty": 0, "stored": 0, "errors": 0}

def test_ingest_reads_mode_from_env_when_not_passed(capsys, monkeypatch):
    # ingest() with no mode arg falls back to _load_mode() (env/config). Set env to
    # opt-in with empty include → onboarding path, proving the env was honoured.
    monkeypatch.setenv("FIDELIS_SESSIONS_MODE", "opt-in")
    monkeypatch.delenv("FIDELIS_SESSIONS_INCLUDE", raising=False)
    stats = I.ingest(dry_run=True)
    out = capsys.readouterr().out
    assert "opt-in mode: no projects included yet" in out
    assert stats["stored"] == 0


# ── purge target collection (the locked contract) ────────────────────────────

def _meta(end_ts, h):
    return {"end_ts": end_ts, "ingest_hash": h, "mem_type": "session"}

def test_iso_before():
    assert S._iso_before("2026-04-20T10:00:00Z", "2026-04-30") is True
    assert S._iso_before("2026-05-30T10:00:00Z", "2026-04-30") is False
    assert S._iso_before("", "2026-04-30") is False  # missing ts never matches

def test_collect_all_takes_everything():
    metas = [_meta("2026-04-01T00:00:00Z", "h1"), _meta("2026-05-01T00:00:00Z", "h2")]
    ids = ["c1", "c2"]
    tids, ths = S._collect_purge_targets(metas, ids, "all", None)
    assert tids == ["c1", "c2"]
    assert ths == ["h1", "h2"]  # hashes collected for ledger cleanup (spec step 3)

def test_collect_before_filters_by_date():
    metas = [_meta("2026-04-01T00:00:00Z", "h1"), _meta("2026-05-01T00:00:00Z", "h2")]
    ids = ["c1", "c2"]
    tids, ths = S._collect_purge_targets(metas, ids, "before", "2026-04-30")
    assert tids == ["c1"]          # only the April session
    assert ths == ["h1"]

def test_collect_before_none_matches_nothing_when_future():
    metas = [_meta("2026-05-01T00:00:00Z", "h2")]
    tids, _ = S._collect_purge_targets(metas, ["c2"], "before", "2026-04-30")
    assert tids == []

def test_purge_span_reflects_matched_set_not_whole_corpus():
    # honesty/polish: the summary span must describe what WILL be deleted,
    # not the entire corpus. Regression guard for the 2026-06-06 fix.
    metas = [_meta("2026-04-01T00:00:00Z", "h1"),   # matched (before 04-30)
             _meta("2026-05-30T00:00:00Z", "h2")]   # NOT matched
    ids = ["c1", "c2"]
    target_ids, _ = S._collect_purge_targets(metas, ids, "before", "2026-04-30")
    target_set = set(target_ids)
    ends = sorted(m.get("end_ts", "") for cid, m in zip(ids, metas)
                  if cid in target_set and m.get("end_ts"))
    span = f"{ends[0][:10]}..{ends[-1][:10]}" if ends else "unknown"
    assert span == "2026-04-01..2026-04-01"   # NOT ..2026-05-30


# ── honesty invariants (these protect the brand) ─────────────────────────────

def test_backup_boundary_present_in_messages():
    # deletion claims must always carry the source/backup-not-touched boundary
    assert "NOT touched" in S.BACKUP_BOUNDARY
    assert "~/.claude/projects" in S.BACKUP_BOUNDARY
    assert "~/Backups/claude-sessions" in S.BACKUP_BOUNDARY

def test_no_benchmark_numbers_in_product_strings():
    # Honesty rule: benchmark numbers must never appear in USER-FACING strings.
    # Check the module-level constants + any string literal printed to users,
    # not source comments/docstrings (those legitimately discuss the rule).
    forbidden = ["93.4", "96.4"]
    for const in (S.BACKUP_BOUNDARY, S.PRIVACY_NOTICE):
        for f in forbidden:
            assert f not in const, f"benchmark number {f} leaked into product string"
    # the sanctioned honest label appears in the search-empty product string
    src = open(importlib.import_module("fidelis.sessions_cmd").__file__).read()
    assert "best on multi-turn queries" in src

def test_privacy_notice_mentions_local_and_purge():
    assert "~/.cogito" in S.PRIVACY_NOTICE
    assert "purge" in S.PRIVACY_NOTICE


# ── B2: ingest progress signal (no silent multi-minute "hang") ────────────────

class _IngestArgs:
    def __init__(self, all=False, since=None, dry_run=True, verbose=False):
        self.all = all
        self.since = since
        self.dry_run = dry_run
        self.verbose = verbose


def test_ingest_prints_candidate_count_up_front(capsys, monkeypatch):
    # B2: before the loop, ingest() must announce how many sessions it will index
    # so a long run doesn't read as a freeze.
    monkeypatch.setattr(I, "_iter_sessions",
                        lambda since=None, exclude=None, mode="opt-out", include=None: iter([
                            ("s1", I.Path("/x/s1.jsonl"), "-Users-rbr-proj"),
                            ("s2", I.Path("/x/s2.jsonl"), "-Users-rbr-proj"),
                        ]))
    monkeypatch.setattr(I, "_parse_jsonl", lambda path: [])  # all skipped_empty; no DB touch
    monkeypatch.setattr(I, "_load_ledger", lambda: {"existing": "id"})  # not first run
    I.ingest(dry_run=True)
    out = capsys.readouterr().out
    assert "2 sessions" in out
    assert "may take a few minutes" in out


# ── B3: first-run nudge about the 7-day default window ────────────────────────

def test_first_run_true_when_ledger_empty(monkeypatch):
    monkeypatch.setattr(I, "_load_ledger", lambda: {})
    assert S._is_first_run() is True


def test_first_run_false_when_ledger_has_entries(monkeypatch):
    monkeypatch.setattr(I, "_load_ledger", lambda: {"h1": "id1"})
    assert S._is_first_run() is False


def test_nudge_prints_on_first_run_default_window(capsys, monkeypatch):
    monkeypatch.setattr(S, "_is_first_run", lambda: True)
    monkeypatch.setattr("fidelis.ingest_claude_sessions.ingest",
                        lambda **kw: dict(scanned=0, stored=0, skipped_dedup=0,
                                          skipped_empty=0, errors=0))
    S.cmd_ingest(_IngestArgs(all=False, since=None, dry_run=True))
    out = capsys.readouterr().out
    assert "--all" in out and "full history" in out


def test_nudge_suppressed_when_not_first_run(capsys, monkeypatch):
    monkeypatch.setattr(S, "_is_first_run", lambda: False)
    monkeypatch.setattr("fidelis.ingest_claude_sessions.ingest",
                        lambda **kw: dict(scanned=0, stored=0, skipped_dedup=0,
                                          skipped_empty=0, errors=0))
    S.cmd_ingest(_IngestArgs(all=False, since=None, dry_run=True))
    out = capsys.readouterr().out
    assert "full history" not in out


def test_nudge_suppressed_when_all_flag(capsys, monkeypatch):
    # --all is not the default window, so the nudge must not fire even on first run.
    monkeypatch.setattr(S, "_is_first_run", lambda: True)
    monkeypatch.setattr("fidelis.ingest_claude_sessions.ingest",
                        lambda **kw: dict(scanned=0, stored=0, skipped_dedup=0,
                                          skipped_empty=0, errors=0))
    S.cmd_ingest(_IngestArgs(all=True, since=None, dry_run=True))
    out = capsys.readouterr().out
    assert "full history" not in out
