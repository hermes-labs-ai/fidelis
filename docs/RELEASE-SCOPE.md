# Release Scope — fidelis v0.0.8 (2026-04-24)

First release under the `fidelis` name (renamed from `cogito-ergo`).
Version reset to 0.0.8 because this is a new PyPI package.
Single source of truth for what this release ships, what it defers, and
why. Honest; not aspirational.

## In scope (v0.0.8)

### Rename
- PyPI: `cogito-ergo` → `fidelis`; imports `from cogito` → `from fidelis`;
  CLI `cogito ...` → `fidelis ...`.
- Data paths (`~/.cogito/`), env vars (`COGITO_*`), and ChromaDB
  collection names retained for backward-compat with existing
  deployments. Migration is optional, not forced.

### Feature
- `since` parameter on `/recall` and `fidelis recall --since` — timestamp
  filtering on atomic recall.

### Default change (intentional)
- **`recall_hybrid` default tier flipped from `filter` to `zero_llm`.**
  This is the headline repositioning: the production moat is the zero-LLM
  path, not the benchmark-tuned LLM tier. Callers that relied on the
  `filter` default must now pass `tier="filter"` explicitly.

### Observability
- `src/cogito/telemetry.py` — escalation-rate log + `rate()` summariser.
  Makes the 80%-vs-10% calibration miss measurable from runtime, not
  only from bench post-hoc. Crash-safe; never raises in the caller's
  hot path.

### Tests added
- `tests/test_telemetry.py` — 5 tests covering roundtrip, window
  truncation, unwritable-path no-op, corrupt-line handling.
- `tests/test_verify_guard.py` — regression pin for STATUS.md known
  bug #1 (verify-guard activation field absent). xfail until fixed.
- `tests/test_graceful_degrade_corruption.py` — exercises
  `degrade.replay_queue` corrupt-file branch.
- `tests/test_zero_llm_regression.py` — 4 tests pinning the zero-LLM
  default-tier behavior: no LLM calls, works without LLM config,
  matches explicit-tier output, handles empty corpus.

### Docs
- `docs/THRESHOLD-AUDIT.md` — pins prod vs bench escalation thresholds.
- `docs/RELEASE-SCOPE.md` — this file.
- README repositioned: zero-LLM tier leads; LLM tiers labeled
  benchmark-tuned and experimental; per-category breakdown published;
  honest ceiling on TR/Pref surfaces as a table, not buried in prose.

### Infra (outside the repo)
- `~/ai-infra/checks/cogito_enabled.sh` — new invariant, distinct from
  `cogito_reachable`, catches the launchd-disabled-label case that
  broke 2026-04-24.
- `~/ai-infra/INVARIANTS.yaml` — `cogito_enabled` wired; remediations
  updated to `launchctl kickstart -k` which handles wedged handlers.

## Held for a later release

These are known gaps. Each has a concrete unlock condition.

| Item | Unlock condition | Current state |
|---|---|---|
| Calibration fix on filter/flagship tiers | Per-qtype threshold fit on stratified samples | 80.2% escalation vs 10% intended documented; not fixed |
| Dispatcher / Path A–B split | `docs/DISPATCHER_DESIGN.md` implementation | Design doc only |
| Verify-guard activation fix | Per-question output dict carries a `route_decision` / `guard_activated` field | `llm_verify_one` is called but no output-side telemetry |
| Temporal-reasoning improvement | Structured metadata tagging at ingest (not prompt injection) | Chunk-level date injection was 0pp net, reverted |
| Preference-question calibration | Dedicated preference-prompt path (E2 shows 66.7% with v3_routing.py) | Experimental; not in hot path yet |
| Cost-claim benchmarking vs mem0/zep | Apples-to-apples cost-per-query run at fixed accuracy | I-don't-know numbers today; will not claim cheaper until measured |

## Version classifier remains Alpha

`pyproject.toml` keeps `Development Status :: 3 - Alpha` — the known
gaps listed above warrant Alpha, not Beta. The release is **honest
about what it is**: a production-ready zero-LLM retrieval backbone
with an experimental LLM-escalation tier. Stable API for the zero-LLM
path; tier semantics may evolve for filter/flagship.

## Audit trail

- Built during 2026-04-24 hardening session.
- hermes-rubric Opus-run audit: 6.5/10 pre-hardening, 8.1/10 post-repositioning.
- Hermes-seal: verified, in window, continuity category passed.
- Test baseline: 61 → 72 passing + 2 defensive skips. Zero regressions.

## Not in this release

- No cost-comparison claims against mem0/zep/letta. Those require measured
  per-query costs at fixed accuracy, which we don't have. Do not add to
  README or marketing until that data exists.
- No 96.4% headline. That number is preserved for flagship-tier
  documentation; it is not the released product's default.
- No public push of this branch. `release/v0.0.8-zero-llm-first` is
  local-only until explicit approval.
