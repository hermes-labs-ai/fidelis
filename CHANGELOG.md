# Changelog

# Unreleased

- Add `fidelis mcp install --client copilot` and `fidelis mcp uninstall
  --client copilot`, which register the bundled stdio MCP server in GitHub
  Copilot CLI's documented `mcp-config.json` (`~/.copilot` or `$COPILOT_HOME`)
  with atomic writes, backups, and ownership checks that preserve unrelated
  entries. The `copilot` binary is not required at configuration time.
  `fidelis mcp uninstall --force` removes an entry that does not look like
  ours, for the case where the ownership check is wrong.
- Atomic config rewrites now keep the destination file's permission bits, and
  a config Fidelis creates is `0600`. Previously every rewrite reset an
  `mcp-config.json` or `settings.local.json` to the umask default, widening a
  file that can hold MCP `env` secrets.
- Config backups no longer overwrite each other when two mutations land in the
  same second, so an install immediately followed by an uninstall keeps both
  restore points.

## v0.0.95 — 2026-09-02

- Add a portable `fidelis mcp serve` entry point and official MCP Registry
  metadata for `uvx`-based installation.
- Add the PyPI README ownership marker required for the
  `io.github.hermes-labs-ai/fidelis-memory` registry namespace.

## v0.0.94 — 2026-09-02

- Add supported Codex MCP installation and removal through the shared
  `codex mcp` configuration, with exact ownership checks that preserve
  unrelated entries.
- Add `fidelis_orient`, a deterministic MCP tool that selects a bounded
  evidence lane, returns retrieved records without rephrasing them, and
  abstains when a turn lacks a known referent or historical-context cue.
- Add citation and CodeMeta metadata and align the agent-facing `llms.txt`
  retrieval headline with the shipped zero-LLM benchmark evidence.

## v0.0.93 — 2026-08-05

- Publish the Hermes Labs distribution to PyPI as `fidelis-memory`.
- Make **Fidelis Memory** the user-facing product name across package metadata
  and primary documentation, while preserving the `fidelis` repository, import,
  and CLI names for compatibility.
- Replace the GitHub-source install path in primary documentation with the
  version-pinned PyPI installation command.

## v0.0.92 — 2026-08-04

- Preserve `/add` input verbatim when the extraction model returns no facts,
  expose the degraded fallback through HTTP and CLI, and apply the same
  no-silent-loss rule during queued-write replay.
- Correct every current install surface to use tagged GitHub source; the PyPI
  project named `fidelis` is unrelated to Hermes Labs Fidelis.
- Align public package status with v0.0.91 and remove a dead benchmark link.
- Narrow local-data and compliance wording to the demonstrated boundary.
- Default mem0 telemetry off for direct server launches so SIGTERM completes
  cleanly, while preserving explicit opt-in; document the remaining Chroma
  settings and extend CI coverage through Python 3.14.
- Remove cross-project proof language from current Fidelis presentation.
- Add a regression test for the public installation contract.

# fidelis era
## v0.0.91 — 2026-06-18 (public release)
- Public open-source release on GitHub + Zenodo DOI.
- Repo hygiene only: relativized benchmark paths, removed run-log artifacts, refreshed module naming, added Zenodo metadata. No API changes vs v0.0.9.


## v0.0.5 — 2026-04-24 (first release as `fidelis`, renamed from `cogito-ergo`)

**Package rename: `cogito-ergo` → `fidelis`.** Version reset for the new
PyPI package. The old `cogito-ergo` PyPI entries (0.0.8 and 0.3.0) remain
published but are now deprecated pointers to this package. Note: a prior
`cogito-ergo` v0.0.5 also exists in the changelog below — these are
**different packages**; the version numbers do not collide on PyPI.

**Headline: zero-LLM is the default retrieval tier.** The repositioning
reflects what the benchmark numbers actually show — the zero-LLM path is
the production moat (83.2% R@1 at $0, fully local); the LLM tiers are
benchmark-tuned and held as experimental until calibration is fixed.

### Rename details

- PyPI package: `cogito-ergo` → `fidelis`
- Import name: `from cogito` → `from fidelis`
- CLI: `cogito ...` → `fidelis ...`, `cogito-server` → `fidelis-server`
- Data paths kept as `~/.cogito/` for backward-compat with existing deployments
- Env vars kept as `COGITO_*` for backward-compat
- ChromaDB collection names kept as `cogito_memory` to avoid data loss

### Default change (breaking for callers relying on default tier)

- `recall_hybrid()` default tier flipped from `filter` to `zero_llm`.
  Callers that relied on the filter default must pass `tier="filter"`
  explicitly. Affects: `fidelis recall-hybrid` CLI, `POST /recall_hybrid`,
  `recall_hybrid(...)` Python API.

### Features

- New `since` parameter on `/recall` and `fidelis recall --since` —
  ISO-8601 timestamp filter applied after Stage 2.

### Observability & reliability

- New `fidelis.telemetry` module: escalation-rate log + `rate()` summariser.
  Makes the 80%-vs-10% calibration miss measurable at runtime. Crash-safe.
- `degrade.replay_queue` corrupt-file branch now covered by tests.

### Tests

- `tests/test_zero_llm_regression.py` — 4 tests pin zero-LLM default:
  no filter/flagship invocation, works without LLM config, matches
  explicit-tier output, handles empty corpus.
- `tests/test_telemetry.py` — 5 tests for the new observability module.
- `tests/test_verify_guard.py` — regression pin for the documented
  verify-guard-inactive bug; xfail until fixed.
- `tests/test_graceful_degrade_corruption.py` — replay_queue corruption.
- Test baseline: 61 → 72 passing + 2 defensive skips.

### Benchmarks (for reference, not default-path claims)

- `recall_hybrid` at `tier="flagship"` hits 96.4% R@1 on LongMemEval_S
  (470 questions, runP-v35, 2026-04-18). Escalates on ~80% of queries
  under current calibration vs 10% intended — known limitation
  documented in STATUS.md and `docs/RELEASE-SCOPE.md`. Do not use the
  96.4% number as a cost-model baseline.
- Zero-LLM per-category R@1 (now the default): single-session-assistant
  100%, knowledge-update 96%, single-session-user 95%, multi-session 83%,
  single-session-preference 67%, temporal-reasoning 66%.

### Docs

- README repositioned: zero-LLM leads, LLM tiers labeled experimental,
  per-category table published.
- `docs/THRESHOLD-AUDIT.md` — pins prod vs bench escalation constants.
- `docs/RELEASE-SCOPE.md` — in-scope / held-for-later with unlock criteria.

---

# cogito-ergo era (predecessor package, retained for history)

## v0.3.0 — 2026-04-16

- New `recall_hybrid` path: BM25 + dense + RRF with tiered LLM escalation.
  Port of the architecture that reached **93.4% R@1 on LongMemEval_S** (up
  from the 56% mem0 baseline). Superseded by v0.3.1 (96.4%).
- New `POST /recall_hybrid` HTTP endpoint and `fidelis recall-hybrid` CLI.
- New tiers: `zero_llm` (no LLM, fastest), `filter` (cheap rerank, default),
  `flagship` (stronger model, 4x larger snippets).
- New env vars: `COGITO_FLAGSHIP_ENDPOINT`, `COGITO_FLAGSHIP_TOKEN`,
  `COGITO_FLAGSHIP_MODEL`, `COGITO_FLAGSHIP_TIMEOUT_MS`,
  `COGITO_HYBRID_COSINE_WEIGHT`. Optional `[hybrid]` extra for `bm25s`.
- Existing `/recall` and `/recall_b` behavior is unchanged. The hybrid path
  is strictly opt-in.

## v0.2.0 — 2026-03-28

- Dual-pipeline recall: zero-LLM `recall_b` (RRF multi-query) feeds `recall` (integer-pointer LLM filter)
- Snapshot layer for high-fidelity context injection
- Combined eval harness (`bench/`)
- Technical extraction prompt baked into default config (no external prompt file needed)
- Qwen3/qwen3.5 support via native Ollama `/api/chat` with `think:false`

## v0.0.5 — 2026-03-28

- Fixed benchmark attribution: qwen3.5:2b filter model (not claude-haiku-4-5)
- Added Hermes Labs PyPI metadata (author, homepage, keywords)
- Added agents.md for AI agent discoverability
- Updated llms.txt with full API shapes and integration notes
- Added "Built by Hermes Labs" ecosystem section to README

## v0.1.0 — 2026-03

- Initial two-stage integer-pointer recall pipeline
- mem0 + ChromaDB vector store backend
- HTTP server with `/recall`, `/add`, `/snapshot` endpoints
- `fidelis calibrate` for vocab map generation
