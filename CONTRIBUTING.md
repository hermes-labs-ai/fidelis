# Contributing

## Running tests

```bash
pip install -e ".[dev]"
(
  queue_dir="$(mktemp -d)" || exit   # no temp dir, no test run (never fall back to ~/.cogito/queue)
  [ -n "$queue_dir" ] || exit 1
  trap 'rm -rf -- "$queue_dir"' EXIT
  FIDELIS_QUEUE_DIR="$queue_dir" python -m pytest tests/ \
    --ignore=tests/scaffold/test_e2e_store_query.py \
    --ignore=tests/scaffold/test_backend_portability.py \
    --ignore=tests/scaffold/test_anthropic_cache_wire.py \
    --ignore=tests/scaffold/test_openai_format_compatibility.py \
    --ignore=tests/scaffold/test_streaming_marker_integrity.py \
    -q
)
```

This is the suite CI runs; CI likewise points `FIDELIS_QUEUE_DIR` at a temporary directory. The whole block runs in a subshell: the variable applies to that one `pytest` command only, nothing is exported, the temp directory is removed on exit, the block's exit status is pytest's own, and a failure to create the temp directory stops it before any test runs. It is safe to paste into an interactive or sourced shell. It needs no Ollama or ChromaDB. The five ignored files are not part of it: `test_e2e_store_query.py` starts a real `fidelis-server` and needs Ollama, `test_backend_portability.py` and `test_openai_format_compatibility.py` include live smoke tests against local Ollama (or the `claude` CLI) that skip when those are absent, and `test_anthropic_cache_wire.py` and `test_streaming_marker_integrity.py` are mocked-transport wire-format tests that CI does not run. Run them separately when you change that surface. Guard any other test that needs a live server with a conditional skip such as `pytest.mark.skipif(not _ollama_reachable(), reason="requires local Ollama on :11434")` (see `tests/test_graceful_shutdown.py`), so it still runs where the server is available.

## Bench cases

Bench files live in `bench/`, with different consumers and formats:

- `bench/cases.json` is the default input to `bench/benchmark.py`: a list of objects with `query`, an `expected` list of keywords, and optional `difficulty` or `notes`.
- `bench/eval_cases.json` is the default static input to `bench/eval.py` (override with `--cases`): a list of objects with `query`, an `expected` list of keywords, `case_type`, and optional `notes`.
- `bench/hardset.json` contains historical LongMemEval records with `question`, `gold_session_ids`, and retrieval results. Older experimental runners such as `bench/longmemeval_combined_pipeline_v33.py` consume it; it is not an input to `bench/eval.py` or `bench/benchmark.py`.

`python bench/eval.py` requires a running `fidelis-server` and its live corpus. In addition to the static cases, it generates direct-recall cases from that corpus via `/recall_b` by default. `--static-only` skips generation but still requires the server for the eval. Use a disposable server and store, never a store holding real data. For a server-free check, run `python bench/eval.py --help` or `python bench/benchmark.py --help`. These live benchmarks are not part of the server-free test suite and are not needed for most contributions. Add cases that cover real retrieval failures or regressions.

## Code style

- Line length: 100
- Linter: `ruff` (installed by the `dev` extra); CI runs `ruff check src/ tests/`. `ruff check .` also covers `bench/` and other scripts and currently reports findings that CI does not enforce. Formatting with `ruff format` is not enforced either.
- Target: Python 3.10+

## `top_score` input contract

`wrap_system_prompt(qtype, top_score=...)` accepts any numeric value for `top_score`:

| Input | Behaviour |
|---|---|
| `None` | Treated as unknown → `[retrieval-quality: unknown]` |
| `float` in `[0, 1]` | Used as-is for confidence band selection |
| `float` outside `[0, 1]` (e.g. `-0.5`, `1.5`) | **Clamped** silently to `[0, 1]` — no exception raised |
| `nan` / `inf` / `-inf` | Treated as `None` → `[retrieval-quality: unknown]` |
| `int` `0` / `1` | Coerced to `float 0.0` / `1.0` |
| `bool` `True` / `False` | Coerced to `1.0` / `0.0` (Python bool subclasses int) |

Rationale: retrieval backends may return cosine values fractionally outside `[0, 1]` due to float
arithmetic, or `nan`/`inf` on degenerate inputs. The scaffold uses `top_score` only for a
cosmetic confidence label, so clamping is the principle-of-least-surprise choice — it never
raises and never breaks the caller's pipeline.

See `tests/scaffold/test_top_score_contract.py` for the full contract test suite.

## Pull requests

1. Fork and branch from `main`.
2. Keep changes focused — one feature or fix per PR.
3. Add a test if the change touches retrieval logic.
4. Update `CHANGELOG.md` under an `Unreleased` section.
