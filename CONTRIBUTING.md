# Contributing

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ \
  --ignore=tests/scaffold/test_e2e_store_query.py \
  --ignore=tests/scaffold/test_backend_portability.py \
  --ignore=tests/scaffold/test_anthropic_cache_wire.py \
  --ignore=tests/scaffold/test_openai_format_compatibility.py \
  --ignore=tests/scaffold/test_streaming_marker_integrity.py \
  -q
```

This is the suite CI runs. It needs no Ollama or ChromaDB. The five ignored files are not part of it: `test_e2e_store_query.py` starts a real `fidelis-server` and needs Ollama, `test_backend_portability.py` and `test_openai_format_compatibility.py` include live smoke tests against local Ollama (or the `claude` CLI) that skip when those are absent, and `test_anthropic_cache_wire.py` and `test_streaming_marker_integrity.py` are mocked-transport wire-format tests that CI does not run. Run them separately when you change that surface. Mark any other test that needs a live server with `pytest.mark.skip(reason="requires server")`.

## Bench cases

Bench cases live in `bench/`. `bench/cases.json` and `bench/eval_cases.json` are lists of objects with a `query` and an `expected` list of keywords, plus a `difficulty` or `case_type` field. `python bench/eval.py` runs the combined eval against a running `fidelis-server` and its live corpus, so it is not part of the server-free test suite and is not needed for most contributions. Add cases that cover real retrieval failures or regressions.

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
