# Contributing

## Running tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests in `tests/` are server-free — no Ollama or ChromaDB required. Mark any test that needs a live server with `pytest.mark.skip(reason="requires server")`.

## Adding bench cases

Bench cases live in `bench/`. Each case is a JSON file with `query`, `expected_ids`, and optional `notes`. Run the combined eval with:

```bash
python bench/eval.py
```

Add cases that cover real retrieval failures or regressions. Include the memory seed data in `bench/seeds/` if needed.

## Code style

- Line length: 100
- Formatter/linter: `ruff` (`pip install ruff`, then `ruff check .` and `ruff format .`)
- Target: Python 3.10+

## Pull requests

1. Fork and branch from `main`.
2. Keep changes focused — one feature or fix per PR.
3. Add a test if the change touches retrieval logic.
4. Update `CHANGELOG.md` under an `Unreleased` section.
