# Changelog

## v0.2.0 — 2026-03-28

- Dual-pipeline recall: zero-LLM `recall_b` (RRF multi-query) feeds `recall` (integer-pointer LLM filter)
- Snapshot layer for high-fidelity context injection
- Combined eval harness (`bench/`)
- Technical extraction prompt baked into default config (no external prompt file needed)
- Qwen3/qwen3.5 support via native Ollama `/api/chat` with `think:false`

## v0.1.0 — 2026-03

- Initial two-stage integer-pointer recall pipeline
- mem0 + ChromaDB vector store backend
- HTTP server with `/recall`, `/add`, `/snapshot` endpoints
- `cogito calibrate` for vocab map generation
