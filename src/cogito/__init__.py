"""
cogito-ergo — a memory layer for AI agents.

Two-layer architecture:
  Hot layer  — MEMORY.md: always-in-context index, injected at session start
  Cold layer — Vector store: searchable history, queried on demand

Key innovation: /recall endpoint uses a cheap LLM (e.g. Haiku) as an
integer-pointer fidelity filter. The cheap LLM outputs only integer indices —
never memory text — so it cannot corrupt or hallucinate into the content
returned to the main agent.
"""

__version__ = "0.1.0"

from cogito.recall import recall  # noqa: F401
