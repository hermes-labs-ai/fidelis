"""
cogito-ergo — two-stage memory retrieval for AI agents.

Three-layer architecture:
  Snapshot   — compressed markdown index (~741 tokens), built once via `cogito snapshot`
  recall_b   — zero-LLM sub-query decomposition + RRF (Stage 1, 127ms)
  recall     — integer-pointer LLM filter (Stage 2, +1176ms)

Key innovation: the filter LLM outputs only integer indices (e.g. [3, 7, 12]) —
never memory text — so it cannot corrupt or hallucinate into the content
returned to the main agent. Fidelity is structural, not a prompting convention.
"""

__version__ = "0.2.0"

from cogito.recall import recall  # noqa: F401
