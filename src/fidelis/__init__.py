"""
fidelis — faithful memory retrieval for AI agents.

Default path (zero-LLM): BM25 + dense + RRF fusion, 83.2% R@1 on
LongMemEval_S, $0/query, ~90ms, fully local.

Optional LLM tiers (experimental): ``filter`` and ``flagship``. The filter
LLM outputs only integer indices (e.g. [3, 7, 12]) — never memory text —
so it cannot corrupt or hallucinate into the content returned to the agent.
Fidelity is structural, not a prompting convention.

fidelis was previously published as ``cogito-ergo`` (0.0.8 and 0.3.0 on PyPI).
Data paths and env var names retain the ``cogito`` prefix for continuity
with existing deployments.
"""

__version__ = "0.0.8"

from fidelis.recall import recall  # noqa: F401

# Note: avoid shadowing the ``fidelis.recall_hybrid`` submodule. Users who want
# the function directly can do ``from fidelis.recall_hybrid import recall_hybrid``.
