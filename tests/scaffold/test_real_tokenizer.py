"""Real-tokenizer token-budget enforcement tests for Fidelis scaffold.

Uses tiktoken cl100k_base (GPT-4 / OpenAI, reasonable proxy for most modern
providers including Anthropic cl100k-family) and o200k_base (GPT-4o) to verify
the scaffold stays under the 200-token hard cap at all qtype × top_score combos.

Also validates whether the heuristic in preflight.py (4 chars ≈ 1 token) is
conservative (heuristic >= real_count). Results from this test file expose
cases where the heuristic UNDER-estimates — meaning the heuristic is not a safe
upper bound and preflight.py's length-check could let scaffolds past that are
actually over-budget on some tokenizers.
"""

from __future__ import annotations

import pytest

tiktoken = pytest.importorskip("tiktoken")

from fidelis.scaffold._core import wrap_system_prompt, _QTYPE_PROC  # noqa: E402

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

QTYPES = list(_QTYPE_PROC.keys())

# All top_score values exercised in production paths
SIM_VALUES = [None, 0.3, 0.5, 0.7, 0.9, 1.0]

TOKEN_CAP = 200

# Preload encoders once (expensive to reinitialise per test)
@pytest.fixture(scope="module")
def enc_cl100k():
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture(scope="module")
def enc_o200k():
    return tiktoken.get_encoding("o200k_base")


def _heuristic(text: str) -> int:
    """Mirror of preflight.py _approx_tokens (3.5 chars/token, conservative)."""
    return max(1, int(len(text) / 3.5))


# --------------------------------------------------------------------------- #
# Parametrised: every qtype × top_score under 200 tokens (cl100k_base)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qtype", QTYPES)
@pytest.mark.parametrize("sim", SIM_VALUES)
def test_real_token_count_cl100k_under_200(qtype, sim, enc_cl100k):
    """cl100k_base real token count must stay <= 200 for every qtype + similarity."""
    text = wrap_system_prompt(qtype, top_score=sim)
    real_count = len(enc_cl100k.encode(text))
    heuristic_count = _heuristic(text)
    print(
        f"\n[cl100k] qtype={qtype!r} sim={sim} "
        f"chars={len(text)} heuristic={heuristic_count} real={real_count}"
    )
    assert real_count <= TOKEN_CAP, (
        f"REAL TOKEN OVERFLOW (cl100k_base): qtype={qtype!r}, sim={sim}, "
        f"real_count={real_count} > cap={TOKEN_CAP}. "
        "Tighten the scaffold text for this qtype or raise TOKEN_CAP with justification."
    )


# --------------------------------------------------------------------------- #
# Parametrised: every qtype × top_score under 200 tokens (o200k_base)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qtype", QTYPES)
@pytest.mark.parametrize("sim", SIM_VALUES)
def test_real_token_count_o200k_under_200(qtype, sim, enc_o200k):
    """o200k_base (GPT-4o tokenizer) real count must also stay <= 200."""
    text = wrap_system_prompt(qtype, top_score=sim)
    real_count = len(enc_o200k.encode(text))
    heuristic_count = _heuristic(text)
    print(
        f"\n[o200k] qtype={qtype!r} sim={sim} "
        f"chars={len(text)} heuristic={heuristic_count} real={real_count}"
    )
    assert real_count <= TOKEN_CAP, (
        f"REAL TOKEN OVERFLOW (o200k_base): qtype={qtype!r}, sim={sim}, "
        f"real_count={real_count} > cap={TOKEN_CAP}. "
        "Tighten the scaffold text for this qtype or raise TOKEN_CAP with justification."
    )


# --------------------------------------------------------------------------- #
# Heuristic conservatism check: does 4-char heuristic >= real count?
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qtype", QTYPES)
@pytest.mark.parametrize("sim", SIM_VALUES)
def test_heuristic_is_conservative_cl100k(qtype, sim, enc_cl100k):
    """The 4-chars-per-token heuristic in preflight.py should be >= real token count.

    If this fails it means preflight.py's length gate could pass a scaffold that
    is actually OVER-budget on cl100k_base. This is a design gap — not an overflow
    (the absolute cap test above handles that), but the heuristic provides false
    safety.
    """
    text = wrap_system_prompt(qtype, top_score=sim)
    real_count = len(enc_cl100k.encode(text))
    heuristic_count = _heuristic(text)
    print(
        f"\n[heuristic vs cl100k] qtype={qtype!r} sim={sim} "
        f"heuristic={heuristic_count} real={real_count} "
        f"delta={real_count - heuristic_count:+d}"
    )
    assert heuristic_count >= real_count, (
        f"HEURISTIC UNDER-ESTIMATES (cl100k_base): qtype={qtype!r}, sim={sim}, "
        f"heuristic={heuristic_count} < real={real_count}. "
        "The 4-char/token approximation is not conservative here — "
        "preflight.py's token-budget gate is not a safe upper bound for this input. "
        "Consider switching to real tokenizer enforcement in preflight()."
    )


# --------------------------------------------------------------------------- #
# Worst-case combination: maximize token count across all combos (cl100k)
# --------------------------------------------------------------------------- #

def test_worst_case_combination_cl100k(enc_cl100k):
    """Identify and hard-assert the single worst-case qtype + sim under cl100k.

    This test finds the maximum real token count across all combos and asserts
    it is still under cap. It also prints a leaderboard for visibility.
    """
    results = []
    for qtype in QTYPES:
        for sim in SIM_VALUES:
            text = wrap_system_prompt(qtype, top_score=sim)
            real = len(enc_cl100k.encode(text))
            heuristic = _heuristic(text)
            results.append((real, heuristic, qtype, sim, len(text)))

    results.sort(reverse=True)

    print("\n[worst-case leaderboard — cl100k_base]")
    print(f"{'rank':<5} {'qtype':<35} {'sim':>6} {'chars':>7} {'heur':>6} {'real':>6}")
    print("-" * 70)
    for i, (real, heur, qtype, sim, chars) in enumerate(results[:10], 1):
        flag = " <<< WORST" if i == 1 else ""
        under = " HEUR_UNDER" if heur < real else ""
        print(f"{i:<5} {qtype:<35} {str(sim):>6} {chars:>7} {heur:>6} {real:>6}{flag}{under}")

    worst_real, _, worst_qtype, worst_sim, _ = results[0]
    assert worst_real <= TOKEN_CAP, (
        f"WORST-CASE OVERFLOW: qtype={worst_qtype!r}, sim={worst_sim}, "
        f"real={worst_real} > cap={TOKEN_CAP}"
    )


# --------------------------------------------------------------------------- #
# Worst-case combination: o200k_base
# --------------------------------------------------------------------------- #

def test_worst_case_combination_o200k(enc_o200k):
    """Same worst-case analysis under o200k_base (GPT-4o tokenizer)."""
    results = []
    for qtype in QTYPES:
        for sim in SIM_VALUES:
            text = wrap_system_prompt(qtype, top_score=sim)
            real = len(enc_o200k.encode(text))
            heuristic = _heuristic(text)
            results.append((real, heuristic, qtype, sim, len(text)))

    results.sort(reverse=True)

    print("\n[worst-case leaderboard — o200k_base]")
    print(f"{'rank':<5} {'qtype':<35} {'sim':>6} {'chars':>7} {'heur':>6} {'real':>6}")
    print("-" * 70)
    for i, (real, heur, qtype, sim, chars) in enumerate(results[:10], 1):
        flag = " <<< WORST" if i == 1 else ""
        under = " HEUR_UNDER" if heur < real else ""
        print(f"{i:<5} {qtype:<35} {str(sim):>6} {chars:>7} {heur:>6} {real:>6}{flag}{under}")

    worst_real, _, worst_qtype, worst_sim, _ = results[0]
    assert worst_real <= TOKEN_CAP, (
        f"WORST-CASE OVERFLOW (o200k): qtype={worst_qtype!r}, sim={worst_sim}, "
        f"real={worst_real} > cap={TOKEN_CAP}"
    )
