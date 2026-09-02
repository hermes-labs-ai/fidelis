"""Keep public install and evidence links bound to this repository."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SURFACES = (
    ROOT / "README.md",
    ROOT / "llms.txt",
    ROOT / "agents.md",
    ROOT / "docs" / "full-reference.md",
)


def test_public_surfaces_do_not_install_unrelated_pypi_project():
    for path in PUBLIC_SURFACES:
        text = path.read_text()
        assert "pip install fidelis\n" not in text, path
        assert "pip install fidelis " not in text, path
        assert "pypi.org/project/fidelis" not in text, path


def test_primary_surfaces_install_the_fidelis_memory_distribution():
    for path in (ROOT / "README.md", ROOT / "llms.txt", ROOT / "docs" / "full-reference.md"):
        text = path.read_text()
        assert 'python3 -m pip install "fidelis-memory==0.0.95"' in text, path


def test_current_readme_links_shipped_benchmark_receipts():
    readme = (ROOT / "README.md").read_text()
    assert "bench/runs/zeroLLM-full-20260424/aggregate.json" not in readme
    retrieval = ROOT / "bench" / "runs" / "runP-v35" / "aggregate.json"
    qa = ROOT / "experiments" / "zeroLLM-FLAGSHIP-evidence" / "SUMMARY.json"
    assert retrieval.is_file()
    assert qa.is_file()
    assert "bench/runs/runP-v35/aggregate.json" in readme
    assert "experiments/zeroLLM-FLAGSHIP-evidence/SUMMARY.json" in readme
    recall = json.loads(retrieval.read_text())["stage1_metrics"]["recall_any"]
    assert recall["R@1"] == 0.832
    assert recall["R@5"] == 0.983


def test_llms_txt_headline_matches_readme_retrieval_claim():
    """llms.txt is the agent-facing summary; its headline R@1 must be the same
    zero-LLM stage-1 figure the README and aggregate.json carry, never the
    opt-in LLM-filter (stage 2) figure presented as the default-path result."""
    llms = (ROOT / "llms.txt").read_text()
    headline = llms.splitlines()[1]
    retrieval = ROOT / "bench" / "runs" / "runP-v35" / "aggregate.json"
    stage1 = json.loads(retrieval.read_text())["stage1_metrics"]["recall_any"]
    expected = f"{stage1['R@1'] * 100:.1f}% R@1"
    assert expected in headline, headline
    assert "zero-LLM" in headline, headline
    assert "96.4% R@1" not in headline, headline
    assert "bench/runs/runP-v35/aggregate.json" in llms


def test_readme_scopes_no_telemetry_claim_to_documented_defaults():
    readme = (ROOT / "README.md").read_text()
    assert "no outbound network calls" not in readme
    assert "Direct server launches disable mem0 telemetry by default" in readme
    assert "MEM0_TELEMETRY=True" in readme
    assert "ANONYMIZED_TELEMETRY=False" in readme
    assert "CHROMA_TELEMETRY_DISABLED=True" in readme
