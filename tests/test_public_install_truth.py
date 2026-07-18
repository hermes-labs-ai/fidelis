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
        assert "pip install fidelis" not in text, path
        assert "pypi.org/project/fidelis" not in text, path


def test_primary_surfaces_pin_current_github_source_release():
    for path in (ROOT / "README.md", ROOT / "llms.txt", ROOT / "docs" / "full-reference.md"):
        text = path.read_text()
        assert "--branch v0.0.91 --depth 1" in text, path
        assert "python3 -m pip install ." in text, path


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


def test_readme_scopes_no_telemetry_claim_to_documented_service_path():
    readme = (ROOT / "README.md").read_text()
    assert "no outbound network calls" not in readme
    assert "Direct server launches need explicit telemetry settings" in readme
    assert "MEM0_TELEMETRY=False" in readme
