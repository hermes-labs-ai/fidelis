"""Server-free re-verification of examples/real-world/*.

Every check here calls a pure function or parses a file. Nothing starts the
HTTP service, contacts Ollama, opens a store, or imports the MCP/CLI modules
(those are inspected with ``ast`` only).
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from fidelis import supersession
from fidelis.temporal import (
    SupersessionIndex,
    apply_temporal,
    build_temporal_fields,
)
from fidelis.write_gate import evaluate

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "real-world"


def _load(*parts: str):
    return json.loads(EX.joinpath(*parts).read_text(encoding="utf-8"))


def test_every_example_has_input_expected_and_readme():
    dirs = sorted(p for p in EX.iterdir() if p.is_dir())
    assert len(dirs) >= 3
    for d in dirs:
        assert (d / "README.md").is_file(), d
        assert list(d.glob("input*.json")), d
        assert (d / "expected.json").is_file(), d
        readme = (d / "README.md").read_text(encoding="utf-8")
        assert "## Limits" in readme and "## Provenance" in readme, d


# --- correction chain (fidelis.temporal) -----------------------------------


def test_correction_chain_matches_expected():
    data = _load("correction-chain-temporal", "input.json")
    expected = _load("correction-chain-temporal", "expected.json")
    records = {r["id"]: r for r in data["records"]}
    payloads: dict[str, dict] = {}
    edges = []
    for r in data["records"]:
        fields = build_temporal_fields(
            r["text"], now=r["write_time"], declared=r["declared"], record_id=r["id"]
        )
        payloads[r["id"]] = fields
        for target in json.loads(fields.get("supersedes_json", "[]")):
            edges.append((target, r["id"], fields["recorded_at"]))
    index = SupersessionIndex.from_edges(edges)

    actual = {}
    for view in data["views"]:
        hits = [
            {"id": i, "text": records[i]["text"], "payload": payloads[i]}
            for i in data["relevance_order"]
        ]
        result = apply_temporal(hits, index=index, now=view["now"], as_of=view["as_of"])
        actual[view["name"]] = [
            {"id": h["id"], "text": h["text"], "temporal": h["temporal"]} for h in result
        ]
    assert actual == expected

    current = {h["id"]: h for h in actual["current_view"]}
    assert current["rec-a"]["temporal"]["status"] == "superseded"  # kept, not dropped
    assert current["rec-a"]["text"] == records["rec-a"]["text"]  # verbatim
    assert [h["id"] for h in actual["as_of_before_correction"]] == ["rec-a", "rec-filler"]


def test_correction_chain_texts_come_from_the_regression_test():
    source = (ROOT / "tests" / "test_superseded_not_dropped.py").read_text(encoding="utf-8")
    data = _load("correction-chain-temporal", "input.json")
    for rec in data["records"]:
        if rec["id"] in ("rec-a", "rec-b"):
            assert rec["text"] in source


# --- retraction / ephemera (fidelis.supersession) --------------------------


def _pointer_results(tmp_path, monkeypatch):
    pointers = _load("retraction-and-ephemera-read-filter", "pointers.json")
    hits = _load("retraction-and-ephemera-read-filter", "input.json")["hits"]
    path = tmp_path / "pointers.json"
    path.write_text(json.dumps(pointers), encoding="utf-8")
    monkeypatch.setattr(supersession, "_cache", {"key": None, "rules": None})
    cfg = {"supersession_pointers_path": str(path)}
    return hits, {
        "annotate_superseded": supersession.annotate_superseded(hits, cfg),
        "mark_superseded_metadata": [
            {"id": m["id"], "supersession": m["supersession"]}
            for m in supersession.mark_superseded(hits, cfg)
            if "supersession" in m
        ],
        "filter_ephemera_kept_ids": [m["id"] for m in supersession.filter_ephemera(hits, cfg)],
        "mark_ephemera_flagged_ids": [
            m["id"] for m in supersession.mark_ephemera(hits, cfg) if "ephemera" in m
        ],
    }


def test_retraction_and_ephemera_match_expected(tmp_path, monkeypatch):
    hits, actual = _pointer_results(tmp_path, monkeypatch)
    assert actual == _load("retraction-and-ephemera-read-filter", "expected.json")
    # annotation never drops a hit
    assert [m["id"] for m in actual["annotate_superseded"]] == [h["id"] for h in hits]


def test_mark_superseded_keeps_source_text_byte_identical(tmp_path, monkeypatch):
    hits = _load("retraction-and-ephemera-read-filter", "input.json")["hits"]
    pointers = tmp_path / "p.json"
    pointers.write_text(
        (EX / "retraction-and-ephemera-read-filter" / "pointers.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr(supersession, "_cache", {"key": None, "rules": None})
    marked = supersession.mark_superseded(hits, {"supersession_pointers_path": str(pointers)})
    assert [m["text"] for m in marked] == [h["text"] for h in hits]


def test_missing_pointers_file_fails_open(tmp_path, monkeypatch):
    hits = _load("retraction-and-ephemera-read-filter", "input.json")["hits"]
    monkeypatch.setattr(supersession, "_cache", {"key": None, "rules": None})
    cfg = {"supersession_pointers_path": str(tmp_path / "absent.json")}
    assert supersession.filter_ephemera(hits, cfg) == hits
    assert supersession.annotate_superseded(hits, cfg) == hits


# --- write gate (fidelis.write_gate) ---------------------------------------


def test_write_gate_matches_expected():
    texts = _load("write-gate-screening", "input.json")["texts"]
    expected = _load("write-gate-screening", "expected.json")
    actual = []
    for text in texts:
        d = evaluate(text)
        actual.append(
            {"text": text, "accept": d.accept, "reason": d.reason, "detail": d.detail}
        )
    assert actual == expected
    assert {e["reason"] for e in expected if not e["accept"]} == {
        "scaffold_prompt",
        "probe_residue",
        "harness_envelope",
    }


def test_write_gate_strings_come_from_the_gate_test():
    source = (ROOT / "tests" / "test_write_gate.py").read_text(encoding="utf-8")
    for text in _load("write-gate-screening", "input.json")["texts"]:
        first_line = text.splitlines()[0]
        assert first_line in source, text


# --- MCP client config (static only) ---------------------------------------


def _pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)
    scripts_block = re.search(r"\[project\.scripts\]\n(.*?)(?:\n\[|\Z)", text, re.DOTALL).group(1)
    scripts = dict(re.findall(r'^([\w-]+)\s*=\s*"([^"]+)"', scripts_block, re.MULTILINE))
    return version, scripts


def _module_ast(name: str) -> ast.Module:
    return ast.parse((ROOT / "src" / "fidelis" / name).read_text(encoding="utf-8"))


def _tool_names_from_ast(tree: ast.Module) -> set[str]:
    """Keys of the module-level dict that maps tool names to handlers."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and node.keys:
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if len(keys) == len(node.keys) and keys and all(
                isinstance(k, str) and k.startswith("fidelis_") for k in keys
            ):
                return set(keys)
    raise AssertionError("tool dispatch table not found")


@pytest.mark.parametrize(
    "name", ["input-mcp.json", "input-gemini-extension.json"]
)
def test_mcp_config_launches_the_declared_entry_point(name):
    version, scripts = _pyproject()
    expected = _load("mcp-client-config", "expected.json")
    entry = _load("mcp-client-config", name)["mcpServers"]["fidelis"]

    assert entry["command"] == expected["launch"]["executable"]
    args = entry["args"]
    assert args[:2] == ["--from", f"fidelis-memory=={version}"]
    assert args[2] == expected["launch"]["console_script"]
    assert args[3:] == expected["launch"]["argv"]
    assert scripts[args[2]] == expected["console_script_targets"]["fidelis"]
    assert scripts["fidelis-mcp"] == expected["console_script_targets"]["fidelis-mcp"]


def test_mcp_entry_points_and_tools_exist_statically():
    _, scripts = _pyproject()
    expected = _load("mcp-client-config", "expected.json")
    for target in expected["console_script_targets"].values():
        module, func = target.split(":")
        assert scripts and module.startswith("fidelis.")
        tree = _module_ast(module.split(".")[1] + ".py")
        assert any(
            isinstance(n, ast.FunctionDef) and n.name == func for n in tree.body
        ), target

    cli_source = (ROOT / "src" / "fidelis" / "cli.py").read_text(encoding="utf-8")
    assert 'sub.add_parser("mcp"' in cli_source
    assert 'mcp_sub.add_parser("serve"' in cli_source

    assert _tool_names_from_ast(_module_ast("mcp_server.py")) == set(expected["tools"])


def test_mcp_tool_names_match_public_docs():
    expected = _load("mcp-client-config", "expected.json")
    reference = (ROOT / "docs" / "full-reference.md").read_text(encoding="utf-8")
    for tool in expected["tools"]:
        assert f"`{tool}`" in reference


# --- benchmark case (parsing only) -----------------------------------------


def test_bench_cases_are_verbatim_hardset_entries():
    hardset = {c["qid"]: c for c in json.loads((ROOT / "bench" / "hardset.json").read_text())}
    cases = _load("bench-hardset-hit-at-5", "input.json")["cases"]
    assert len(cases) == 3
    for case in cases:
        assert hardset[case["qid"]] == case


def test_bench_expected_metrics_recompute():
    cases = _load("bench-hardset-hit-at-5", "input.json")["cases"]
    actual = []
    for c in cases:
        gold = set(c["gold_session_ids"])
        found = sorted(gold & set(c["s1_top5_ids"]))
        actual.append(
            {
                "qid": c["qid"],
                "n_gold": len(gold),
                "gold_in_top5": found,
                "hit_at_5_any_gold": bool(found),
                "gold_recall_at_5": round(len(found) / len(gold), 4),
                "all_gold_in_top5": len(found) == len(gold),
                "matches_recorded_s1_hit_at_5": bool(found) == c["s1_hit_at_5"],
            }
        )
    assert actual == _load("bench-hardset-hit-at-5", "expected.json")
    # hit@5 is "any gold": one case counts as a hit while finding 1 of 3 gold sessions.
    assert any(a["hit_at_5_any_gold"] and not a["all_gold_in_top5"] for a in actual)


def test_all_recorded_hardset_flags_agree_with_id_lists():
    for c in json.loads((ROOT / "bench" / "hardset.json").read_text()):
        gold = set(c["gold_session_ids"])
        assert bool(gold & set(c["s1_top5_ids"])) == c["s1_hit_at_5"], c["qid"]
        assert bool(gold & set(c["s2_top5_ids"])) == c["s2_hit_at_5"], c["qid"]
