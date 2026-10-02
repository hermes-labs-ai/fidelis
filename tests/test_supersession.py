"""Server-free contracts for read-time supersession and ephemera pointers."""

import copy
import json
import os

import pytest

from fidelis import supersession


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(supersession, "_cache", {"key": None, "rules": None})


@pytest.fixture
def pointers(tmp_path):
    path = tmp_path / "pointers.json"
    path.write_text(json.dumps({
        "retracted": [
            {"pattern": "old claim", "status": "RETRACTED", "note": "replaced", "id": "r1"},
            {"pattern": "claim", "note": "second match", "id": "r2"},
        ],
        "ephemera_patterns": [r"^tool_result", r"^You are QA"],
    }), encoding="utf-8")
    return path


def hits():
    return [
        {"id": "old", "text": "OLD CLAIM", "score": 0.9},
        {"id": "new", "text": "new claim", "score": 0.8},
        {"id": "junk", "text": "tool_result: transient"},
        {"id": "empty", "text": None},
        {"id": "missing"},
    ]


APIS = [supersession.filter_ephemera, supersession.mark_ephemera,
        supersession.mark_superseded, supersession.annotate_superseded]


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("contents", [None, "not json", '{"retracted": [{"pattern": "["}], '
                         '"ephemera_patterns": ["["]}', "{}"])
def test_missing_corrupt_or_empty_pointers_fail_open(tmp_path, api, contents):
    path = tmp_path / "pointers.json"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    memories = hits()
    before = copy.deepcopy(memories)
    assert api(memories, {"supersession_pointers_path": str(path)}) == before
    assert memories == before


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("cfg", [{}, {"supersession_pointers_path": None},
                                {"supersession_pointers_path": ""}])
def test_unconfigured_pointers_fail_open(api, cfg):
    memories = hits()
    assert api(memories, cfg) == memories


@pytest.mark.parametrize("api", [supersession.mark_superseded, supersession.annotate_superseded])
def test_supersession_keeps_order_and_input_and_uses_first_rule(pointers, api):
    memories = hits()
    before = copy.deepcopy(memories)
    result = api(memories, {"supersession_pointers_path": str(pointers)})
    assert [m["id"] for m in result] == [m["id"] for m in memories]
    assert memories == before
    assert result[0]["supersession"] == {"status": "RETRACTED", "id": "r1", "note": "replaced"}
    assert result[1]["supersession"] == {
        "status": "SUPERSEDED", "id": "r2", "note": "second match",
    }
    assert result[2:] == before[2:]
    if api is supersession.mark_superseded:
        assert [m.get("text") for m in result] == [m.get("text") for m in before]
    else:
        assert result[0]["text"] == "[RETRACTED:r1] replaced || original: OLD CLAIM"
        assert result[1]["text"] == "[SUPERSEDED:r2] second match || original: new claim"


def test_supersession_rule_defaults(pointers):
    pointers.write_text(json.dumps({"retracted": [{"pattern": "old"}]}), encoding="utf-8")
    result = supersession.mark_superseded(hits(), {"supersession_pointers_path": str(pointers)})
    assert result[0]["supersession"] == {"status": "SUPERSEDED", "id": "", "note": ""}


@pytest.mark.parametrize("text", ["tool_result: raw", "User: tool_result: role",
                                 "[voice:user] [tag] TOOL_RESULT: nested",
                                 "useful turn\nAssistant: You are QA for this task"])
def test_ephemera_filter_and_marker_handle_storage_envelopes(pointers, text):
    memories = [{"id": "noise", "text": text}, {"id": "fact", "text": "durable fact"},
                {"id": "empty", "text": ""}, {"id": "missing"}]
    before = copy.deepcopy(memories)
    cfg = {"supersession_pointers_path": str(pointers)}
    assert supersession.filter_ephemera(memories, cfg) == memories[1:]
    result = supersession.mark_ephemera(memories, cfg)
    assert [m["id"] for m in result] == [m["id"] for m in memories]
    assert result[0] == {**memories[0], "ephemera": {
        "matched": True, "source": "current_truth_ephemera_pattern",
    }}
    assert result[1:] == before[1:]
    assert memories == before


@pytest.mark.parametrize("mode", ["enabled", "disabled", "missing", "unconfigured", "empty"])
def test_filter_honors_positive_limit_on_all_paths(pointers, mode):
    memories = [hits()[2], hits()[0], hits()[1]]
    cfg = {"supersession_pointers_path": str(pointers)}
    expected = memories[1:2]
    if mode == "disabled":
        cfg["ephemera_filter"] = False
        expected = memories[:1]
        assert supersession.mark_ephemera(memories, cfg) is memories
    elif mode == "missing":
        cfg["supersession_pointers_path"] = str(pointers.parent / "missing.json")
        expected = memories[:1]
    elif mode == "unconfigured":
        cfg = {}
        expected = memories[:1]
    elif mode == "empty":
        pointers.write_text("{}", encoding="utf-8")
        expected = memories[:1]
    assert supersession.filter_ephemera(memories, cfg, limit=1) == expected


@pytest.mark.parametrize("api,field,replacement", [
    (supersession.mark_superseded, "retracted", [{"pattern": "new", "id": "changed"}]),
    (supersession.mark_ephemera, "ephemera_patterns", ["^new"]),
])
def test_cache_reuses_unchanged_mtime_then_reloads(pointers, api, field, replacement):
    cfg = {"supersession_pointers_path": str(pointers)}
    memories = hits()
    original_stat = pointers.stat()
    before = api(memories, cfg)
    pointers.write_text(json.dumps({field: replacement}), encoding="utf-8")
    os.utime(pointers, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert api(memories, cfg) == before
    os.utime(pointers, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 2_000_000_000))
    after = api(memories, cfg)
    assert after != before
    assert after[0] == memories[0]
    if api is supersession.mark_superseded:
        assert after[1]["supersession"] == {"status": "SUPERSEDED", "id": "changed", "note": ""}
    else:
        assert after[1]["ephemera"]["matched"] is True


@pytest.mark.parametrize("api", [supersession.mark_superseded, supersession.mark_ephemera])
def test_cache_does_not_leak_across_pointer_paths(pointers, tmp_path, api):
    memories = hits()
    first = api(memories, {"supersession_pointers_path": str(pointers)})
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    stat = pointers.stat()
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert first != memories
    assert api(memories, {"supersession_pointers_path": str(other)}) == memories
