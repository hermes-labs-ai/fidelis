from __future__ import annotations

import json

from fidelis.context import context_packet, plan_context
from fidelis import mcp_server


def test_non_question_recall_request_triggers_context():
    plan = plan_context("I need to remember our Fidelis work from a few months ago")
    assert plan.disposition == "retrieve"
    assert plan.entity == "Fidelis"
    assert plan.conversational_role == "historical_recall"
    assert plan.evidence_lane == "historical"


def test_same_entity_selects_different_evidence_lanes():
    maintenance = plan_context("We need to fix Hercules and run its tests")
    transfer = plan_context("Could Hercules help with symbolic language systems?")
    assert maintenance.evidence_lane == "maintenance"
    assert transfer.evidence_lane == "conceptual"
    assert maintenance.retrieval_query != transfer.retrieval_query


def test_casual_known_entity_uses_identity_only_orientation():
    plan = plan_context("Hercules was a weird name.")
    assert plan.conversational_role == "identity_only"
    assert plan.evidence_lane == "identity"


def test_unrelated_turn_abstains():
    plan = plan_context("Write a haiku about the moon")
    assert plan.disposition == "abstain"
    assert plan.conversational_role == "no_memory_needed"
    assert plan.retrieval_query is None


def test_recent_turn_resolves_referent():
    plan = plan_context(
        "What is its current status?",
        recent_turns=["We were discussing Fidelis."],
    )
    assert plan.entity == "Fidelis"
    assert plan.evidence_lane == "current"


def test_packet_preserves_verbatim_record_and_pointer():
    record = {
        "id": "abc",
        "text": "The exact historical wording.",
        "score": 0.9,
        "metadata": {"source_pointer": "session.jsonl:42"},
    }
    packet = context_packet(plan_context("Recall Fidelis"), [record])
    assert packet["records"][0]["text"] == record["text"]
    assert packet["records"][0]["metadata"] == record["metadata"]
    assert packet["records"][0]["record_id"] == "abc"
    assert packet["orientation"]["authority"].startswith("derived index")


def test_mcp_orient_abstains_without_server_call(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_http_post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no recall")),
    )
    packet = json.loads(mcp_server._tool_orient({"utterance": "Write a haiku"}))
    assert packet["plan"]["disposition"] == "abstain"
    assert packet["evidence_status"] == "insufficient"


def test_mcp_orient_runs_bounded_zero_llm_lane(monkeypatch):
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {
            "memories": [
                {"id": "one", "text": "Fidelis is a local memory system.", "score": 1.0}
            ]
        }

    monkeypatch.setattr(mcp_server, "_http_post", fake_post)
    packet = json.loads(
        mcp_server._tool_orient(
            {"utterance": "We need to maintain Fidelis", "limit": 99}
        )
    )
    assert calls[0][0] == "/recall_hybrid"
    assert calls[0][1]["tier"] == "zero_llm"
    assert calls[0][1]["limit"] == 20
    assert "repository implementation status" in calls[0][1]["text"]
    assert packet["records"][0]["text"] == "Fidelis is a local memory system."
