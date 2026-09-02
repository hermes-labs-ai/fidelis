"""Deterministic, evidence-bound context planning for agent turns.

The planner answers a narrower question than retrieval: does this utterance
need historical context, and which evidence lane should be searched?  It does
not summarize memories or create truth.  Callers keep the returned records
verbatim and may use the plan as an orientation card for an agent.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


_KNOWN_ENTITIES = ("Fidelis", "Cogito", "Hermes", "Hercules", "RoliTwin")
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b")
_RECALL_RE = re.compile(
    r"\b(?:remember|recall|catch (?:me )?up|where did we leave|"
    r"previous (?:work|chat|task)|earlier (?:work|chat|task)|"
    r"from (?:a few|several|two|three) (?:days|weeks|months) ago)\b",
    re.IGNORECASE,
)
_PAST_WORK_RE = re.compile(
    r"\b(?:our|my|we)\b.*\b(?:project|repo(?:sitory)?|work|decision|"
    r"release|implementation|experiment|system|chat|task)\b",
    re.IGNORECASE,
)
_MAINTENANCE_RE = re.compile(
    r"\b(?:fix|debug|maintain|implement|integrat|test|deploy|ship|repo|"
    r"code|bug|issue|dependency|installer)\w*\b",
    re.IGNORECASE,
)
_TRANSFER_RE = re.compile(
    r"\b(?:apply|adapt|reuse|transfer|help with|use .* for|could .* help)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"\b(?:compare|different from|versus|vs\.?|relationship between)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:why did we|why we|decision|rationale|decid\w*|chos\w*|chose)\b",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"\b(?:originally|at the time|back then|histor\w*|timeline|months? ago|"
    r"weeks? ago|days? ago|previously|earlier)\b",
    re.IGNORECASE,
)
_CURRENT_RE = re.compile(
    r"\b(?:current(?:ly)?|latest|now|today|status|where (?:is|are) .* now)\b",
    re.IGNORECASE,
)
_IDENTITY_RE = re.compile(r"^\s*(?:what|who)\s+(?:is|was|are)\b", re.IGNORECASE)
_CASUAL_RE = re.compile(
    r"\b(?:weird|funny|good|bad|interesting|strange)\b[^?]*[.!]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ContextPlan:
    """Closed plan for one context-sensitive retrieval decision."""

    schema: str
    disposition: str
    entity: str | None
    conversational_role: str
    evidence_lane: str | None
    retrieval_query: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entity_from_turns(utterance: str, recent_turns: Iterable[str]) -> str | None:
    combined = [utterance, *list(recent_turns)[-4:]]
    for index, text in enumerate(combined):
        lowered = text.casefold()
        for entity in _KNOWN_ENTITIES:
            if entity.casefold() in lowered:
                return entity
        # Capitalization alone is not evidence that a turn names prior work:
        # sentence starters such as "Can", "Please", and "How" otherwise
        # become false project entities.  Infer an unknown proper noun only
        # from recent context or when the current turn itself asks for
        # identity/history/current-state context.  Callers can always provide
        # ``entity_hint`` for a new project name.
        allow_unknown = index > 0 or any(
            pattern.search(text)
            for pattern in (
                _RECALL_RE,
                _PAST_WORK_RE,
                _TRANSFER_RE,
                _COMPARISON_RE,
                _DECISION_RE,
                _HISTORICAL_RE,
                _CURRENT_RE,
                _IDENTITY_RE,
            )
        )
        if not allow_unknown:
            continue
        candidates = [
            item
            for item in _ENTITY_RE.findall(text)
            if item.casefold()
            not in {
                "what",
                "where",
                "when",
                "why",
                "could",
                "should",
                "write",
                "draft",
                "create",
                "build",
                "explain",
                "review",
                "summarize",
                "can",
                "how",
                "please",
                "tell",
                "this",
                "that",
                "the",
                "we",
                "our",
                "my",
                "i",
            }
        ]
        if candidates:
            return candidates[0]
    return None


def _query(entity: str | None, lane: str, utterance: str) -> str:
    subject = entity or utterance.strip()
    suffixes = {
        "identity": "identity purpose aliases orientation",
        "maintenance": "repository implementation status tests issues decisions",
        "conceptual": "purpose conceptual rationale design possible reuse",
        "comparison": "comparison relationship differences decisions",
        "decision": "decision rationale reason evidence",
        "historical": "historical state timeline original intent",
        "current": "current state latest supported decision status",
        "context": "prior work context decisions current state",
    }
    return f"{subject} {suffixes[lane]}".strip()


def plan_context(
    utterance: str,
    *,
    recent_turns: Iterable[str] = (),
    entity_hint: str | None = None,
) -> ContextPlan:
    """Classify an utterance, including non-questions, into one evidence lane."""

    local = utterance.strip()
    recent = tuple(recent_turns)[-4:]
    entity = entity_hint.strip() if entity_hint and entity_hint.strip() else None
    entity = entity or _entity_from_turns(local, recent)
    source_bound = bool(entity or _RECALL_RE.search(local) or _PAST_WORK_RE.search(local))

    if not local:
        role, lane, reason = "no_memory_needed", None, "empty utterance"
    elif not source_bound:
        role, lane, reason = (
            "no_memory_needed",
            None,
            "no known referent or historical-context cue",
        )
    elif _COMPARISON_RE.search(local):
        role, lane, reason = "comparative_context", "comparison", "comparison cue"
    elif _MAINTENANCE_RE.search(local):
        role, lane, reason = "maintenance_context", "maintenance", "maintenance cue"
    elif _TRANSFER_RE.search(local):
        role, lane, reason = "conceptual_transfer", "conceptual", "reuse cue"
    elif _DECISION_RE.search(local):
        role, lane, reason = "decision_context", "decision", "decision cue"
    elif _HISTORICAL_RE.search(local):
        role, lane, reason = "historical_recall", "historical", "historical cue"
    elif _CURRENT_RE.search(local):
        role, lane, reason = "context_refresh", "current", "current-state cue"
    elif _IDENTITY_RE.search(local) or (entity and _CASUAL_RE.search(local)):
        role, lane, reason = "identity_only", "identity", "orientation is sufficient"
    else:
        role, lane, reason = "context_refresh", "context", "historical context requested"

    disposition = "abstain" if lane is None else "retrieve"
    return ContextPlan(
        schema="fidelis-context-plan/v1",
        disposition=disposition,
        entity=entity,
        conversational_role=role,
        evidence_lane=lane,
        retrieval_query=_query(entity, lane, local) if lane else None,
        reason=reason,
    )


def context_packet(plan: ContextPlan, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind a plan to unmodified evidence records; never synthesize a summary."""

    selected = []
    for index, record in enumerate(records):
        item = dict(record)
        item.setdefault("record_id", item.get("id") or f"result-{index + 1}")
        selected.append(item)
    return {
        "schema": "fidelis-context-packet/v1",
        "orientation": {
            "entity": plan.entity,
            "conversational_role": plan.conversational_role,
            "evidence_lane": plan.evidence_lane,
            "authority": "derived index; records remain verbatim evidence",
        },
        "plan": plan.to_dict(),
        "records": selected,
        "evidence_status": "available" if selected else "insufficient",
    }
