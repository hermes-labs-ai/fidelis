"""
cogito scaffold — structured state memory for AI agents.

The model is stateless. The text is the state.
The scaffold rewrites every turn within a fixed token budget.
No retrieval, no embeddings, no vector search.

Architecture:
  - Agent sends: user message + assistant response after each turn
  - Server extracts: state delta (JSON) via small local model
  - Server returns: updated scaffold for agent's next system prompt
  - Scaffold rewrites in place — 1 becomes 2, not 1+1

Based on LPCI (Linguistically Persistent Cognitive Interface), Hermes Labs 2026.
"""

from __future__ import annotations

import dataclasses
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── State Schema ─────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """The entire cognitive state of a session. This IS the model's memory."""

    # Identity & mode
    role: str = ""
    style: str = ""

    # What we're doing
    goal: str = ""
    subgoals: list[str] = field(default_factory=list)

    # What we know
    decisions: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    # What we must not do
    constraints: list[str] = field(default_factory=list)

    # What's open
    open_threads: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)

    # Vocabulary — domain terms that anchor the session
    vocabulary: dict[str, str] = field(default_factory=dict)

    # Turn counter
    turn: int = 0

    def to_scaffold(self, token_budget: int = 7000) -> str:
        """Render state as a dense scaffold for model injection."""
        sections = []

        if self.role or self.style:
            sections.append(f"## Identity\nRole: {self.role}\nStyle: {self.style}")

        if self.goal:
            goal_block = f"## Current Goal\n{self.goal}"
            if self.subgoals:
                goal_block += "\nActive sub-tasks:\n" + "\n".join(f"- {s}" for s in self.subgoals)
            sections.append(goal_block)

        if self.decisions:
            sections.append("## Decisions (final)\n" + "\n".join(f"- {d}" for d in self.decisions))

        if self.facts:
            sections.append("## Known Facts\n" + "\n".join(f"- {f}" for f in self.facts))

        if self.artifacts:
            sections.append("## Artifacts Produced\n" + "\n".join(f"- {a}" for a in self.artifacts))

        if self.constraints:
            sections.append("## Constraints (MUST respect)\n" + "\n".join(f"- NOT: {c}" for c in self.constraints))

        if self.open_threads:
            sections.append("## Open Threads\n" + "\n".join(f"- {t}" for t in self.open_threads))

        if self.uncertainties:
            sections.append("## Uncertainties\n" + "\n".join(f"- {u}" for u in self.uncertainties))

        if self.vocabulary:
            vocab_lines = [f"- {k}: {v}" for k, v in self.vocabulary.items()]
            sections.append("## Vocabulary\n" + "\n".join(vocab_lines))

        sections.append(f"\n[Session turn: {self.turn}]")

        scaffold = "\n\n".join(sections)

        char_budget = token_budget * 4
        if len(scaffold) > char_budget:
            scaffold = self._trim_to_budget(char_budget)

        return scaffold

    def _trim_to_budget(self, char_budget: int) -> str:
        """Trim least important state to fit budget."""
        self.uncertainties = self.uncertainties[:3]
        if len(self.facts) > 10:
            self.facts = self.facts[-10:]
        if len(self.artifacts) > 5:
            self.artifacts = self.artifacts[-5:]
        if len(self.vocabulary) > 10:
            keys = list(self.vocabulary.keys())
            for k in keys[:-10]:
                del self.vocabulary[k]
        return self.to_scaffold(token_budget=char_budget // 4)


# ── State Extraction ─────────────────────────────────────────────────────────

_UPDATE_PROMPT = """You are a state extraction engine. Given a conversation turn (user message + assistant response), update the session state.

Current state:
{current_state}

Latest exchange:
User: {user_message}
Assistant: {assistant_response}

Extract state changes as JSON. Only include fields that changed. Possible fields:
- "goal": string (if goal changed or was clarified)
- "add_subgoals": [strings] (new sub-tasks identified)
- "remove_subgoals": [strings] (sub-tasks completed)
- "add_decisions": [strings] (new irreversible decisions)
- "add_facts": [strings] (new established truths)
- "add_artifacts": [strings] (new things produced)
- "add_constraints": [strings] (new hard boundaries)
- "add_open_threads": [strings] (new unresolved questions)
- "remove_open_threads": [strings] (questions resolved)
- "add_uncertainties": [strings] (new unknowns)
- "remove_uncertainties": [strings] (unknowns resolved)
- "add_vocabulary": {{"term": "meaning"}} (new domain terms)
- "style": string (if communication style changed)

Respond ONLY with valid JSON. Be precise and terse. Every word costs tokens."""


def extract_state_delta(
    state: SessionState,
    user_message: str,
    assistant_response: str,
    cfg: dict[str, Any],
) -> dict:
    """Use a small model to extract state changes from a conversation turn."""
    model = cfg.get("scaffold_model", "qwen3.5:4b")
    ollama_url = cfg.get("ollama_url", "http://localhost:11434")
    budget = cfg.get("scaffold_budget", 7000)

    prompt = _UPDATE_PROMPT.format(
        current_state=state.to_scaffold(token_budget=min(budget, 2000)),
        user_message=user_message[:1000],
        assistant_response=assistant_response[:1000],
    )

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }).encode()

    req = urllib.request.Request(
        f"{ollama_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            content = data.get("message", {}).get("content", "")
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group(0))
    except Exception as e:
        print(f"[scaffold] extraction failed: {e}", flush=True)

    return {}


def apply_delta(state: SessionState, delta: dict) -> None:
    """Apply extracted state changes to the session state."""
    if "goal" in delta:
        state.goal = delta["goal"]
    if "style" in delta:
        state.style = delta["style"]

    for key, attr in [
        ("add_subgoals", "subgoals"),
        ("add_decisions", "decisions"),
        ("add_facts", "facts"),
        ("add_artifacts", "artifacts"),
        ("add_constraints", "constraints"),
        ("add_open_threads", "open_threads"),
        ("add_uncertainties", "uncertainties"),
    ]:
        if key in delta and isinstance(delta[key], list):
            getattr(state, attr).extend(delta[key])

    for key, attr in [
        ("remove_subgoals", "subgoals"),
        ("remove_open_threads", "open_threads"),
        ("remove_uncertainties", "uncertainties"),
    ]:
        if key in delta and isinstance(delta[key], list):
            current = getattr(state, attr)
            for item in delta[key]:
                setattr(state, attr, [x for x in current if item.lower() not in x.lower()])

    if "add_vocabulary" in delta and isinstance(delta["add_vocabulary"], dict):
        state.vocabulary.update(delta["add_vocabulary"])

    state.turn += 1


# ── Persistence ──────────────────────────────────────────────────────────────

_SESSIONS_DIR = Path.home() / ".cogito" / "sessions"


def save_session(session_id: str, state: SessionState) -> Path:
    """Persist session state to disk."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _SESSIONS_DIR / f"{session_id}.json"
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(state), f, indent=2)
    return path


def load_session(session_id: str) -> SessionState | None:
    """Load session state from disk, or None if not found."""
    path = _SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return SessionState(**data)
    except Exception:
        return None
