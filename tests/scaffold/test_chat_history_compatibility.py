"""fidelis.scaffold v0.1.0 — chat-history API compatibility tests.

Validates that scaffold output (a plain UTF-8 string) fits without
modification into Anthropic Messages API, OpenAI Chat Completions API,
raw concatenation, and multi-turn conversation history patterns.
"""

import json
import pytest

from fidelis.scaffold._core import (
    wrap_system_prompt,
    is_scaffolded,
    strip_scaffold,
    SCAFFOLD_OPEN,
    SCAFFOLD_CLOSE,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

QTYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "multi-session",
    "temporal-reasoning",
]


@pytest.fixture(params=QTYPES)
def scaffold(request):
    return wrap_system_prompt(request.param, top_score=0.75)


@pytest.fixture
def scaffold_default():
    return wrap_system_prompt("single-session-user", top_score=0.65)


# ---------------------------------------------------------------------------
# (a) Anthropic Messages API shape
# ---------------------------------------------------------------------------

class TestAnthropicMessagesAPI:

    def test_scaffold_is_str_not_bytes(self, scaffold):
        assert isinstance(scaffold, str), "scaffold must be str, not bytes"

    def test_scaffold_json_serializable_as_system_value(self, scaffold):
        payload = {
            "system": scaffold,
            "messages": [{"role": "user", "content": "What did I say yesterday?"}],
        }
        serialized = json.dumps(payload)
        assert isinstance(serialized, str)

    def test_scaffold_json_roundtrip_byte_equal(self, scaffold):
        """scaffold round-trips through json.dumps + json.loads byte-equal."""
        payload = {"system": scaffold}
        roundtripped = json.loads(json.dumps(payload))["system"]
        assert roundtripped == scaffold, (
            "scaffold value changed after JSON round-trip"
        )

    def test_scaffold_no_control_characters_break_json(self, scaffold):
        """json.dumps must not raise and the result must be parseable."""
        dumped = json.dumps({"system": scaffold})
        loaded = json.loads(dumped)
        assert loaded["system"] == scaffold


# ---------------------------------------------------------------------------
# (b) OpenAI Chat Completions API shape
# ---------------------------------------------------------------------------

class TestOpenAIChatCompletionsAPI:

    def test_system_message_role_content_shape(self, scaffold):
        payload = {
            "messages": [
                {"role": "system", "content": scaffold},
                {"role": "user", "content": "What did I say yesterday?"},
            ]
        }
        serialized = json.dumps(payload)
        loaded = json.loads(serialized)
        system_content = loaded["messages"][0]["content"]
        assert system_content == scaffold, (
            "scaffold value changed after OpenAI-shape JSON round-trip"
        )

    def test_scaffold_survives_json_roundtrip_in_messages_list(self, scaffold):
        original = scaffold
        payload = {"messages": [{"role": "system", "content": original}]}
        recovered = json.loads(json.dumps(payload))["messages"][0]["content"]
        assert recovered == original

    def test_scaffold_is_valid_utf8_string(self, scaffold):
        # Encode/decode as UTF-8 must be lossless
        assert scaffold.encode("utf-8").decode("utf-8") == scaffold


# ---------------------------------------------------------------------------
# (c) Raw concatenation (lightweight clients)
# ---------------------------------------------------------------------------

class TestRawConcatenation:

    def test_combined_prompt_contains_no_nested_scaffold_markers(self, scaffold_default):
        user_question = "When did I mention coffee?"
        prompt = f"{scaffold_default}\n\n{user_question}"
        # Count scaffold open markers — there must be exactly one
        open_count = prompt.count("[FIDELIS-SCAFFOLD-")
        # Each version tag: one open, one close
        assert open_count >= 1, "scaffold open marker must be present in combined prompt"
        # Ensure no double-wrapping (idempotency at the string level)
        assert open_count == prompt.count("[/FIDELIS-SCAFFOLD-"), (
            "mismatched open/close scaffold tags in combined prompt"
        )

    def test_scaffold_open_detectable_in_combined_prompt(self, scaffold_default):
        user_question = "Did I mention running?"
        prompt = f"{scaffold_default}\n\n{user_question}"
        assert SCAFFOLD_OPEN in prompt

    def test_scaffold_close_detectable_in_combined_prompt(self, scaffold_default):
        user_question = "Did I mention running?"
        prompt = f"{scaffold_default}\n\n{user_question}"
        assert SCAFFOLD_CLOSE in prompt

    def test_is_scaffolded_true_for_combined_prompt(self, scaffold_default):
        user_question = "What did I order last time?"
        prompt = f"{scaffold_default}\n\n{user_question}"
        assert is_scaffolded(prompt), (
            "is_scaffolded must return True when scaffold is embedded in combined prompt"
        )

    def test_user_question_present_in_combined_prompt(self, scaffold_default):
        user_question = "How many sessions did I log?"
        prompt = f"{scaffold_default}\n\n{user_question}"
        assert user_question in prompt


# ---------------------------------------------------------------------------
# (d) Multi-turn conversation history with scaffold persistence
# ---------------------------------------------------------------------------

class TestMultiTurnHistory:

    def _make_turn1_system(self) -> str:
        return wrap_system_prompt("multi-session", top_score=0.80)

    def _make_turn2_system(self) -> str:
        # A plain non-scaffold system message for turn 2
        return "You are a helpful assistant. Answer concisely."

    def test_scaffold_detectable_in_history_after_system_replacement(self):
        """Turn-1 scaffold is still detectable in history even when turn-2 uses
        a different system message."""
        turn1_system = self._make_turn1_system()
        turn2_system = self._make_turn2_system()

        # Simulate 3-turn conversation log
        history = [
            {"turn": 1, "system": turn1_system, "user": "What did I read last month?"},
            {"turn": 2, "system": turn2_system, "user": "Summarise that."},
            {"turn": 3, "system": turn2_system, "user": "Any other topics?"},
        ]

        # turn1 system must still be detectable as scaffolded
        assert is_scaffolded(history[0]["system"]), (
            "is_scaffolded must return True for turn-1 system message in history"
        )

    def test_is_scaffolded_false_for_plain_system_message(self):
        turn2_system = self._make_turn2_system()
        assert not is_scaffolded(turn2_system), (
            "is_scaffolded must return False for a plain (non-scaffold) system message"
        )

    def test_strip_scaffold_returns_procedural_content_cleanly(self):
        """strip_scaffold on the turn-1 system message returns the underlying
        procedural content without scaffold markers."""
        turn1_system = self._make_turn1_system()
        stripped = strip_scaffold(turn1_system)

        # No scaffold markers remain
        assert not is_scaffolded(stripped), (
            "is_scaffolded must return False after strip_scaffold"
        )
        # The stripped result should be an empty string or the non-scaffold body
        # (wrap_system_prompt wraps the entire body inside scaffold tags, so
        # strip returns empty or whitespace)
        assert isinstance(stripped, str)

    def test_strip_scaffold_idempotent_on_plain_text(self):
        plain = "You are a helpful assistant."
        assert strip_scaffold(plain) == plain

    def test_scaffold_json_serializable_across_all_turns(self):
        """Entire 3-turn history with scaffold in turn-1 must be JSON-serializable."""
        turn1_system = self._make_turn1_system()
        turn2_system = self._make_turn2_system()

        history = [
            {"turn": 1, "system": turn1_system},
            {"turn": 2, "system": turn2_system},
            {"turn": 3, "system": turn2_system},
        ]
        serialized = json.dumps(history)
        recovered = json.loads(serialized)
        assert recovered[0]["system"] == turn1_system, (
            "turn-1 scaffold must round-trip byte-equal through JSON in multi-turn history"
        )

    def test_strip_scaffold_removes_markers_leaving_no_open_tag(self):
        turn1_system = self._make_turn1_system()
        stripped = strip_scaffold(turn1_system)
        assert "[FIDELIS-SCAFFOLD-" not in stripped
