"""Unit tests for the v0.7 Stage D prompt injection in therapeutic builders.

These tests cover the new ``_format_procedural_rules_block`` and
``_format_recall_toggle_constraint`` helpers in
``agent/therapeutic/prompts.py``, plus verification that the 6 public
system-prompt builders correctly weave the dynamic blocks into their
output based on state.

Coverage split:

1. The two helper functions directly — pure string-shape tests with
   crafted state dicts. Covers empty / populated / both-toggle-states
   cases for each helper.
2. The 6 builders (``build_supportive_system_prompt``,
   ``build_reflective_system_prompt``, ``build_clarifying_system_prompt``,
   ``build_psychoeducation_system_prompt``, ``build_closing_system_prompt``,
   ``build_guided_exercise_system_prompt``) — verifies that each
   builder:
     a. Returns the static knowledge+instructions content when memory
        is empty
     b. Appends the rules block when rules are populated
     c. Appends the recall-toggle constraint (off/on variant) based on
        the toggle state
     d. Correctly threads the dynamic blocks AFTER the instructions
3. Regression guard: the crisis response builder does NOT inject
   rules or recall toggle. This is a safety decision (see
   ``agent/prompts/crisis.py`` docstring for rationale).

These are shape tests, not quality tests. The eval harnesses handle
prompt-quality grading separately; these tests pin the mechanical
weaving of state into prompt text.
"""

from __future__ import annotations

from typing import Any, cast

from agent.prompts import build_crisis_response_system_prompt
from agent.state import AgentState
from agent.therapeutic.dispatcher import build_therapeutic_dispatch_system_prompt
from agent.therapeutic.prompts import (
    _format_procedural_rules_block,
    _format_recall_toggle_constraint,
    build_clarifying_system_prompt,
    build_closing_system_prompt,
    build_guided_exercise_system_prompt,
    build_psychoeducation_system_prompt,
    build_reflective_system_prompt,
    build_supportive_system_prompt,
    build_technique_system_prompt,
    build_therapeutic_response_prompt,
)


# ─── Test helpers ──────────────────────────────────────────────────────────


def _make_state(
    *,
    rules: list[str] | None = None,
    recall_enabled: bool | None = None,
    working_memory: list[Any] | None = None,
) -> AgentState:
    """Build a minimal AgentState with procedural fields configured.

    The builders under test only read ``state["procedural_profile"]`` for
    procedural rules and recall toggle, plus ``state["history"]`` and
    ``state["working_memory"]`` for the user-prompt helpers (which
    aren't exercised here). A partial dict is sufficient — AgentState
    is a TypedDict so the type annotation is a type-checker assertion,
    not a runtime constructor.
    """

    session_memory: dict[str, Any] = {
        "summary": "",
        "active_concerns": [],
        "open_loops": [],
        "current_goal": None,
    }
    procedural_profile: dict[str, Any] = {}
    if rules is not None:
        procedural_profile["procedural_rules"] = rules
    if recall_enabled is not None:
        procedural_profile["proactive_recall_enabled"] = recall_enabled

    state: dict[str, Any] = {
        "message": "hello",
        "history": [],
        "working_memory": working_memory or [],
        "session_memory": session_memory,
        "procedural_profile": procedural_profile,
    }
    return cast(AgentState, state)


# ─── _format_procedural_rules_block ────────────────────────────────────────


class TestRulesBlockHelper:
    """Direct tests of ``_format_procedural_rules_block``."""

    def test_empty_rules_returns_empty_string(self) -> None:
        """No rules → no injection. The prompt stays static."""

        state = _make_state(rules=[])
        assert _format_procedural_rules_block(state) == ""

    def test_missing_procedural_profile_returns_empty_string(self) -> None:
        """Missing ``procedural_rules`` key (e.g., pre-Stage-C state) is safe.

        The helper uses ``.get("procedural_rules") or []`` so a state
        without the field falls through to the empty case. This is
        the defensive path that keeps old test fixtures working.
        """

        state: dict[str, Any] = {
            "message": "hello",
            "history": [],
            "working_memory": [],
            "session_memory": {
                "summary": "",
                "active_concerns": [],
                "open_loops": [],
                "current_goal": None,
            },
        }
        block = _format_procedural_rules_block(cast(AgentState, state))
        assert block == ""

    def test_single_rule_produces_formatted_block(self) -> None:
        """One rule → block contains the rule text and silent-follow guidance."""

        state = _make_state(rules=["You've said meditation makes you more anxious."])
        block = _format_procedural_rules_block(state)

        assert "Style rules from past conversations" in block
        assert "- You've said meditation makes you more anxious." in block
        # The silent-follow guidance is present
        assert "Follow these rules silently" in block
        assert "Do NOT quote them" in block

    def test_multiple_rules_all_appear_as_bullets(self) -> None:
        """Two rules → both appear as separate bullet entries."""

        state = _make_state(
            rules=[
                "You prefer shorter responses.",
                "You've said meditation makes you more anxious.",
            ]
        )
        block = _format_procedural_rules_block(state)

        assert "- You prefer shorter responses." in block
        assert "- You've said meditation makes you more anxious." in block

    def test_block_starts_with_double_newline(self) -> None:
        """The block begins with ``\\n\\n`` so it concatenates cleanly.

        The builders do ``f"{knowledge}\\n\\n{instructions}{rules_block}"``
        — the rules block must provide its OWN leading whitespace so
        there's visible separation from the instructions block above.
        """

        state = _make_state(rules=["You prefer shorter responses."])
        block = _format_procedural_rules_block(state)
        assert block.startswith("\n\n")


# ─── _format_recall_toggle_constraint ──────────────────────────────────────


class TestRecallToggleHelper:
    """Direct tests of ``_format_recall_toggle_constraint``.

    Note: this helper ALWAYS returns a non-empty block, regardless of
    state. The constraint is always injected — the variant (off / on)
    depends on the toggle state. An empty block would mean the LLM
    had no memory-reference guidance at all, which isn't what we want.
    """

    def test_recall_off_emits_do_not_reference_constraint(self) -> None:
        """Default (False) → the 'do not explicitly reference' constraint."""

        state = _make_state(recall_enabled=False)
        block = _format_recall_toggle_constraint(state)

        assert "proactive recall: OFF" in block
        assert "do NOT explicitly reference past sessions" in block

    def test_recall_on_emits_relaxed_constraint(self) -> None:
        """True → the relaxed 'may reference sparingly' constraint."""

        state = _make_state(recall_enabled=True)
        block = _format_recall_toggle_constraint(state)

        assert "proactive recall: ON" in block
        assert "may reference relevant past memories" in block
        assert "sparingly" in block

    def test_missing_toggle_defaults_to_off(self) -> None:
        """Missing ``proactive_recall_enabled`` key → OFF variant.

        Mirrors the ``False`` default in the store layer. A user with
        no stored profile gets the same constraint as a user who has
        explicitly toggled recall off.
        """

        state: dict[str, Any] = {
            "message": "hello",
            "history": [],
            "working_memory": [],
            "session_memory": {
                "summary": "",
                "active_concerns": [],
                "open_loops": [],
                "current_goal": None,
            },
        }
        block = _format_recall_toggle_constraint(cast(AgentState, state))
        assert "proactive recall: OFF" in block

    def test_block_starts_with_double_newline(self) -> None:
        """Same ``\\n\\n`` leading-whitespace contract as the rules block."""

        state = _make_state(recall_enabled=False)
        block = _format_recall_toggle_constraint(state)
        assert block.startswith("\n\n")


# ─── Therapeutic builder integration ───────────────────────────────────────


class TestTherapeuticBuilderInjection:
    """Integration tests for the 6 therapeutic system-prompt builders.

    Each builder should weave procedural rules and the recall toggle
    into its output when state contains them, and fall back to the
    static knowledge+instructions content when state is empty.
    """

    BUILDERS = [
        ("supportive", build_supportive_system_prompt),
        ("reflective", build_reflective_system_prompt),
        ("clarifying", build_clarifying_system_prompt),
        ("psychoeducation", build_psychoeducation_system_prompt),
        ("closing", build_closing_system_prompt),
        ("guided_exercise", build_guided_exercise_system_prompt),
    ]

    def test_empty_state_produces_bare_static_prompt(self) -> None:
        """With no rules and no toggle, each builder returns its static
        content plus the default recall-OFF constraint.

        Note: the recall-toggle constraint is ALWAYS emitted (off is
        the default). Only the rules block is conditional.
        """

        state = _make_state()  # no rules, no toggle override
        for name, builder in self.BUILDERS:
            prompt = builder(state)
            # Knowledge file composition produces non-empty text
            assert len(prompt) > 0, f"{name}: empty prompt"
            # No rules block — "Style rules" substring must NOT appear
            assert "Style rules from past conversations" not in prompt, (
                f"{name}: rules block unexpectedly present"
            )
            # Recall-OFF constraint IS present (default)
            assert "proactive recall: OFF" in prompt, (
                f"{name}: recall-off constraint missing"
            )

    def test_rules_are_injected_when_present(self) -> None:
        """When ``procedural_profile.procedural_rules`` is populated, the rules
        block appears in all 6 builders' output."""

        rules = [
            "You prefer shorter responses.",
            "You've said meditation makes you more anxious.",
        ]
        state = _make_state(rules=rules)
        for name, builder in self.BUILDERS:
            prompt = builder(state)
            assert "Style rules from past conversations" in prompt, (
                f"{name}: rules header missing"
            )
            assert "- You prefer shorter responses." in prompt, (
                f"{name}: first rule missing"
            )
            assert "- You've said meditation makes you more anxious." in prompt, (
                f"{name}: second rule missing"
            )
            # The silent-follow guidance must be present so the LLM
            # knows not to quote the rules
            assert "Follow these rules silently" in prompt, (
                f"{name}: silent-follow guidance missing"
            )


def test_technique_prompt_requires_attuned_opening_before_structure() -> None:
    state = _make_state()
    state["therapeutic_approach"] = "cbt"

    prompt = build_technique_system_prompt(state)

    assert "Lead with a brief, attuned acknowledgment before any question" in prompt
    assert 'Do not open with bare consent ("Yes.", "Okay.", "Sure.")' in prompt


def test_closing_prompt_handles_wrap_up_takeaway_requests() -> None:
    """Closing should answer explicit wrap-up takeaway requests directly."""

    prompt = build_closing_system_prompt(_make_state())

    assert "Give one takeaway when asked" in prompt
    assert "summarize the main takeaway" in prompt
    assert "exactly ONE concise" in prompt
    assert "Do not ask for more" in prompt
    assert "start a new exercise" in prompt
    assert "reopen" in prompt
    assert "exploration" in prompt


def test_closing_prompt_handles_one_word_acknowledgments_quietly() -> None:
    """Closing should not turn terse acknowledgments into a sendoff."""

    prompt = build_closing_system_prompt(_make_state())

    assert 'one-word acknowledgment such as\n  "ok"' in prompt
    assert '"Okay." is enough' in prompt
    assert "Do not summarize the arc" in prompt
    assert "add an open-door sentence" in prompt
    assert "parental close" in prompt
    assert "One-word acknowledgments" in prompt


def test_supportive_prompt_handles_low_content_opening_orientation() -> None:
    """Supportive openings may orient lightly without becoming intake."""

    prompt = build_supportive_system_prompt(_make_state())

    assert "for low-content openings" in prompt
    assert "We don't need a plan" in prompt
    assert "is there something specific you want from this session" in prompt
    assert "do not use session-plan framing" in prompt
    assert "respond to that content first" in prompt
    assert "goals for the session" in prompt
    assert "Exception: for low-content session openings" in prompt
    assert "ask exactly one" in prompt
    assert 'a question ending in "?"' in prompt
    assert "Good low-content opening" in prompt
    assert "it does not actually ask the optional orientation question" in prompt


def test_supportive_prompt_breaks_uniform_response_shape() -> None:
    """Supportive mode should not force reflection -> explanation -> question."""

    prompt = build_supportive_system_prompt(_make_state())
    normalized = " ".join(prompt.split())

    assert "Vary reply shape across turns" in prompt
    assert "a paraphrase that stands alone" in normalized
    assert "a single question with no preamble" in normalized
    assert "If two consecutive replies followed" in normalized
    assert "drop one of the parts" in normalized
    assert "reflection -> explanation -> question" in prompt
    assert "Let a reflection stand on its own" in prompt
    assert "pairing a reflection with an explanation" in prompt


def test_supportive_prompt_handles_acknowledgments_and_capability_questions() -> None:
    """Supportive mode should answer terse acknowledgments and capability asks directly."""

    prompt = build_supportive_system_prompt(_make_state())

    assert 'one-word acknowledgment such as "ok"' in prompt
    assert 'Often "Okay." is enough' in prompt
    assert "Do not add a takeaway" in prompt
    assert "parental close" in prompt
    assert "specific cases below override" in prompt
    assert "what you can do for them" in prompt
    assert "answer as a stance" in prompt
    assert "not a feature list" in prompt
    assert "what you will be doing in the conversation" in prompt


def test_psychoeducation_prompt_handles_pop_neuro_practical_requests() -> None:
    """Pop-neuro shorthand should not trigger a lecture before practical help."""

    prompt = build_psychoeducation_system_prompt(_make_state())
    normalized = " ".join(prompt.split())

    assert "Pop-neuroscience shorthand" in prompt
    assert "I need dopamine" in prompt
    assert "answer the practical need first" in prompt
    assert "Do not open by correcting the neuroscience framing" in normalized
    assert "lecturing about brain chemistry" in prompt
    assert "What can I do to get dopamine?" in prompt
    assert "Stand up and step outside for two minutes" in prompt
    assert "That's it for now" in prompt


def test_dispatch_prompt_separates_technique_from_exercise_track_starts() -> None:
    prompt = build_therapeutic_dispatch_system_prompt()

    assert "NOT asking to start a named exercise track" in prompt
    assert (
        "those are guided_exercise turns because the agent should begin the "
        "matching stepwise exercise" in prompt
    )
    assert "can we figure out a way to test it" in prompt
    assert "can we look at what actually matters to me" in prompt
    assert "If the user names self-criticism AND explicitly asks to do " in prompt
    assert (
        "do NOT use technique just because the user wants to 'talk it through'"
        in prompt
    )
    assert "consolidating progress, naming strengths" in prompt
    assert (
        "I keep avoiding work tasks because I get anxious and start spiraling" in prompt
    )
    assert "can choose ACT as the therapeutic_approach" in prompt
    assert "pairs wrap-up language with a takeaway request" in prompt
    assert "before we wrap up, what's the main takeaway?" in prompt
    assert "what should I remember from this?" in prompt
    assert "A turn that says 'thanks, that helps'" in prompt

    def test_recall_on_switches_constraint_variant(self) -> None:
        """With ``proactive_recall_enabled=True``, the prompt contains
        the ON variant of the constraint instead of the OFF variant."""

        state = _make_state(recall_enabled=True)
        for name, builder in self.BUILDERS:
            prompt = builder(state)
            assert "proactive recall: ON" in prompt, (
                f"{name}: recall-on constraint missing"
            )
            assert "proactive recall: OFF" not in prompt, (
                f"{name}: stale recall-off constraint present"
            )

    def test_rules_appear_AFTER_instructions(self) -> None:
        """The rules block is a suffix: it appears AFTER the mode's
        instructions block, not before or in the middle.

        This matches the schema's ``injection_point: system_prompt_suffix``
        spec. Using a signature string from each mode's instructions
        block, we verify the rules block's position.
        """

        state = _make_state(
            rules=["You prefer shorter responses."],
        )

        # Each mode has a unique signature string in its instructions.
        # We verify the rules block appears AFTER it.
        signatures = {
            "supportive": "SUPPORTIVE mode",
            "reflective": "REFLECTIVE mode",
            "clarifying": "CLARIFYING mode",
            "psychoeducation": "PSYCHOEDUCATION mode",
            "closing": "CLOSING mode",
            "guided_exercise": "GUIDED_EXERCISE mode",
        }

        for name, builder in self.BUILDERS:
            prompt = builder(state)
            mode_sig = signatures[name]
            sig_index = prompt.find(mode_sig)
            rules_index = prompt.find("Style rules from past conversations")
            assert sig_index >= 0, f"{name}: mode signature not found"
            assert rules_index >= 0, f"{name}: rules block not found"
            assert rules_index > sig_index, (
                f"{name}: rules block ({rules_index}) appears BEFORE "
                f"instructions block ({sig_index})"
            )

    def test_recall_toggle_appears_AFTER_rules(self) -> None:
        """The recall-toggle constraint is the final block in the prompt.

        Order (top to bottom):
          1. Knowledge files
          2. Mode instructions
          3. Rules block (if rules exist)
          4. Recall-toggle constraint (always present)

        This test pins that order for the builders that have all three
        dynamic sections present.
        """

        state = _make_state(
            rules=["You prefer shorter responses."],
            recall_enabled=False,
        )
        for name, builder in self.BUILDERS:
            prompt = builder(state)
            rules_index = prompt.find("Style rules from past conversations")
            recall_index = prompt.find("proactive recall: OFF")
            assert rules_index >= 0, f"{name}: rules block missing"
            assert recall_index >= 0, f"{name}: recall block missing"
            assert recall_index > rules_index, (
                f"{name}: recall block ({recall_index}) appears BEFORE "
                f"rules block ({rules_index})"
            )


# ─── Crisis response builder: deliberate exception ────────────────────────


class TestCrisisBuilderExemption:
    """The crisis response builder is DELIBERATELY exempt from rules
    and recall-toggle injection. See the docstring of
    ``build_crisis_response_system_prompt`` for the safety rationale.

    These tests lock that exemption in place so a future drive-by
    refactor can't accidentally start injecting rules into crisis
    responses.
    """

    def test_crisis_builder_takes_no_arguments(self) -> None:
        """The builder's zero-arg signature is the first signal that it
        doesn't read state."""

        prompt = build_crisis_response_system_prompt()
        assert len(prompt) > 0

    def test_crisis_prompt_never_contains_rules_block(self) -> None:
        """Regression guard: the 'Style rules from past conversations'
        header must NEVER appear in the crisis response prompt,
        regardless of what's in state (which the builder doesn't
        read anyway).

        This test doesn't use state at all, because the builder
        doesn't take any. But the invariant is: if the builder's
        signature ever grows a ``state`` param AND the implementation
        starts reading procedural_rules, this assertion should still
        hold against any state configuration. The test is a pin on
        behavior, not a pin on signature.
        """

        prompt = build_crisis_response_system_prompt()
        assert "Style rules from past conversations" not in prompt
        assert "Follow these rules silently" not in prompt

    def test_crisis_prompt_never_contains_recall_toggle(self) -> None:
        """Regression guard: the recall-toggle constraint must NEVER
        appear in the crisis response prompt.

        Crisis responses should not cite "last session we talked
        about X" regardless of whether the user has proactive recall
        on or off. Omitting the constraint block keeps the crisis
        path free of memory-interaction etiquette that doesn't belong
        in a safety-critical response.
        """

        prompt = build_crisis_response_system_prompt()
        assert "proactive recall" not in prompt


class TestTherapeuticResponsePrompt:
    """Tests for the shared therapeutic user-prompt builder."""

    def test_formats_structured_working_memory_on_demand(self) -> None:
        """Raw working-memory dicts should be rendered at prompt time."""

        state = _make_state(
            working_memory=[
                {
                    "type": "semantic",
                    "evidence_quote": "I have a sister named Sarah.",
                },
                {
                    "type": "episodic",
                    "summary": "talked about grief after my dog died.",
                    "primary_themes": ["grief"],
                    "is_catch_up": True,
                },
            ]
        )

        prompt = build_therapeutic_response_prompt(state, mode="supportive")
        assert "Relevant context from past sessions:" in prompt
        assert "- Previously noted: I have a sister named Sarah." in prompt
        assert "- Last session (grief): talked about grief after my dog died." in prompt

    def test_legacy_string_working_memory_entries_render_unchanged(self) -> None:
        """Legacy ``str`` entries (from older checkpoints or manual fixtures)
        should pass through the formatter unchanged."""

        state = _make_state(
            working_memory=[
                "Previously noted: legacy fact about the user.",  # type: ignore[list-item]
            ]
        )
        prompt = build_therapeutic_response_prompt(state, mode="supportive")
        assert "legacy fact about the user" in prompt
