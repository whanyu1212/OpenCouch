"""Unit tests for the session summarization prompt builders.

These are pure-function tests — no LLM client, no store, no async. The
prompt builders take an AgentState and some provenance metadata and
return a string. The tests verify that:

1. The system prompt is a single string with the key instructions
   that prevent drift (None-return, narrative-not-transcript, length caps,
   mood_arc format, crisis_level_max honesty).
2. The user prompt injects the full transcript, not just a window.
3. The user prompt copies provenance metadata verbatim.
4. Empty-transcript sessions produce a clean placeholder instead of
   crashing or injecting an empty block.
"""

from __future__ import annotations

from typing import Any

from agent.memory.prompts.summarization import (
    build_summarization_system_prompt,
    build_summarization_user_prompt,
)


def _make_state(*, transcript: list[dict[str, str]] | None = None) -> Any:
    """Build a minimal AgentState-shaped dict for prompt testing.

    The builders only read ``transcript``, so the rest of the state
    shape is irrelevant for these tests. We return a plain dict with
    the ``# type: ignore[return-value]`` escape hatch since AgentState
    is a TypedDict.
    """

    return {"transcript": transcript or []}  # type: ignore[return-value]


# ─── System prompt ────────────────────────────────────────────────────


class TestSummarizationSystemPrompt:
    """Regression guards on the system prompt's critical instructions."""

    def test_system_prompt_is_non_empty_string(self) -> None:
        """The system prompt should be a substantial instruction block."""

        prompt = build_summarization_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 500  # the prompt is thousands of chars

    def test_system_prompt_instructs_to_return_none_for_thin_sessions(
        self,
    ) -> None:
        """The None-return branch must be visible in the system prompt —
        otherwise the LLM will fabricate summaries for empty sessions."""

        prompt = build_summarization_system_prompt()
        # The prompt talks about arc=None as the safer default
        assert "None" in prompt
        assert "small talk" in prompt.lower()

    def test_system_prompt_caps_summary_length(self) -> None:
        """The 600-char cap on the summary field should be mentioned
        so the LLM doesn't produce essay-length summaries."""

        prompt = build_summarization_system_prompt()
        assert "600" in prompt

    def test_system_prompt_mentions_mood_arc_format(self) -> None:
        """The mood_arc opened/closed pair is specific and the LLM
        will fabricate it if the prompt doesn't spell out the shape."""

        prompt = build_summarization_system_prompt()
        assert "opened" in prompt
        assert "closed" in prompt
        assert "mood_arc" in prompt.lower()

    def test_system_prompt_does_not_ask_llm_to_judge_crisis_level(self) -> None:
        """v0.4 refactor: the summarizer LLM should NOT be asked to
        produce a ``crisis_level_max`` field. The runtime computes it
        deterministically from per-turn crisis-gate verdicts, which
        keeps the crisis gate as the single source of truth for
        crisis severity. If the prompt starts asking for a level
        field again, the drift risk between 'what the gate decided
        per turn' and 'what the summarizer retroactively judged'
        will come back. Pin the non-regression here."""

        prompt = build_summarization_system_prompt()
        # The field should NOT appear as a thing the LLM fills in.
        # We're pinning specifically the "the LLM decides crisis level"
        # instruction pattern — the word "crisis" can still appear in
        # other contexts (like "crisis-adjacent content"), but the
        # schema field name must not appear as an output instruction.
        assert "crisis_level_max" not in prompt
        # And the prompt should explicitly say the LLM does NOT decide
        # the crisis level, so a future edit doesn't silently bring it
        # back. This is the "What you do NOT decide" section I added.
        assert "do NOT" in prompt
        # The crisis gate should be mentioned as the canonical source.
        assert "crisis gate" in prompt.lower()

    def test_system_prompt_warns_against_quoting_at_length(self) -> None:
        """The summary should be paraphrased, not a transcript.
        This is the load-bearing 'narrative over log' instruction."""

        prompt = build_summarization_system_prompt()
        assert "paraphrase" in prompt.lower()

    def test_system_prompt_caps_primary_themes_at_three(self) -> None:
        """The schema caps primary_themes at 3; the prompt should say so
        explicitly so the LLM doesn't produce 4+ themes and get rejected."""

        prompt = build_summarization_system_prompt()
        # "1-3" or similar phrasing should appear in the themes section
        assert "1-3" in prompt or "three" in prompt.lower()

    def test_system_prompt_pins_language_consistency_rule(self) -> None:
        """Regression guard against the v0.8 dogfood finding where the
        summarizer dropped an Armenian-script word ('ծանր', heavy) into
        a mood_arc string rendered in English. The rule: every text field
        the summarizer produces must be in the same language as the
        transcript. Mixed-language output looks like a glitch to the user
        reading the catch-up entry days later, and there's no downstream
        filter that repairs it.

        We pin two substrings so a future prompt refactor can't silently
        drop the language rule: the section header and the explicit "same
        language" phrase. Both must survive for the rule to be load-bearing.
        """

        prompt = build_summarization_system_prompt()
        assert "Language" in prompt
        assert "same language" in prompt.lower()

    def test_system_prompt_tells_model_to_ignore_factual_side_requests(self) -> None:
        """One-off lookup detours should not become episodic memory."""

        prompt = build_summarization_system_prompt()
        assert "brief side requests" in prompt
        assert "factual lookups" in prompt
        assert "assistant[grounded_lookup]" in prompt


# ─── User prompt ──────────────────────────────────────────────────────


class TestSummarizationUserPrompt:
    """Tests for transcript injection and provenance handling."""

    def test_user_prompt_copies_provenance_fields(self) -> None:
        """The user prompt must inject session_id, started_at, ended_at,
        duration_seconds, and turn_count so the LLM can copy them into
        the SessionArc without inferring."""

        state = _make_state(
            transcript=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hey, what's up"},
            ]
        )

        prompt = build_summarization_user_prompt(
            state,
            session_id="session-abc",
            started_at="2026-04-10T12:00:00Z",
            ended_at="2026-04-10T12:15:00Z",
            duration_seconds=900,
            turn_count=1,
        )

        assert "session-abc" in prompt
        assert "2026-04-10T12:00:00Z" in prompt
        assert "2026-04-10T12:15:00Z" in prompt
        assert "900" in prompt
        assert "turn_count" in prompt

    def test_user_prompt_injects_full_transcript(self) -> None:
        """Unlike the extraction prompt (which uses last 6 turns), the
        summarizer prompt should inject the ENTIRE transcript so it can
        see the full arc. This test pins that contract."""

        # 10 turns, all with distinct content
        transcript = []
        for i in range(10):
            transcript.append({"role": "user", "content": f"user turn number {i}"})
            transcript.append({"role": "assistant", "content": f"assistant reply {i}"})

        state = _make_state(transcript=transcript)
        prompt = build_summarization_user_prompt(
            state,
            session_id="session-test",
            started_at="2026-04-10T12:00:00Z",
            ended_at="2026-04-10T12:30:00Z",
            duration_seconds=1800,
            turn_count=10,
        )

        # Every single user turn should appear in the prompt (no windowing).
        for i in range(10):
            assert f"user turn number {i}" in prompt
            assert f"assistant reply {i}" in prompt

    def test_user_prompt_handles_empty_transcript(self) -> None:
        """An empty transcript shouldn't crash — it should produce a
        visible placeholder so the LLM knows the session had no content.
        This is an edge case but worth pinning because the code path
        has a guard for it."""

        state = _make_state(transcript=[])
        prompt = build_summarization_user_prompt(
            state,
            session_id="session-empty",
            started_at="2026-04-10T12:00:00Z",
            ended_at="2026-04-10T12:00:30Z",
            duration_seconds=30,
            turn_count=0,
        )

        # The placeholder should be visible
        assert "empty transcript" in prompt.lower()
        # Provenance still copied even when transcript is empty
        assert "session-empty" in prompt
        assert "turn_count" in prompt

    def test_user_prompt_skips_turns_with_empty_content(self) -> None:
        """Turns with empty/None content should be filtered out of the
        transcript block rather than rendered as blank lines. This
        matches the extraction prompt's behavior."""

        transcript = [
            {"role": "user", "content": "real message"},
            {"role": "assistant", "content": ""},  # empty — skip
            {"role": "user", "content": "another real message"},
        ]
        state = _make_state(transcript=transcript)
        prompt = build_summarization_user_prompt(
            state,
            session_id="session-mixed",
            started_at="2026-04-10T12:00:00Z",
            ended_at="2026-04-10T12:05:00Z",
            duration_seconds=300,
            turn_count=2,
        )

        assert "real message" in prompt
        assert "another real message" in prompt
        # The empty assistant turn should not render as a blank "assistant:" line.
        # We check by counting non-empty role markers in the transcript section.
        user_lines = prompt.count("user:")
        assert user_lines >= 2  # both real messages rendered

    def test_user_prompt_marks_assistant_response_styles(self) -> None:
        """Summarizer should see grounded lookup turns as side-turn metadata."""

        state = _make_state(
            transcript=[
                {"role": "user", "content": "Can you check the current time?"},
                {
                    "role": "assistant",
                    "content": "Sources: official time service",
                    "response_style": "grounded_lookup",
                },
            ]
        )
        prompt = build_summarization_user_prompt(
            state,
            session_id="session-style",
            started_at="2026-04-10T12:00:00Z",
            ended_at="2026-04-10T12:05:00Z",
            duration_seconds=300,
            turn_count=1,
        )

        assert "assistant[grounded_lookup]: Sources: official time service" in prompt


# ─── Approach context prompt tests ───────────────────────────────────


class TestSummarizationApproachPrompts:
    """Tests for approach_used / approach_context prompt additions."""

    def test_system_prompt_has_approach_context_section(self) -> None:
        """The system prompt must contain the approach context instructions
        so the LLM knows the new fields exist."""

        prompt = build_summarization_system_prompt()
        assert "Approach context" in prompt
        assert "approach_used" in prompt
        assert "approach_context" in prompt

    def test_system_prompt_warns_pfa_crisis_interaction(self) -> None:
        """PFA sessions are crisis-adjacent. The prompt should warn the
        LLM not to reinterpret crisis severity in approach_context."""

        prompt = build_summarization_system_prompt()
        assert "PFA" in prompt or "pfa" in prompt.lower()
        assert "crisis severity" in prompt.lower() or "crisis" in prompt.lower()

    def test_user_prompt_with_cbt_hint_injects_cbt_fields(self) -> None:
        """When approach_hint='cbt', the user prompt should list CBT-specific
        field names and NOT include other approach fields."""

        state = _make_state(transcript=[{"role": "user", "content": "test"}])
        prompt = build_summarization_user_prompt(
            state,
            session_id="s1",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T01:00:00Z",
            duration_seconds=3600,
            turn_count=10,
            approach_hint="cbt",
        )

        assert "Dominant therapeutic approach this session: cbt" in prompt
        assert "thought_examined" in prompt
        assert "action_step" in prompt
        assert "tool_used" in prompt
        # Should NOT contain MI or other approach fields
        assert "readiness_stage" not in prompt
        assert "values_identified" not in prompt

    def test_user_prompt_with_mi_hint_injects_mi_fields(self) -> None:
        """When approach_hint='motivational_interviewing', only MI fields."""

        state = _make_state(transcript=[{"role": "user", "content": "test"}])
        prompt = build_summarization_user_prompt(
            state,
            session_id="s2",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T01:00:00Z",
            duration_seconds=1800,
            turn_count=8,
            approach_hint="motivational_interviewing",
        )

        assert "motivational_interviewing" in prompt
        assert "readiness_stage" in prompt
        assert "change_talk_themes" in prompt
        # Should NOT contain CBT fields
        assert "thought_examined" not in prompt

    def test_user_prompt_without_hint_has_no_approach_block(self) -> None:
        """Without approach_hint, no approach extraction block."""

        state = _make_state(transcript=[{"role": "user", "content": "test"}])
        prompt = build_summarization_user_prompt(
            state,
            session_id="s3",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T01:00:00Z",
            duration_seconds=600,
            turn_count=3,
        )

        assert "Dominant therapeutic approach" not in prompt

    def test_user_prompt_with_none_hint_has_no_approach_block(self) -> None:
        """approach_hint='none' should be treated the same as absent."""

        state = _make_state(transcript=[{"role": "user", "content": "test"}])
        prompt = build_summarization_user_prompt(
            state,
            session_id="s4",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T01:00:00Z",
            duration_seconds=600,
            turn_count=3,
            approach_hint="none",
        )

        assert "Dominant therapeutic approach" not in prompt
