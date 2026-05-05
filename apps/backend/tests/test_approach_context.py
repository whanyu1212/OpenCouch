"""Tests for the approach-context pipeline: schema, working memory, and retrieval.

These tests verify the full Option D data path:
1. Schema: typed approach-context models round-trip through JSON.
2. Working memory: EpisodicWorkingMemoryEntry carries and formats approach context.
3. Load memory: _episodic_entry_from_record extracts approach fields from stored records.
4. SessionMemoryBuffer: approach_counts accumulates per-turn and computes dominant.
5. Missing approach fields degrade gracefully.
"""

from __future__ import annotations

from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.models import (
    ACTContext,
    CBTContext,
    DBTContext,
    GriefContext,
    IPTContext,
    MIContext,
    PFAContext,
    SessionArc,
)
from agent.working_memory import (
    format_working_memory_entry,
    make_episodic_working_memory_entry,
)


# ─── 1. Schema round-trip tests ─────────────────────────────────────────


class TestApproachContextSchema:
    """Approach context models round-trip cleanly through JSON."""

    def test_cbt_context_round_trip(self) -> None:
        ctx = CBTContext(
            thought_examined="I'm going to get fired",
            action_step="speak up in one meeting",
            tool_used="thought_record",
        )
        dumped = ctx.model_dump(mode="json")
        assert dumped["approach"] == "cbt"
        reloaded = CBTContext.model_validate(dumped)
        assert reloaded.thought_examined == "I'm going to get fired"

    def test_mi_context_round_trip(self) -> None:
        ctx = MIContext(
            readiness_stage="contemplation",
            change_talk_themes=["health", "kids"],
            sustain_talk_themes=["effort"],
        )
        dumped = ctx.model_dump(mode="json")
        assert dumped["approach"] == "motivational_interviewing"
        reloaded = MIContext.model_validate(dumped)
        assert reloaded.readiness_stage == "contemplation"
        assert reloaded.change_talk_themes == ["health", "kids"]

    def test_act_context_round_trip(self) -> None:
        ctx = ACTContext(
            values_identified=["family", "honesty"],
            committed_action="call my sister this week",
        )
        dumped = ctx.model_dump(mode="json")
        reloaded = ACTContext.model_validate(dumped)
        assert reloaded.values_identified == ["family", "honesty"]
        assert reloaded.committed_action == "call my sister this week"

    def test_grief_context_round_trip(self) -> None:
        ctx = GriefContext(person_lost="Mom", relationship="mother")
        dumped = ctx.model_dump(mode="json")
        reloaded = GriefContext.model_validate(dumped)
        assert reloaded.person_lost == "Mom"
        assert reloaded.time_since_loss is None

    def test_ipt_context_round_trip(self) -> None:
        ctx = IPTContext(
            problem_area="role_transition",
            key_relationship="partner",
            communication_step_planned="tell them I need space",
        )
        dumped = ctx.model_dump(mode="json")
        reloaded = IPTContext.model_validate(dumped)
        assert reloaded.problem_area == "role_transition"

    def test_dbt_context_round_trip(self) -> None:
        ctx = DBTContext(
            skills_used=["TIPP", "opposite action"], primary_domain="distress_tolerance"
        )
        dumped = ctx.model_dump(mode="json")
        reloaded = DBTContext.model_validate(dumped)
        assert reloaded.skills_used == ["TIPP", "opposite action"]

    def test_pfa_context_round_trip(self) -> None:
        ctx = PFAContext(
            crisis_type="panic attack", support_connected="crisis text line"
        )
        dumped = ctx.model_dump(mode="json")
        reloaded = PFAContext.model_validate(dumped)
        assert reloaded.crisis_type == "panic attack"

    def test_session_arc_with_approach_context_round_trips(self) -> None:
        """Full SessionArc → JSON → SessionArc with discriminated union."""

        arc = SessionArc(
            session_id="test",
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T01:00:00Z",
            duration_seconds=3600,
            turn_count=10,
            primary_themes=["work stress"],
            summary="User examined a thought about competence.",
            mood_arc={"opened": "anxious", "closed": "calmer"},
            approach_used="cbt",
            approach_context=CBTContext(
                thought_examined="I'm incompetent",
                action_step="ask for feedback",
            ),
        )
        dumped = arc.model_dump(mode="json")
        reloaded = SessionArc.model_validate(dumped)
        assert isinstance(reloaded.approach_context, CBTContext)
        assert reloaded.approach_context.thought_examined == "I'm incompetent"

    def test_session_arc_without_approach_fields(self) -> None:
        """Old-style arcs without approach fields should validate cleanly."""

        old = {
            "session_id": "old",
            "started_at": "2025-01-01T00:00:00Z",
            "ended_at": "2025-01-01T01:00:00Z",
            "duration_seconds": 3600,
            "turn_count": 5,
            "primary_themes": ["sleep"],
            "summary": "Old session about sleep.",
            "mood_arc": {"opened": "tired", "closed": "tired"},
        }
        arc = SessionArc.model_validate(old)
        assert arc.approach_used is None
        assert arc.approach_context is None


# ─── 2. Working memory formatting tests ──────────────────────────────────


class TestWorkingMemoryApproachFormatting:
    """format_working_memory_entry renders approach context correctly."""

    def test_cbt_entry_shows_thought_and_action(self) -> None:
        entry = make_episodic_working_memory_entry(
            summary="Examined a thought about work.",
            primary_themes=["work stress"],
            is_catch_up=True,
            approach_used="cbt",
            approach_context={
                "approach": "cbt",
                "thought_examined": "I'll get fired",
                "action_step": "speak up in one meeting",
                "tool_used": "thought_record",
            },
        )
        rendered = format_working_memory_entry(entry)
        assert "CBT" in rendered
        assert "Thought: I'll get fired" in rendered
        assert "Action step: speak up in one meeting" in rendered
        assert "Tool: thought_record" in rendered

    def test_entry_without_approach_renders_unchanged(self) -> None:
        entry = make_episodic_working_memory_entry(
            summary="User discussed sleep.",
            primary_themes=["sleep"],
            is_catch_up=False,
        )
        rendered = format_working_memory_entry(entry)
        assert rendered == "Last session (sleep): User discussed sleep."
        assert "[" not in rendered

    def test_entry_with_approach_but_no_context_has_no_bracket_suffix(self) -> None:
        entry = make_episodic_working_memory_entry(
            summary="Short MI session.",
            primary_themes=["motivation"],
            is_catch_up=False,
            approach_used="motivational_interviewing",
        )
        rendered = format_working_memory_entry(entry)
        assert "MOTIVATIONAL INTERVIEWING" in rendered
        assert "[" not in rendered

    def test_partial_context_skips_null_fields(self) -> None:
        entry = make_episodic_working_memory_entry(
            summary="Grief session.",
            primary_themes=["grief"],
            is_catch_up=True,
            approach_used="grief_support",
            approach_context={
                "approach": "grief_support",
                "person_lost": "Dad",
                "relationship": None,
                "time_since_loss": None,
            },
        )
        rendered = format_working_memory_entry(entry)
        assert "Person lost: Dad" in rendered
        assert "Relationship" not in rendered
        assert "Time since loss" not in rendered

    def test_list_fields_render_as_comma_separated(self) -> None:
        entry = make_episodic_working_memory_entry(
            summary="MI session.",
            primary_themes=["ambivalence"],
            is_catch_up=False,
            approach_used="motivational_interviewing",
            approach_context={
                "approach": "motivational_interviewing",
                "readiness_stage": "contemplation",
                "change_talk_themes": ["health", "kids", "energy"],
                "sustain_talk_themes": [],
            },
        )
        rendered = format_working_memory_entry(entry)
        assert "Change talk: health, kids, energy" in rendered
        assert "Sustain talk" not in rendered  # empty list skipped


# ─── 3. _episodic_entry_from_record passthrough ─────────────────────────


class TestEpisodicEntryFromRecordApproach:
    """_episodic_entry_from_record passes approach fields through."""

    def test_new_record_with_approach(self) -> None:
        # Simulate what a stored record looks like after model_dump
        record = {
            "summary": "User examined a thought.",
            "primary_themes": ["work"],
            "approach_used": "cbt",
            "approach_context": {
                "approach": "cbt",
                "thought_examined": "I always fail",
                "action_step": None,
                "tool_used": None,
            },
        }
        entry = make_episodic_working_memory_entry(
            summary=record["summary"],
            primary_themes=record.get("primary_themes") or [],
            is_catch_up=True,
            approach_used=record.get("approach_used"),
            approach_context=record.get("approach_context"),
        )
        assert entry["approach_used"] == "cbt"
        assert entry["approach_context"]["thought_examined"] == "I always fail"

    def test_old_record_without_approach(self) -> None:
        record = {
            "summary": "Old session.",
            "primary_themes": ["sleep"],
        }
        entry = make_episodic_working_memory_entry(
            summary=record["summary"],
            primary_themes=record.get("primary_themes") or [],
            is_catch_up=False,
            approach_used=record.get("approach_used"),
            approach_context=record.get("approach_context"),
        )
        assert "approach_used" not in entry
        assert "approach_context" not in entry


# ─── 4. SessionMemoryBuffer approach counting ───────────────────────────


class TestSessionMemoryBufferApproach:
    """approach_counts accumulates per-turn and dominant_approach works."""

    def test_empty_buffer_returns_none(self) -> None:
        buf = SessionMemoryBuffer(session_id="test")
        assert buf.dominant_approach() is None

    def test_single_approach_dominates(self) -> None:
        buf = SessionMemoryBuffer(session_id="test")
        buf.record_approach("cbt")
        buf.record_approach("cbt")
        buf.record_approach("cbt")
        assert buf.dominant_approach() == "cbt"

    def test_none_and_none_string_are_filtered(self) -> None:
        buf = SessionMemoryBuffer(session_id="test")
        buf.record_approach("cbt")
        buf.record_approach(None)
        buf.record_approach("none")
        buf.record_approach(None)
        assert buf.approach_counts == {"cbt": 1}
        assert buf.dominant_approach() == "cbt"

    def test_mixed_approaches_picks_most_frequent(self) -> None:
        buf = SessionMemoryBuffer(session_id="test")
        buf.record_approach("act")
        buf.record_approach("act")
        buf.record_approach("act")
        buf.record_approach("cbt")
        buf.record_approach("cbt")
        assert buf.dominant_approach() == "act"

    def test_json_round_trip_preserves_counts(self) -> None:
        buf = SessionMemoryBuffer(session_id="test")
        buf.record_approach("grief_support")
        buf.record_approach("grief_support")
        dumped = buf.model_dump(mode="json")
        reloaded = SessionMemoryBuffer.model_validate(dumped)
        assert reloaded.dominant_approach() == "grief_support"

    def test_backward_compat_without_approach_counts(self) -> None:
        """Old persisted buffers without approach_counts should load cleanly."""
        old = {
            "session_id": "old",
            "semantic_candidates": [],
            "procedural_candidates": [],
        }
        buf = SessionMemoryBuffer.model_validate(old)
        assert buf.approach_counts == {}
        assert buf.dominant_approach() is None
