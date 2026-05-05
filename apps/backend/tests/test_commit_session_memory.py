"""Tests for the session-end memory promotion pass."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import cast

import pytest

from agent.memory.policy.candidates import (
    SessionMemoryBuffer,
    build_procedural_candidate,
    build_semantic_candidate,
)
from agent.memory.models import (
    EntityRef,
    ExtractionResult,
    MemoryWrite,
    MoodArc,
    ProceduralExtractionResult,
    ProceduralRuleDraft,
    SessionArc,
    SummarizationResult,
)
from agent.memory.episodic import (
    session_arc_to_stored as _session_arc_to_stored,
)
from agent.memory.modes import MemoryMode
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime.session_commit import run_commit_session_memory
from agent.persistence import PersistentAgentRuntime
from agent.state import AgentState
from services.llm.base import BaseLLMClient, StructuredResponseT


def _semantic_write(
    *,
    category: str = "trigger",
    predicate: str = "WORRIES_ABOUT",
    object_type: str = "Concern",
    object_identifier: str = "family conflict panic",
    evidence_quote: str = "Family conflict is a big trigger for panic.",
    source_turn_index: int = 0,
) -> MemoryWrite:
    return MemoryWrite(
        category=category,  # type: ignore[arg-type]
        subject=EntityRef(type="User", identifier="user-1"),
        predicate=predicate,  # type: ignore[arg-type]
        object=EntityRef(type=object_type, identifier=object_identifier),  # type: ignore[arg-type]
        evidence_quote=evidence_quote,
        confidence="high",
        source_session_id="thread-test",
        source_turn_index=source_turn_index,
    )


def _partial_state(
    *,
    user_id: str = "user-1",
    session_id: str = "thread-test",
    transcript: list[dict[str, str]] | None = None,
) -> AgentState:
    return cast(
        AgentState,
        {
            "user_id": user_id,
            "session_id": session_id,
            "transcript": transcript
            or [
                {
                    "role": "user",
                    "content": "Family conflict is a big trigger for panic.",
                }
            ],
        },
    )


def _stored_arc(
    *,
    summary: str = "User kept returning to panic after family conflict and wants help handling it.",
    primary_themes: list[str] | None = None,
    open_loops: list[str] | None = None,
) -> SessionArc:
    return _session_arc_to_stored(
        SessionArc(
            session_id="thread-test",
            started_at="2026-04-19T10:00:00Z",
            ended_at="2026-04-19T10:20:00Z",
            duration_seconds=1200,
            turn_count=4,
            primary_themes=primary_themes or ["family conflict", "panic"],
            summary=summary,
            mood_arc=MoodArc(opened="anxious", closed="steadier"),
            open_loops=open_loops or ["wants a plan for family conflict"],
            resolved_threads=[],
        ),
        owner_id="user-1",
    )


class _FakeSessionCommitLLM(BaseLLMClient):
    """Deterministic fake LLM for the runtime integration test."""

    def __init__(
        self,
        *,
        extraction_result: ExtractionResult,
        summarization_result: SummarizationResult,
        procedural_result: ProceduralExtractionResult | None = None,
    ) -> None:
        self.extraction_result = extraction_result
        self.summarization_result = summarization_result
        self.procedural_result = procedural_result or ProceduralExtractionResult(
            rules=[],
            reason="no procedural rules in session commit test",
        )

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "Let's slow it down and look at what happened."

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "fake"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__

        if schema_name == "CrisisAssessmentSchema":
            from agent.safety.service import CrisisAssessmentSchema

            return cast(
                StructuredResponseT,
                CrisisAssessmentSchema(
                    level=0,
                    confidence="high",
                    reason="safe — fake llm for session commit test",
                    needs_crisis_response=False,
                    needs_clarification=False,
                ),
            )

        if schema_name == "DispatchDecision":
            from agent.memory.models import DispatchDecision

            return cast(
                StructuredResponseT,
                DispatchDecision(
                    response_style="supportive",
                    reasoning="supportive mode for session commit test",
                    confidence="high",
                ),
            )

        if schema_name == "ExtractionResult":
            return cast(StructuredResponseT, self.extraction_result)

        if schema_name == "ProceduralExtractionResult":
            return cast(StructuredResponseT, self.procedural_result)

        if schema_name == "SummarizationResult":
            return cast(StructuredResponseT, self.summarization_result)

        raise RuntimeError(f"_FakeSessionCommitLLM: unexpected schema {schema_name}")


@pytest.mark.asyncio
async def test_supported_held_candidate_writes_at_session_end() -> None:
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(),
        message="Family conflict is a big trigger for panic.",
    )
    buffer = SessionMemoryBuffer(
        session_id="thread-test", semantic_candidates=[candidate]
    )

    result = await run_commit_session_memory(
        _partial_state(),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(),
    )

    assert result is not None
    assert result.semantic_writes == 1
    assert result.semantic_skips == 0
    assert await store.arecord_count(("user-1", "semantic")) == 1
    records = await store.asearch(("user-1", "semantic"), query=None)
    assert records[0].value["write_timing"] == "session_end"
    assert records[0].value["policy_version"] == "phase3_v1"


@pytest.mark.asyncio
async def test_session_end_correction_supersedes_stale_fact() -> None:
    store = OpenCouchMemoryStore()
    await store.aput(
        ("user-1", "semantic"),
        "fact-old",
        {
            "id": "fact-old",
            "category": "context",
            "subject": {"type": "User", "identifier": "user-1"},
            "predicate": "EXPERIENCED",
            "object": {"type": "Event", "identifier": "sister moved out"},
            "evidence_quote": "My sister moved out last month.",
            "confidence": "high",
            "source_session_id": "thread-old",
            "source_turn_index": 0,
            "created_at": "2026-04-18T10:00:00Z",
            "last_referenced_at": "2026-04-18T10:00:00Z",
            "dormant_at": None,
            "superseded_by": None,
            "user_visible": True,
        },
    )
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            predicate="EXPERIENCED",
            object_type="Event",
            object_identifier="sister moved back in",
            evidence_quote="Actually, my sister moved back in this week.",
        ),
        message="Actually, my sister moved back in this week.",
    )
    buffer = SessionMemoryBuffer(
        session_id="thread-test", semantic_candidates=[candidate]
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "Actually, my sister moved back in this week.",
                }
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(
            summary="User clarified that their sister moved back in this week.",
            primary_themes=["family", "home"],
            open_loops=[],
        ),
    )

    assert result is not None
    assert result.semantic_writes == 1
    records = await store.asearch(("user-1", "semantic"), query=None, limit=10)
    stale = next(record for record in records if record.key == "fact-old")
    fresh = next(record for record in records if record.key != "fact-old")
    assert stale.value["superseded_by"] == fresh.key
    assert stale.value["dormant_at"] is not None
    assert fresh.value["write_timing"] == "session_end"


@pytest.mark.asyncio
async def test_unsupported_held_candidate_stays_unwritten() -> None:
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(
            evidence_quote="Family conflict is a big trigger for panic.",
        ),
        message="Family conflict is a big trigger for panic.",
    )
    buffer = SessionMemoryBuffer(
        session_id="thread-test", semantic_candidates=[candidate]
    )

    result = await run_commit_session_memory(
        _partial_state(),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=None,
    )

    assert result is not None
    assert result.semantic_writes == 0
    assert result.semantic_skips == 1
    assert await store.arecord_count(("user-1", "semantic")) == 0


@pytest.mark.asyncio
async def test_repetition_candidate_commits_after_two_turns() -> None:
    store = OpenCouchMemoryStore()
    candidate_a = build_semantic_candidate(
        _semantic_write(
            category="context",
            object_identifier="making mistakes at work",
            evidence_quote="I always assume one mistake means I'm incompetent.",
            source_turn_index=0,
        ),
        message="I always assume one mistake means I'm incompetent.",
    )
    candidate_b = build_semantic_candidate(
        _semantic_write(
            category="context",
            object_identifier="making mistakes at work",
            evidence_quote="Every week I tell myself one mistake means I'm incompetent.",
            source_turn_index=1,
        ),
        message="Every week I tell myself one mistake means I'm incompetent.",
    )
    buffer = SessionMemoryBuffer(
        session_id="thread-test",
        semantic_candidates=[candidate_a, candidate_b],
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "I always assume one mistake means I'm incompetent.",
                },
                {
                    "role": "user",
                    "content": "Every week I tell myself one mistake means I'm incompetent.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=None,
    )

    assert result is not None
    assert result.semantic_writes == 1
    assert result.semantic_skips == 0
    assert await store.arecord_count(("user-1", "semantic")) == 1
    records = await store.asearch(("user-1", "semantic"), query=None)
    assert records[0].value["write_timing"] == "promotion"


@pytest.mark.asyncio
async def test_single_turn_negative_self_belief_does_not_promote_from_summary_alone() -> (
    None
):
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            object_identifier="making mistakes at work",
            evidence_quote="I always assume one mistake means I'm incompetent.",
            source_turn_index=0,
        ),
        message="I always assume one mistake means I'm incompetent.",
    )
    buffer = SessionMemoryBuffer(
        session_id="thread-test", semantic_candidates=[candidate]
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "I always assume one mistake means I'm incompetent.",
                }
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(
            summary="User kept returning to fears that one mistake means they are incompetent.",
            primary_themes=["work stress", "shame"],
        ),
    )

    assert result is not None
    assert result.semantic_writes == 0
    assert result.semantic_skips == 1
    assert await store.arecord_count(("user-1", "semantic")) == 0


@pytest.mark.asyncio
async def test_cross_session_repetition_promotes_negative_self_belief() -> None:
    store = OpenCouchMemoryStore()
    prior_arc = _session_arc_to_stored(
        SessionArc(
            session_id="thread-prior",
            started_at="2026-04-18T10:00:00Z",
            ended_at="2026-04-18T10:20:00Z",
            duration_seconds=1200,
            turn_count=4,
            primary_themes=["work stress", "shame"],
            summary=(
                "User described a recurring fear that one mistake at work "
                "means they are incompetent."
            ),
            mood_arc=MoodArc(opened="anxious", closed="drained"),
            open_loops=["keeps interpreting mistakes as proof of incompetence"],
            resolved_threads=[],
        ),
        owner_id="user-1",
    )
    await store.aput(
        ("user-1", "episodic"),
        key=prior_arc.id,
        value=prior_arc.model_dump(mode="json"),
    )

    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            object_identifier="making mistakes at work",
            evidence_quote="I always assume one mistake means I'm incompetent.",
            source_turn_index=0,
        ),
        message="I always assume one mistake means I'm incompetent.",
    )
    buffer = SessionMemoryBuffer(
        session_id="thread-test", semantic_candidates=[candidate]
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "I always assume one mistake means I'm incompetent.",
                }
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=None,
    )

    assert result is not None
    assert result.semantic_writes == 1
    assert result.semantic_skips == 0
    records = await store.asearch(("user-1", "semantic"), query=None)
    assert len(records) == 1
    assert records[0].value["write_timing"] == "promotion"


@pytest.mark.asyncio
async def test_repeated_implicit_procedural_preference_promotes_at_session_end() -> (
    None
):
    store = OpenCouchMemoryStore()
    candidate_a = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You've said meditation makes you more anxious.",
            evidence=["Meditation makes me more anxious."],
        ),
        message="Meditation makes me more anxious.",
        session_id="thread-test",
        turn_index=0,
    )
    candidate_b = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You've said meditation makes you more anxious.",
            evidence=["I think meditation makes me more anxious every time."],
        ),
        message="I think meditation makes me more anxious every time.",
        session_id="thread-test",
        turn_index=1,
    )
    buffer = SessionMemoryBuffer(
        session_id="thread-test",
        procedural_candidates=[candidate_a, candidate_b],
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "Meditation makes me more anxious.",
                },
                {
                    "role": "user",
                    "content": "I think meditation makes me more anxious every time.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=None,
    )

    assert result is not None
    assert result.procedural_writes == 1
    assert result.procedural_skips == 0
    profile_record = await store.aget(("user-1", "procedural"), "user_response_style")
    assert profile_record is not None
    stored_rule = profile_record.value["rules"][0]
    assert stored_rule["write_timing"] == "promotion"
    assert stored_rule["policy_version"] == "phase3_v1"
    assert stored_rule["source"] == "consolidation"


@pytest.mark.asyncio
async def test_runtime_end_session_commits_buffered_semantic_candidates(
    tmp_path: Path,
) -> None:
    store = OpenCouchMemoryStore()
    fake = _FakeSessionCommitLLM(
        extraction_result=ExtractionResult(
            facts=[_semantic_write()],
            reason="captured trigger candidate",
        ),
        summarization_result=SummarizationResult(
            arc=SessionArc(
                session_id="thread-test",
                started_at="2026-04-19T10:00:00Z",
                ended_at="2026-04-19T10:20:00Z",
                duration_seconds=1200,
                turn_count=3,
                primary_themes=["family conflict", "panic"],
                summary="User discussed panic triggered by family conflict and wanted help handling it.",
                mood_arc=MoodArc(opened="anxious", closed="steadier"),
                open_loops=["wants a plan for family conflict"],
                resolved_threads=[],
            ),
            reason="captured central family conflict trigger arc",
        ),
    )

    async with PersistentAgentRuntime(
        sqlite_path=tmp_path / "threads.sqlite3",
        memory_store=store,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-test",
            message="Family conflict is a big trigger for panic.",
            user_id="user-1",
            llm_client=fake,
        )

        assert await store.arecord_count(("user-1", "semantic")) == 0
        assert (
            len(
                runtime._session_tracker.session_memory_buffers[
                    "thread-test"
                ].semantic_candidates
            )
            == 1
        )

        stored_arc = await runtime.end_session("thread-test", llm_client=fake)

        assert stored_arc is not None
        assert await store.arecord_count(("user-1", "episodic")) == 1
        assert await store.arecord_count(("user-1", "semantic")) == 1
        assert "thread-test" not in runtime._session_tracker.session_memory_buffers


@pytest.mark.asyncio
async def test_runtime_end_session_promotes_repeated_implicit_procedural_preference(
    tmp_path: Path,
) -> None:
    store = OpenCouchMemoryStore()
    fake = _FakeSessionCommitLLM(
        extraction_result=ExtractionResult(
            facts=[],
            reason="no semantic facts for procedural promotion test",
        ),
        procedural_result=ProceduralExtractionResult(
            rules=[
                ProceduralRuleDraft(
                    rule="You've said meditation makes you more anxious.",
                    evidence=["Meditation makes me more anxious."],
                )
            ],
            reason="implicit procedural preference candidate",
        ),
        summarization_result=SummarizationResult(
            arc=None,
            reason="session too thin for episodic summary",
        ),
    )

    async with PersistentAgentRuntime(
        sqlite_path=tmp_path / "threads.sqlite3",
        memory_store=store,
        memory_mode=MemoryMode.LOCAL,
    ) as runtime:
        await runtime.run_turn(
            thread_id="thread-test",
            message="Meditation makes me more anxious.",
            user_id="user-1",
            llm_client=fake,
        )
        await runtime.run_turn(
            thread_id="thread-test",
            message="I think meditation makes me more anxious every time.",
            user_id="user-1",
            llm_client=fake,
        )

        assert await store.arecord_count(("user-1", "procedural")) == 0
        assert (
            len(
                runtime._session_tracker.session_memory_buffers[
                    "thread-test"
                ].procedural_candidates
            )
            == 2
        )

        stored_arc = await runtime.end_session("thread-test", llm_client=fake)

        assert stored_arc is None
        profile_record = await store.aget(
            ("user-1", "procedural"), "user_response_style"
        )
        assert profile_record is not None
        assert len(profile_record.value["rules"]) == 1
        assert profile_record.value["rules"][0]["write_timing"] == "promotion"
        assert "thread-test" not in runtime._session_tracker.session_memory_buffers
