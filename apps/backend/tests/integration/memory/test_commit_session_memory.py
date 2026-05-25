"""Tests for the session-end memory promotion pass."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, cast

import pytest

import agent.memory.semantic_writes as semantic_writes
import agent.memory.session_commit_service as session_commit_service
from agent.memory.policy.candidates import (
    PolicyDecision,
    ProceduralCandidate,
    SemanticCandidate,
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
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.flows.therapeutic import merge_therapeutic_tool_results
from agent.memory.store import OpenCouchMemoryStore
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.session import run_commit_session_memory
from agent.runtime.session.history import session_conversation_from_transcript
from agent.runtime import PersistentAgentRuntime
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from llm.base import BaseLLMClient, StructuredResponseT


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


def _held_semantic_buffer(
    *candidates: SemanticCandidate,
    hold_action: Literal[
        "commit_at_session_end",
        "require_repetition",
    ] = "commit_at_session_end",
) -> SessionMemoryBuffer:
    buffer = SessionMemoryBuffer(session_id="thread-test")
    for candidate in candidates:
        buffer.hold_semantic(
            candidate,
            PolicyDecision(
                action=hold_action,
                reason=f"test policy held semantic candidate as {hold_action}",
                policy_version="test_policy_v1",
            ),
        )
    return buffer


def _held_procedural_buffer(
    *candidates: ProceduralCandidate,
) -> SessionMemoryBuffer:
    buffer = SessionMemoryBuffer(session_id="thread-test")
    for candidate in candidates:
        buffer.hold_procedural(
            candidate,
            PolicyDecision(
                action="commit_at_session_end",
                reason="test policy held procedural candidate for session end",
                policy_version="test_policy_v1",
            ),
        )
    return buffer


class _FakeSessionCommitLLM(BaseLLMClient):
    """Deterministic fake LLM for the runtime integration test."""

    def __init__(
        self,
        *,
        extraction_result: ExtractionResult,
        summarization_result: SummarizationResult,
        procedural_result: ProceduralExtractionResult | None = None,
        semantic_reconciliation_decision: dict[str, Any] | None = None,
    ) -> None:
        self.extraction_result = extraction_result
        self.summarization_result = summarization_result
        self.procedural_result = procedural_result or ProceduralExtractionResult(
            rules=[],
            reason="no procedural rules in session commit test",
        )
        self.semantic_reconciliation_decision = semantic_reconciliation_decision or {
            "action": "coexist",
            "record_indexes": [],
            "reason": "test semantic reconciliation keeps candidates separate",
            "confidence": "high",
        }

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
            from agent.guardrails.service import CrisisAssessmentSchema

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
                    exercise_start_basis="ambiguous_or_none",
                    reasoning="supportive mode for session commit test",
                    confidence="high",
                ),
            )

        if schema_name == "TurnDispatchDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                route="therapeutic",
                active_flow_action="none",
                reasoning="ordinary session commit test turn",
                confidence="high",
            )

        if schema_name == "ExtractionResult":
            return cast(StructuredResponseT, self.extraction_result)

        if schema_name == "ProceduralExtractionResult":
            return cast(StructuredResponseT, self.procedural_result)

        if schema_name == "SummarizationResult":
            return cast(StructuredResponseT, self.summarization_result)

        if schema_name == "SemanticWritePolicyDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                action="commit_now",
                reason="test semantic candidate can follow policy clamp",
                confidence="high",
            )

        if schema_name == "ProceduralWritePolicyDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                action="commit_at_session_end",
                reason="test implicit procedural preference should wait",
                confidence="high",
            )

        if schema_name == "SemanticReconciliationDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                **self.semantic_reconciliation_decision,
            )

        if schema_name == "ProceduralReconciliationDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                action="append",
                replace_indexes=[],
                reason="test procedural reconciliation appends",
                confidence="high",
            )

        raise RuntimeError(f"_FakeSessionCommitLLM: unexpected schema {schema_name}")


@pytest.mark.asyncio
async def test_commit_scoring_uses_explicit_session_conversation() -> None:
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(),
        message="Family conflict is a big trigger for panic.",
    )
    buffer = _held_semantic_buffer(candidate)
    conversation = session_conversation_from_transcript(
        [
            {
                "role": "user",
                "content": "Family conflict is a big trigger for panic.",
            },
            {"role": "assistant", "content": "That sounds intense."},
        ]
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "stale state transcript without supporting evidence",
                }
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(),
        conversation=conversation,
    )

    assert result is not None
    assert result.semantic_writes == 1
    assert result.semantic_skips == 0


@pytest.mark.asyncio
async def test_privacy_override_drops_held_candidates_at_session_end() -> None:
    store = OpenCouchMemoryStore()
    semantic = build_semantic_candidate(
        _semantic_write(),
        message="Family conflict is a big trigger for panic.",
    )
    procedural = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You've said step-by-step plans help most.",
            evidence=["short step-by-step plans help me most"],
        ),
        message="short step-by-step plans help me most",
        session_id="thread-test",
        turn_index=0,
    )
    buffer = _held_semantic_buffer(semantic)
    buffer.hold_procedural(
        procedural,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test buffers procedural candidate before privacy override",
            policy_version="test_policy_v1",
        ),
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "Family conflict is a big trigger for panic.",
                },
                {
                    "role": "user",
                    "content": "Actually, do not save or remember any of that.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(),
    )

    assert result is not None
    assert result.semantic_writes == 0
    assert result.procedural_writes == 0
    assert result.semantic_skips == 1
    assert result.procedural_skips == 1
    assert await store.arecord_count(("user-1", "semantic")) == 0
    assert await store.arecord_count(("user-1", "procedural")) == 0


@pytest.mark.asyncio
async def test_non_command_memory_language_does_not_drop_held_candidates() -> None:
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(),
        message="Family conflict is a big trigger for panic.",
    )
    buffer = _held_semantic_buffer(candidate)

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "Family conflict is a big trigger for panic.",
                },
                {
                    "role": "user",
                    "content": "I don't remember the details of that argument.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(),
    )

    assert result is not None
    assert result.semantic_writes == 1
    assert result.semantic_skips == 0
    assert await store.arecord_count(("user-1", "semantic")) == 1


@pytest.mark.asyncio
async def test_supported_held_candidate_writes_at_session_end() -> None:
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(),
        message="Family conflict is a big trigger for panic.",
    )
    buffer = _held_semantic_buffer(candidate)

    result = await run_commit_session_memory(
        _partial_state(),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(),
    )

    assert result is not None
    assert result.semantic_writes == 1
    assert result.semantic_skips == 0
    assert result.semantic_failures == 0
    assert result.procedural_failures == 0
    assert result.support_load_failed is False
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
    buffer = _held_semantic_buffer(candidate)

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
        llm_client=_FakeSessionCommitLLM(
            extraction_result=ExtractionResult(facts=[], reason="unused"),
            summarization_result=SummarizationResult(arc=None, reason="unused"),
            semantic_reconciliation_decision={
                "action": "supersede",
                "record_indexes": [0],
                "reason": "new session-end fact corrects the older living situation",
                "confidence": "high",
            },
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
    buffer = _held_semantic_buffer(candidate)

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
    buffer = _held_semantic_buffer(
        candidate_a,
        candidate_b,
        hold_action="require_repetition",
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
async def test_similar_repetition_candidates_promote_once() -> None:
    store = OpenCouchMemoryStore()
    candidate_a = build_semantic_candidate(
        _semantic_write(
            category="context",
            predicate="WORRIES_ABOUT",
            object_identifier="belief that one mistake means incompetence",
            evidence_quote="I always assume one mistake means I am incompetent.",
            source_turn_index=0,
        ),
        message="I always assume one mistake means I am incompetent.",
    )
    candidate_b = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            predicate="EXPERIENCED",
            object_identifier="one small mistake leading to self-labeling as incompetent",
            evidence_quote=(
                "This comes up every week: one small mistake and I tell myself "
                "I am incompetent."
            ),
            source_turn_index=1,
        ),
        message=(
            "This comes up every week: one small mistake and I tell myself "
            "I am incompetent."
        ),
    )
    buffer = _held_semantic_buffer(
        candidate_a,
        candidate_b,
        hold_action="require_repetition",
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "I always assume one mistake means I am incompetent.",
                },
                {
                    "role": "user",
                    "content": (
                        "This comes up every week: one small mistake and I tell "
                        "myself I am incompetent."
                    ),
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
    records = await store.asearch(("user-1", "semantic"), query=None)
    assert len(records) == 1
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
    buffer = _held_semantic_buffer(candidate, hold_action="require_repetition")

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
async def test_repetition_candidate_does_not_treat_current_arc_as_prior_support() -> (
    None
):
    store = OpenCouchMemoryStore()
    stored_arc = _stored_arc(
        summary=(
            "User described a fear that one mistake at work means they are incompetent."
        ),
        primary_themes=["work stress", "shame"],
        open_loops=["one mistake can become a global self-judgment"],
    )
    await store.aput(
        ("user-1", "episodic"),
        key=stored_arc.id,
        value=stored_arc.model_dump(mode="json"),
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
    buffer = _held_semantic_buffer(candidate, hold_action="require_repetition")

    result = await run_commit_session_memory(
        _partial_state(
            session_id="runtime-session-id",
            transcript=[
                {
                    "role": "user",
                    "content": "I always assume one mistake means I'm incompetent.",
                }
            ],
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=stored_arc,
    )

    assert result is not None
    assert result.semantic_writes == 0
    assert result.semantic_skips == 1
    assert await store.arecord_count(("user-1", "semantic")) == 0


@pytest.mark.asyncio
async def test_session_end_self_belief_candidate_still_requires_repetition() -> None:
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            predicate="EXPERIENCED",
            object_identifier="one mistake triggers incompetence belief",
            evidence_quote="I always assume one mistake means I am incompetent.",
            source_turn_index=0,
        ),
        message="I always assume one mistake means I am incompetent.",
    )
    buffer = _held_semantic_buffer(candidate, hold_action="commit_at_session_end")

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "I always assume one mistake means I am incompetent.",
                }
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(
            summary=(
                "User described a fear that one mistake means they are incompetent."
            ),
            primary_themes=["self-judgment"],
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
    buffer = _held_semantic_buffer(candidate, hold_action="require_repetition")

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
async def test_semantic_candidate_yields_to_overlapping_procedural_preference() -> None:
    store = OpenCouchMemoryStore()
    semantic_candidate = build_semantic_candidate(
        _semantic_write(
            category="coping_strategy",
            predicate="USES",
            object_type="CopingStrategy",
            object_identifier="short step-by-step plans",
            evidence_quote="Short step-by-step plans help when I am anxious about presentations.",
            source_turn_index=0,
        ),
        message="Short step-by-step plans help when I am anxious about presentations.",
    )
    procedural_candidate_a = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Remember to keep plans very short when the user is anxious about presentations.",
            evidence=[
                "Please keep plans very short when I am anxious about presentations."
            ],
        ),
        message="Please keep plans very short when I am anxious about presentations.",
        session_id="thread-test",
        turn_index=0,
    )
    procedural_candidate_b = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Remember to keep plans very short when the user is anxious about presentations.",
            evidence=[
                "Short step-by-step plans help when I am anxious about presentations."
            ],
        ),
        message="Short step-by-step plans help when I am anxious about presentations.",
        session_id="thread-test",
        turn_index=1,
    )
    buffer = _held_semantic_buffer(semantic_candidate)
    buffer.hold_procedural(
        procedural_candidate_a,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test policy held overlapping procedural preference",
            policy_version="test_policy_v1",
        ),
    )
    buffer.hold_procedural(
        procedural_candidate_b,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test policy held repeated overlapping procedural preference",
            policy_version="test_policy_v1",
        ),
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "Please keep plans very short when I am anxious about presentations.",
                },
                {
                    "role": "user",
                    "content": "Short step-by-step plans help when I am anxious about presentations.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(
            summary="User said short step-by-step plans help with presentation anxiety.",
            primary_themes=["presentation anxiety", "short plans"],
            open_loops=["wants brief planning support"],
        ),
    )

    assert result is not None
    assert result.semantic_writes == 0
    assert result.semantic_skips == 1
    assert result.procedural_writes == 1
    assert await store.arecord_count(("user-1", "semantic")) == 0
    profile_record = await store.aget(("user-1", "procedural"), "user_response_style")
    assert profile_record is not None
    assert len(profile_record.value["rules"]) == 1


@pytest.mark.asyncio
async def test_low_lexical_overlap_semantic_guidance_still_yields_to_procedural() -> (
    None
):
    store = OpenCouchMemoryStore()
    semantic_candidate = build_semantic_candidate(
        _semantic_write(
            category="coping_strategy",
            predicate="USES",
            object_type="CopingStrategy",
            object_identifier="tiny chunks for presentation prep",
            evidence_quote="Tiny chunks make presentation prep feel manageable for me.",
            source_turn_index=0,
        ),
        message="Tiny chunks make presentation prep feel manageable for me.",
    )
    procedural_candidate_a = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Keep presentation prep bite-sized when the user feels overwhelmed.",
            evidence=[
                "Please keep presentation prep bite-sized when I feel overwhelmed."
            ],
        ),
        message="Please keep presentation prep bite-sized when I feel overwhelmed.",
        session_id="thread-test",
        turn_index=0,
    )
    procedural_candidate_b = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Keep presentation prep bite-sized when the user feels overwhelmed.",
            evidence=["Tiny chunks make presentation prep feel manageable for me."],
        ),
        message="Tiny chunks make presentation prep feel manageable for me.",
        session_id="thread-test",
        turn_index=1,
    )
    buffer = _held_semantic_buffer(semantic_candidate)
    buffer.hold_procedural(
        procedural_candidate_a,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test policy held overlapping bite-sized preference",
            policy_version="test_policy_v1",
        ),
    )
    buffer.hold_procedural(
        procedural_candidate_b,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test policy held repeated overlapping bite-sized preference",
            policy_version="test_policy_v1",
        ),
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "Please keep presentation prep bite-sized when I feel overwhelmed.",
                },
                {
                    "role": "user",
                    "content": "Tiny chunks make presentation prep feel manageable for me.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(
            summary="User said bite-sized presentation prep feels more manageable.",
            primary_themes=["presentation prep", "manageable planning"],
            open_loops=["wants small-step prep support"],
        ),
    )

    assert result is not None
    assert result.semantic_writes == 0
    assert result.semantic_skips == 1
    assert result.procedural_writes == 1
    assert await store.arecord_count(("user-1", "semantic")) == 0


@pytest.mark.asyncio
async def test_distinct_semantic_and_procedural_memories_both_survive() -> None:
    store = OpenCouchMemoryStore()
    semantic_candidate = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            predicate="WORRIES_ABOUT",
            object_type="Concern",
            object_identifier="family conflict panic",
            evidence_quote="Family conflict is a big trigger for panic.",
            source_turn_index=0,
        ),
        message="Family conflict is a big trigger for panic.",
    )
    procedural_candidate_a = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer direct responses when panic spikes.",
            evidence=["From now on, be direct with me when panic spikes."],
        ),
        message="From now on, be direct with me when panic spikes.",
        session_id="thread-test",
        turn_index=0,
    )
    procedural_candidate_b = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer direct responses when panic spikes.",
            evidence=["Please be direct with me when panic spikes."],
        ),
        message="Please be direct with me when panic spikes.",
        session_id="thread-test",
        turn_index=1,
    )
    buffer = _held_semantic_buffer(semantic_candidate)
    buffer.hold_procedural(
        procedural_candidate_a,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test policy held direct-response preference",
            policy_version="test_policy_v1",
        ),
    )
    buffer.hold_procedural(
        procedural_candidate_b,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test policy held repeated direct-response preference",
            policy_version="test_policy_v1",
        ),
    )

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "Family conflict is a big trigger for panic.",
                },
                {
                    "role": "user",
                    "content": "From now on, be direct with me when panic spikes.",
                },
                {
                    "role": "user",
                    "content": "Please be direct with me when panic spikes.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(
            summary="User discussed panic triggered by family conflict and asked for direct responses when panic spikes.",
            primary_themes=["family conflict", "panic"],
            open_loops=["wants direct support during panic"],
        ),
    )

    assert result is not None
    assert result.semantic_writes == 1
    assert result.procedural_writes == 1
    assert await store.arecord_count(("user-1", "semantic")) == 1
    profile_record = await store.aget(("user-1", "procedural"), "user_response_style")
    assert profile_record is not None
    assert len(profile_record.value["rules"]) == 1


@pytest.mark.asyncio
async def test_semantic_clustering_merges_near_paraphrases() -> None:
    store = OpenCouchMemoryStore()
    candidate_a = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            predicate="WORRIES_ABOUT",
            object_type="Concern",
            object_identifier="presentation panic",
            evidence_quote="I feel panic when I have to do presentations.",
            source_turn_index=0,
        ),
        message="I feel panic when I have to do presentations.",
    )
    candidate_b = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            predicate="WORRIES_ABOUT",
            object_type="Concern",
            object_identifier="presentation panic",
            evidence_quote="Doing a presentation makes me feel very panicked.",
            source_turn_index=1,
        ),
        message="Doing a presentation makes me feel very panicked.",
    )
    buffer = _held_semantic_buffer(candidate_a, candidate_b)

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {
                    "role": "user",
                    "content": "I feel panic when I have to do presentations.",
                },
                {
                    "role": "user",
                    "content": "Doing a presentation makes me feel very panicked.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(
            summary="Presentations repeatedly trigger panic for the user.",
            primary_themes=["presentations", "panic"],
            open_loops=["wants presentation support"],
        ),
    )

    assert result is not None
    assert result.semantic_writes == 1
    assert await store.arecord_count(("user-1", "semantic")) == 1


@pytest.mark.asyncio
async def test_semantic_clustering_preserves_distinct_entities() -> None:
    store = OpenCouchMemoryStore()
    candidate_a = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            predicate="WORRIES_ABOUT",
            object_type="Concern",
            object_identifier="boss anxiety",
            evidence_quote="My boss makes me anxious.",
            source_turn_index=0,
        ),
        message="My boss makes me anxious.",
    )
    candidate_b = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            predicate="WORRIES_ABOUT",
            object_type="Concern",
            object_identifier="sister anxiety",
            evidence_quote="My sister makes me anxious.",
            source_turn_index=1,
        ),
        message="My sister makes me anxious.",
    )
    buffer = _held_semantic_buffer(candidate_a, candidate_b)

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {"role": "user", "content": "My boss makes me anxious."},
                {"role": "user", "content": "My sister makes me anxious."},
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=_stored_arc(
            summary="The user feels anxious around both their boss and their sister.",
            primary_themes=["boss anxiety", "family anxiety"],
            open_loops=[],
        ),
        llm_client=_FakeSessionCommitLLM(
            extraction_result=ExtractionResult(facts=[], reason="unused"),
            summarization_result=SummarizationResult(arc=None, reason="unused"),
            semantic_reconciliation_decision={
                "action": "coexist",
                "record_indexes": [],
                "reason": "boss and sister anxiety are distinct semantic memories",
                "confidence": "high",
            },
        ),
    )

    assert result is not None
    assert result.semantic_writes == 2
    assert await store.arecord_count(("user-1", "semantic")) == 2


@pytest.mark.asyncio
async def test_procedural_clustering_merges_paraphrased_preferences() -> None:
    store = OpenCouchMemoryStore()
    candidate_a = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Be more supportive during talks.",
            evidence=["Be more supportive during talks."],
        ),
        message="Be more supportive during talks.",
        session_id="thread-test",
        turn_index=0,
    )
    candidate_b = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Support me more when I am talking in front of people.",
            evidence=["Support me more when I am talking in front of people."],
        ),
        message="Support me more when I am talking in front of people.",
        session_id="thread-test",
        turn_index=1,
    )
    buffer = _held_procedural_buffer(candidate_a, candidate_b)

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {"role": "user", "content": "Be more supportive during talks."},
                {
                    "role": "user",
                    "content": "Support me more when I am talking in front of people.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=None,
    )

    assert result is not None
    assert result.procedural_writes == 1
    profile_record = await store.aget(("user-1", "procedural"), "user_response_style")
    assert profile_record is not None
    assert len(profile_record.value["rules"]) == 1


@pytest.mark.asyncio
async def test_procedural_clustering_preserves_distinct_preferences() -> None:
    store = OpenCouchMemoryStore()
    candidate_a = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Be direct when panic spikes.",
            evidence=["Be direct when panic spikes."],
        ),
        message="Be direct when panic spikes.",
        session_id="thread-test",
        turn_index=0,
    )
    candidate_b = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Keep plans bite-sized when I am overwhelmed.",
            evidence=["Keep plans bite-sized when I am overwhelmed."],
        ),
        message="Keep plans bite-sized when I am overwhelmed.",
        session_id="thread-test",
        turn_index=1,
    )
    candidate_c = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Be direct when panic spikes.",
            evidence=["Please be direct when panic spikes."],
        ),
        message="Please be direct when panic spikes.",
        session_id="thread-test",
        turn_index=2,
    )
    candidate_d = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Keep plans bite-sized when I am overwhelmed.",
            evidence=["Please keep plans bite-sized when I am overwhelmed."],
        ),
        message="Please keep plans bite-sized when I am overwhelmed.",
        session_id="thread-test",
        turn_index=3,
    )
    buffer = _held_procedural_buffer(candidate_a, candidate_b, candidate_c, candidate_d)

    result = await run_commit_session_memory(
        _partial_state(
            transcript=[
                {"role": "user", "content": "Be direct when panic spikes."},
                {
                    "role": "user",
                    "content": "Keep plans bite-sized when I am overwhelmed.",
                },
                {"role": "user", "content": "Please be direct when panic spikes."},
                {
                    "role": "user",
                    "content": "Please keep plans bite-sized when I am overwhelmed.",
                },
            ]
        ),
        memory_store=store,
        session_buffer=buffer,
        stored_arc=None,
    )

    assert result is not None
    assert result.procedural_writes == 2
    profile_record = await store.aget(("user-1", "procedural"), "user_response_style")
    assert profile_record is not None
    assert len(profile_record.value["rules"]) == 2


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
    buffer = _held_procedural_buffer(candidate_a, candidate_b)

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
    assert result.procedural_failures == 0
    assert result.semantic_failures == 0
    assert result.support_load_failed is False
    profile_record = await store.aget(("user-1", "procedural"), "user_response_style")
    assert profile_record is not None
    stored_rule = profile_record.value["rules"][0]
    assert stored_rule["write_timing"] == "promotion"
    assert stored_rule["policy_version"] == "phase3_v1"
    assert stored_rule["source"] == "consolidation"


@pytest.mark.asyncio
async def test_commit_session_memory_marks_prior_support_load_failures() -> None:
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(),
        message="Family conflict is a big trigger for panic.",
    )
    buffer = _held_semantic_buffer(candidate)

    async def _raise_prior_support_failure(
        *args: object, **kwargs: object
    ) -> list[str]:
        raise RuntimeError("forced prior support failure")

    original_loader = session_commit_service._load_prior_session_support_texts
    session_commit_service._load_prior_session_support_texts = (
        _raise_prior_support_failure
    )
    try:
        result = await run_commit_session_memory(
            _partial_state(),
            memory_store=store,
            session_buffer=buffer,
            stored_arc=_stored_arc(),
        )
    finally:
        session_commit_service._load_prior_session_support_texts = original_loader

    assert result is not None
    assert result.support_load_failed is True
    assert result.semantic_writes == 1
    assert result.semantic_failures == 0
    assert result.procedural_failures == 0
    assert await store.arecord_count(("user-1", "semantic")) == 1


@pytest.mark.asyncio
async def test_commit_session_memory_marks_semantic_fetch_failures() -> None:
    store = OpenCouchMemoryStore()
    candidate = build_semantic_candidate(
        _semantic_write(),
        message="Family conflict is a big trigger for panic.",
    )
    buffer = _held_semantic_buffer(candidate)

    async def _raise_fetch_failure(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("forced semantic fetch failure")

    original_fetch = semantic_writes.fetch_existing_semantic_records
    semantic_writes.fetch_existing_semantic_records = _raise_fetch_failure
    try:
        result = await run_commit_session_memory(
            _partial_state(),
            memory_store=store,
            session_buffer=buffer,
            stored_arc=_stored_arc(),
        )
    finally:
        semantic_writes.fetch_existing_semantic_records = original_fetch

    assert result is not None
    assert result.semantic_writes == 0
    assert result.semantic_skips == 0
    assert result.semantic_failures == 1
    assert result.procedural_failures == 0
    assert result.support_load_failed is False
    assert await store.arecord_count(("user-1", "semantic")) == 0


@pytest.mark.asyncio
async def test_commit_session_memory_marks_procedural_upsert_failures() -> None:
    store = OpenCouchMemoryStore()
    candidate_a = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Meditation makes me more anxious.",
            evidence=["Meditation makes me more anxious."],
        ),
        message="Meditation makes me more anxious.",
        session_id="thread-test",
        turn_index=0,
    )
    candidate_b = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="I think meditation makes me more anxious every time.",
            evidence=["I think meditation makes me more anxious every time."],
        ),
        message="I think meditation makes me more anxious every time.",
        session_id="thread-test",
        turn_index=1,
    )
    buffer = _held_procedural_buffer(candidate_a, candidate_b)

    async def _raise_upsert_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced procedural upsert failure")

    original_upsert = session_commit_service.aupsert_procedural_rule
    session_commit_service.aupsert_procedural_rule = _raise_upsert_failure
    try:
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
    finally:
        session_commit_service.aupsert_procedural_rule = original_upsert

    assert result is not None
    assert result.procedural_writes == 0
    assert result.procedural_skips == 0
    assert result.procedural_failures == 1
    assert result.semantic_failures == 0
    profile_record = await store.aget(("user-1", "procedural"), "user_response_style")
    assert profile_record is None


def test_privacy_request_clears_session_buffer() -> None:
    semantic = build_semantic_candidate(
        _semantic_write(),
        message="My sister is visiting next week.",
    )
    procedural = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You've said meditation makes you more anxious.",
            evidence=["Meditation makes me more anxious."],
        ),
        message="Meditation makes me more anxious.",
        session_id="thread-test",
        turn_index=1,
    )
    session_memory = SessionMemoryBuffer(session_id="thread-test")
    session_memory.hold_semantic(
        semantic,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test buffers semantic candidate before privacy override",
            policy_version="test_policy_v1",
        ),
    )
    session_memory.hold_procedural(
        procedural,
        PolicyDecision(
            action="commit_at_session_end",
            reason="test buffers procedural candidate before privacy override",
            policy_version="test_policy_v1",
        ),
    )
    state = _partial_state()
    state["session_memory"] = session_memory.model_dump(mode="json")

    context = OpenAITextRunContext(
        thread_id="thread-test",
        workflow_context=WorkflowContext(
            llm_client=None,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        ),
        current_user_message="Actually, don't save this.",
        user_id="user-1",
        session_id="thread-test",
    )
    context.record_memory_tool_result(
        action_type="forget_by_query",
        response_text="I won't save that.",
        memory_control={"pending_action": None},
        clear_session_buffer=True,
        side_effect="delete_memory",
        retry_safe=True,
    )

    merge_therapeutic_tool_results(
        state,
        run_context=context,
        response_text="I won't save that.",
    )

    cleared = SessionMemoryBuffer.model_validate(state["session_memory"])
    assert cleared.held_semantic_candidates == []
    assert cleared.held_procedural_candidates == []


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

        runtime._session_memory_buffer_for_thread("thread-test").hold_semantic(
            build_semantic_candidate(
                _semantic_write(),
                message="Family conflict is a big trigger for panic.",
            ),
            PolicyDecision(
                action="commit_at_session_end",
                reason="test buffers semantic candidate for session end",
                policy_version="test_policy_v1",
            ),
        )
        await runtime._persist_runtime_session_tracking("thread-test")

        assert await store.arecord_count(("user-1", "semantic")) == 0
        assert (
            len(
                runtime._session_tracker.session_memory_buffers[
                    "thread-test"
                ].held_semantic_candidates
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
        runtime._session_memory_buffer_for_thread("thread-test").hold_procedural(
            build_procedural_candidate(
                ProceduralRuleDraft(
                    rule="You've said meditation makes you more anxious.",
                    evidence=["Meditation makes me more anxious."],
                ),
                message="Meditation makes me more anxious.",
                session_id="thread-test",
                turn_index=0,
            ),
            PolicyDecision(
                action="commit_at_session_end",
                reason="test buffers implicit procedural preference",
                policy_version="test_policy_v1",
            ),
        )
        await runtime._persist_runtime_session_tracking("thread-test")
        await runtime.run_turn(
            thread_id="thread-test",
            message="I think meditation makes me more anxious every time.",
            user_id="user-1",
            llm_client=fake,
        )
        runtime._session_memory_buffer_for_thread("thread-test").hold_procedural(
            build_procedural_candidate(
                ProceduralRuleDraft(
                    rule="You've said meditation makes you more anxious.",
                    evidence=["I think meditation makes me more anxious every time."],
                ),
                message="I think meditation makes me more anxious every time.",
                session_id="thread-test",
                turn_index=1,
            ),
            PolicyDecision(
                action="commit_at_session_end",
                reason="test buffers repeated implicit procedural preference",
                policy_version="test_policy_v1",
            ),
        )
        await runtime._persist_runtime_session_tracking("thread-test")

        assert await store.arecord_count(("user-1", "procedural")) == 0
        assert (
            len(
                runtime._session_tracker.session_memory_buffers[
                    "thread-test"
                ].held_procedural_candidates
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
