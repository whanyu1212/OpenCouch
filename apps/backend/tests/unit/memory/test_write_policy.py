"""Unit tests for the LLM-primary phase-1 memory write policy."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.memory.models import EntityRef, MemoryWrite, ProceduralRuleDraft
from agent.memory.policy.candidates import (
    PolicyDecision,
    SessionMemoryBuffer,
    build_procedural_candidate,
    build_semantic_candidate,
)
from agent.memory.policy.write import (
    decide_procedural_candidate_llm_primary,
    decide_semantic_candidate_llm_primary,
    semantic_hard_policy_guard,
    should_commit_implicit_procedural_preference,
    should_commit_pattern,
)
from llm.base import BaseLLMClient, StructuredResponseT


class _FakePolicyLLM(BaseLLMClient):
    """Fake structured client for write-policy classifier tests."""

    def __init__(self, decision: dict[str, Any]) -> None:
        self.decision = decision
        self.structured_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "unused"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        self.structured_calls += 1
        return cast(StructuredResponseT, response_schema(**self.decision))


class _FailingPolicyLLM(_FakePolicyLLM):
    """Fake policy client that raises from structured generation."""

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        self.structured_calls += 1
        raise RuntimeError("policy LLM failed")


def _semantic_write(
    *,
    category: str,
    predicate: str = "WORRIES_ABOUT",
    object_type: str = "Concern",
    object_identifier: str = "work stress",
    evidence_quote: str,
) -> MemoryWrite:
    return MemoryWrite(
        category=category,  # type: ignore[arg-type]
        subject=EntityRef(type="User", identifier="user-1"),
        predicate=predicate,  # type: ignore[arg-type]
        object=EntityRef(type=object_type, identifier=object_identifier),  # type: ignore[arg-type]
        evidence_quote=evidence_quote,
        confidence="high",
        source_session_id="session-1",
        source_turn_index=2,
    )


def test_explicit_stable_semantic_fact_has_no_hard_guard() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="relationship",
            predicate="KNOWS",
            object_type="Person",
            object_identifier="Sarah",
            evidence_quote="My sister Sarah lives nearby.",
        ),
        message="My sister Sarah lives nearby.",
    )

    decision = semantic_hard_policy_guard(candidate)

    assert decision is None


@pytest.mark.asyncio
async def test_high_sensitivity_semantic_fact_is_clamped_to_session_end() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            object_identifier="panic in family conflict",
            evidence_quote="Family conflict is a big trigger for panic.",
        ),
        message="Family conflict is a big trigger for panic.",
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "model considered it durable",
            "confidence": "high",
        }
    )

    decision = await decide_semantic_candidate_llm_primary(
        candidate,
        llm_client=llm,
    )

    assert decision.action == "commit_at_session_end"
    assert decision.policy_version == "phase1_v1"
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_negative_self_belief_can_be_held_for_repetition_by_llm_policy() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            object_identifier="making mistakes at work",
            evidence_quote="I always assume one mistake means I'm incompetent.",
        ),
        message="I always assume one mistake means I'm incompetent.",
    )
    llm = _FakePolicyLLM(
        {
            "action": "require_repetition",
            "reason": "negative self-belief needs repeated evidence",
            "confidence": "high",
        }
    )

    decision = await decide_semantic_candidate_llm_primary(candidate, llm_client=llm)

    assert decision.action == "require_repetition"
    assert decision.policy_version == "phase1_llm_v1"
    assert llm.structured_calls == 1
    assert (
        should_commit_pattern(hold_action="require_repetition", evidence_count=1)
        is False
    )
    assert (
        should_commit_pattern(hold_action="require_repetition", evidence_count=2)
        is True
    )


@pytest.mark.asyncio
async def test_negative_self_belief_policy_is_clamped_to_repetition() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            predicate="EXPERIENCED",
            object_identifier="one mistake triggers incompetence belief",
            evidence_quote="I always assume one mistake means I am incompetent.",
        ),
        message="I always assume one mistake means I am incompetent.",
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_at_session_end",
            "reason": "model treated it as therapeutic trigger context",
            "confidence": "high",
        }
    )

    decision = await decide_semantic_candidate_llm_primary(candidate, llm_client=llm)

    assert decision.action == "require_repetition"
    assert decision.policy_version == "phase1_v1"
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_turn_scoped_semantic_candidate_is_clamped_to_drop() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="coping_strategy",
            predicate="USES",
            object_type="CopingStrategy",
            object_identifier="slowing thoughts down",
            evidence_quote="Can we slow that thought down for now?",
        ),
        message="Can we slow that thought down for now?",
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "model treated it as a reusable coping strategy",
            "confidence": "high",
        }
    )

    decision = await decide_semantic_candidate_llm_primary(candidate, llm_client=llm)

    assert decision.action == "drop"
    assert decision.policy_version == "phase1_v1"
    assert llm.structured_calls == 1


def test_provenance_semantic_predicate_drops() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            predicate="MENTIONED_IN",
            object_type="Person",
            object_identifier="Sarah",
            evidence_quote="My sister Sarah lives nearby.",
        ),
        message="My sister Sarah lives nearby.",
    )

    decision = semantic_hard_policy_guard(candidate)

    assert decision is not None
    assert decision.action == "drop"


@pytest.mark.asyncio
async def test_implicit_procedural_preference_can_be_held_by_llm_policy() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You've said meditation makes you more anxious.",
            evidence=["Meditation makes me more anxious."],
        ),
        message="Meditation makes me more anxious.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_at_session_end",
            "reason": "implicit preference needs more evidence",
            "confidence": "high",
        }
    )

    decision = await decide_procedural_candidate_llm_primary(
        candidate,
        llm_client=llm,
    )

    assert decision.action == "commit_at_session_end"
    assert decision.policy_version == "phase1_llm_v1"
    assert llm.structured_calls == 1
    assert (
        should_commit_implicit_procedural_preference(
            hold_action="commit_at_session_end",
            evidence_count=1,
        )
        is False
    )
    assert (
        should_commit_implicit_procedural_preference(
            hold_action="commit_at_session_end",
            evidence_count=2,
        )
        is True
    )


@pytest.mark.asyncio
async def test_turn_scoped_procedural_request_uses_llm_policy() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer shorter responses.",
            evidence=["For this reply, keep it short."],
        ),
        message="For this reply, keep it short.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "drop",
            "reason": "turn-scoped request should not become durable memory",
            "confidence": "high",
        }
    )

    decision = await decide_procedural_candidate_llm_primary(candidate, llm_client=llm)

    assert decision.action == "drop"
    assert decision.policy_version == "phase1_llm_v1"
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_turn_scoped_procedural_request_is_clamped_to_drop() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Slow down and stay with the user's thought when they ask.",
            evidence=["Can we slow that thought down for now?"],
        ),
        message="Can we slow that thought down for now?",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "model overgeneralized a turn-scoped request",
            "confidence": "high",
        }
    )

    decision = await decide_procedural_candidate_llm_primary(
        candidate,
        llm_client=llm,
    )

    assert decision.action == "drop"
    assert decision.policy_version == "phase1_v1"
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_safety_conflict_clamps_unsafe_procedural_policy() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Don't ask me if I'm safe.",
            evidence=["Don't ask if I'm safe."],
        ),
        message="Don't ask if I'm safe.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "model incorrectly treated the request as a durable preference",
            "confidence": "high",
            "safety_conflict": True,
        }
    )

    decision = await decide_procedural_candidate_llm_primary(candidate, llm_client=llm)

    assert decision.action == "drop"
    assert decision.policy_version == "phase1_v1"
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_llm_semantic_policy_can_require_repetition_without_marker() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            predicate="EXPERIENCED",
            object_type="Concern",
            object_identifier="ruining friendships",
            evidence_quote="I ruin every friendship eventually.",
        ),
        message="I ruin every friendship eventually.",
    )
    llm = _FakePolicyLLM(
        {
            "action": "require_repetition",
            "reason": "global negative self-belief should need repetition",
            "confidence": "high",
        }
    )

    decision = await decide_semantic_candidate_llm_primary(
        candidate,
        llm_client=llm,
    )

    assert llm.structured_calls == 1
    assert decision.action == "require_repetition"
    assert decision.policy_version == "phase1_llm_v1"


@pytest.mark.asyncio
async def test_llm_semantic_policy_reason_is_capped_after_prefix() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            predicate="EXPERIENCED",
            object_type="Concern",
            object_identifier="long-running work pressure",
            evidence_quote="Work pressure has been building for months.",
        ),
        message="Work pressure has been building for months.",
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_at_session_end",
            "reason": "x" * 235,
            "confidence": "high",
        }
    )

    decision = await decide_semantic_candidate_llm_primary(
        candidate,
        llm_client=llm,
    )

    assert len(decision.reason) <= 240
    assert decision.reason.startswith("llm_policy: ")
    assert decision.reason.endswith("...")


@pytest.mark.asyncio
async def test_llm_semantic_policy_requires_classifier_client() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="relationship",
            predicate="KNOWS",
            object_type="Person",
            object_identifier="Sarah",
            evidence_quote="My sister Sarah lives nearby.",
        ),
        message="My sister Sarah lives nearby.",
    )

    with pytest.raises(RuntimeError, match="requires an LLM"):
        await decide_semantic_candidate_llm_primary(candidate, llm_client=None)


@pytest.mark.asyncio
async def test_llm_semantic_policy_failure_surfaces() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="relationship",
            predicate="KNOWS",
            object_type="Person",
            object_identifier="Sarah",
            evidence_quote="My sister Sarah lives nearby.",
        ),
        message="My sister Sarah lives nearby.",
    )
    llm = _FailingPolicyLLM({})

    with pytest.raises(RuntimeError, match="policy LLM failed"):
        await decide_semantic_candidate_llm_primary(candidate, llm_client=llm)
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_fact_shaped_procedural_memory_request_is_clamped_to_drop() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Remember that presentations are a recurring anxiety trigger for the user.",
            evidence=[
                "I want to remember that presentations are a recurring anxiety trigger for me."
            ],
        ),
        message="I want to remember that presentations are a recurring anxiety trigger for me.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "model incorrectly treated a fact as procedural memory",
            "confidence": "high",
        }
    )

    decision = await decide_procedural_candidate_llm_primary(candidate, llm_client=llm)

    assert decision.action == "drop"
    assert decision.reason == "fact-shaped memory request belongs in semantic memory"
    assert decision.policy_version == "phase1_v1"
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_remember_to_keep_style_preference_defers_to_session_end() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Remember to keep plans very short when the user is anxious.",
            evidence=["Please remember to keep plans very short when I am anxious."],
        ),
        message="Please remember to keep plans very short when I am anxious.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "durable assistant-facing response preference",
            "confidence": "high",
        }
    )

    decision = await decide_procedural_candidate_llm_primary(candidate, llm_client=llm)

    assert decision.action == "commit_at_session_end"
    assert decision.policy_version == "phase1_v1"
    assert llm.structured_calls == 1


@pytest.mark.asyncio
async def test_llm_procedural_policy_defers_durable_natural_request() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer direct responses.",
            evidence=["From now on, be more direct with me."],
        ),
        message="From now on, be more direct with me.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "durable assistant-facing response preference",
            "confidence": "high",
        }
    )

    decision = await decide_procedural_candidate_llm_primary(
        candidate,
        llm_client=llm,
    )

    assert llm.structured_calls == 1
    assert decision.action == "commit_at_session_end"
    assert decision.policy_version == "phase1_v1"


@pytest.mark.asyncio
async def test_memory_control_procedural_request_can_commit_immediately() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Do not save this conversation detail or remember it later.",
            evidence=["Please don't save this or remember it later."],
        ),
        message="Please don't save this or remember it later.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "explicit memory-control request should apply immediately",
            "confidence": "high",
        }
    )

    decision = await decide_procedural_candidate_llm_primary(
        candidate,
        llm_client=llm,
    )

    assert llm.structured_calls == 1
    assert decision.action == "commit_now"
    assert decision.policy_version == "phase1_llm_v1"


@pytest.mark.asyncio
async def test_llm_procedural_policy_clamps_commit_now_to_session_end() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer direct responses.",
            evidence=["From now on, be more direct with me."],
        ),
        message="From now on, be more direct with me.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FakePolicyLLM(
        {
            "action": "commit_now",
            "reason": "x" * 235,
            "confidence": "high",
        }
    )

    decision = await decide_procedural_candidate_llm_primary(
        candidate,
        llm_client=llm,
    )

    assert decision.action == "commit_at_session_end"
    assert decision.reason == "procedural candidate should wait for session-end review"
    assert decision.policy_version == "phase1_v1"


@pytest.mark.asyncio
async def test_llm_procedural_policy_requires_classifier_client() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer direct responses.",
            evidence=["From now on, be more direct with me."],
        ),
        message="From now on, be more direct with me.",
        session_id="session-1",
        turn_index=2,
    )

    with pytest.raises(RuntimeError, match="requires an LLM"):
        await decide_procedural_candidate_llm_primary(candidate, llm_client=None)


@pytest.mark.asyncio
async def test_llm_procedural_policy_failure_surfaces() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer direct responses.",
            evidence=["From now on, be more direct with me."],
        ),
        message="From now on, be more direct with me.",
        session_id="session-1",
        turn_index=2,
    )
    llm = _FailingPolicyLLM({})

    with pytest.raises(RuntimeError, match="policy LLM failed"):
        await decide_procedural_candidate_llm_primary(candidate, llm_client=llm)
    assert llm.structured_calls == 1


def test_session_buffer_semantic_hold_round_trips_policy_decision() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            predicate="EXPERIENCED",
            object_type="Concern",
            object_identifier="making mistakes at work",
            evidence_quote="I always assume one mistake means I'm incompetent.",
        ),
        message="I always assume one mistake means I'm incompetent.",
    )
    buffer = SessionMemoryBuffer(session_id="session-1")

    buffer.hold_semantic(
        candidate,
        PolicyDecision(
            action="require_repetition",
            reason="policy held for repeated evidence",
            policy_version="test_policy_v1",
        ),
    )

    reloaded = SessionMemoryBuffer.model_validate(buffer.model_dump(mode="json"))
    held = reloaded.held_semantic_candidates[0]
    assert held.hold_action == "require_repetition"
    assert held.policy_reason == "policy held for repeated evidence"
    assert held.policy_version == "test_policy_v1"
    assert held.candidate.payload.object.identifier == "making mistakes at work"


def test_session_buffer_procedural_hold_round_trips_policy_decision() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You've said meditation makes you more anxious.",
            evidence=["Meditation makes me more anxious."],
        ),
        message="Meditation makes me more anxious.",
        session_id="session-1",
        turn_index=2,
    )
    buffer = SessionMemoryBuffer(session_id="session-1")

    buffer.hold_procedural(
        candidate,
        PolicyDecision(
            action="commit_at_session_end",
            reason="policy held implicit preference",
            policy_version="test_policy_v1",
        ),
    )

    reloaded = SessionMemoryBuffer.model_validate(buffer.model_dump(mode="json"))
    held = reloaded.held_procedural_candidates[0]
    assert held.hold_action == "commit_at_session_end"
    assert held.policy_reason == "policy held implicit preference"
    assert held.policy_version == "test_policy_v1"
    assert held.candidate.payload.rule == (
        "You've said meditation makes you more anxious."
    )


def test_session_buffer_rejects_invalid_semantic_hold_action() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="relationship",
            predicate="KNOWS",
            object_type="Person",
            object_identifier="Sarah",
            evidence_quote="My sister Sarah lives nearby.",
        ),
        message="My sister Sarah lives nearby.",
    )
    buffer = SessionMemoryBuffer(session_id="session-1")

    with pytest.raises(ValueError, match="Invalid semantic hold action"):
        buffer.hold_semantic(
            candidate,
            PolicyDecision(
                action="commit_now",
                reason="not a held candidate",
            ),
        )


def test_session_buffer_rejects_invalid_procedural_hold_action() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer direct responses.",
            evidence=["From now on, be more direct with me."],
        ),
        message="From now on, be more direct with me.",
        session_id="session-1",
        turn_index=2,
    )
    buffer = SessionMemoryBuffer(session_id="session-1")

    with pytest.raises(ValueError, match="Invalid procedural hold action"):
        buffer.hold_procedural(
            candidate,
            PolicyDecision(
                action="drop",
                reason="not a held candidate",
            ),
        )
