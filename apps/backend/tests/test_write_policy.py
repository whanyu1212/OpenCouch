"""Unit tests for the deterministic phase-1 memory write policy."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.memory.models import EntityRef, MemoryWrite, ProceduralRuleDraft
from agent.memory.policy.candidates import (
    build_procedural_candidate,
    build_semantic_candidate,
)
from agent.memory.policy.write import (
    decide_procedural_candidate_llm_primary,
    decide_procedural_candidate,
    decide_semantic_candidate_llm_primary,
    decide_semantic_candidate,
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


def test_explicit_stable_semantic_fact_commits_now() -> None:
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

    decision = decide_semantic_candidate(candidate)

    assert decision.action == "commit_now"


def test_high_sensitivity_semantic_fact_waits_for_session_end() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="trigger",
            object_identifier="panic in family conflict",
            evidence_quote="Family conflict is a big trigger for panic.",
        ),
        message="Family conflict is a big trigger for panic.",
    )

    decision = decide_semantic_candidate(candidate)

    assert decision.action == "commit_at_session_end"


def test_negative_self_belief_requires_repetition() -> None:
    candidate = build_semantic_candidate(
        _semantic_write(
            category="context",
            object_identifier="making mistakes at work",
            evidence_quote="I always assume one mistake means I'm incompetent.",
        ),
        message="I always assume one mistake means I'm incompetent.",
    )

    decision = decide_semantic_candidate(candidate)

    assert decision.action == "require_repetition"
    assert should_commit_pattern(candidate, evidence_count=1) is False
    assert should_commit_pattern(candidate, evidence_count=2) is True


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

    decision = decide_semantic_candidate(candidate)

    assert decision.action == "drop"


def test_explicit_procedural_request_commits_now() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer shorter responses.",
            evidence=["Please keep responses shorter."],
        ),
        message="Please keep responses shorter.",
        session_id="session-1",
        turn_index=2,
    )

    decision = decide_procedural_candidate(candidate)

    assert decision.action == "commit_now"


def test_implicit_procedural_preference_requires_repetition_to_promote() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You've said meditation makes you more anxious.",
            evidence=["Meditation makes me more anxious."],
        ),
        message="Meditation makes me more anxious.",
        session_id="session-1",
        turn_index=2,
    )

    decision = decide_procedural_candidate(candidate)

    assert decision.action == "commit_at_session_end"
    assert (
        should_commit_implicit_procedural_preference(candidate, evidence_count=1)
        is False
    )
    assert (
        should_commit_implicit_procedural_preference(candidate, evidence_count=2)
        is True
    )


def test_turn_scoped_procedural_request_drops() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="You prefer shorter responses.",
            evidence=["For this reply, keep it short."],
        ),
        message="For this reply, keep it short.",
        session_id="session-1",
        turn_index=2,
    )

    decision = decide_procedural_candidate(candidate)

    assert decision.action == "drop"


def test_safety_conflicting_procedural_request_drops() -> None:
    candidate = build_procedural_candidate(
        ProceduralRuleDraft(
            rule="Don't ask me if I'm safe.",
            evidence=["Don't ask if I'm safe."],
        ),
        message="Don't ask if I'm safe.",
        session_id="session-1",
        turn_index=2,
    )

    decision = decide_procedural_candidate(candidate)

    assert decision.action == "drop"


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
async def test_llm_procedural_policy_can_commit_durable_natural_request() -> None:
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
    assert decision.action == "commit_now"
    assert decision.policy_version == "phase1_llm_v1"


@pytest.mark.asyncio
async def test_llm_procedural_policy_reason_is_capped_after_prefix() -> None:
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

    assert len(decision.reason) <= 240
    assert decision.reason.startswith("llm_policy: ")
    assert decision.reason.endswith("...")
