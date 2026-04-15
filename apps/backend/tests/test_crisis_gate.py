"""Deterministic crisis-gate tests.

These exercise the rule-based crisis classifier and the override detector
without requiring an LLM client. The end-to-end ``run_agent`` tests have been
removed pending the therapeutic-response rebuild.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent.graph import build_initial_state, run_agent
from agent.memory.store import Namespace, OpenCouchMemoryStore, StoreRecord
from agent.models import AgentInput, ResponseKind
from agent.nodes.crisis_gate import (
    assess_crisis_risk_deterministically,
    detect_crisis_override,
)


@pytest.mark.asyncio
async def test_routes_clear_imminent_risk_to_crisis() -> None:
    """Imminent-risk language should route directly to crisis mode."""

    result = await run_agent(
        AgentInput(message="I have pills and I am going to kill myself tonight.")
    )

    assert result.crisis.level == 3
    assert result.response_type == ResponseKind.CRISIS
    assert result.crisis.needs_crisis_response


@pytest.mark.asyncio
async def test_routes_clear_ideation_to_crisis() -> None:
    """Clear suicidal ideation should route to crisis mode."""

    result = await run_agent(
        AgentInput(message="I've been thinking about ending it all.")
    )

    assert result.crisis.level == 2
    assert result.response_type == ResponseKind.CRISIS


@pytest.mark.asyncio
async def test_does_not_overtrigger_common_idiom() -> None:
    """Common idioms should not overtrigger crisis routing."""

    result = await run_agent(AgentInput(message="Work is killing me lately."))

    assert result.crisis.level == 0
    assert result.crisis.needs_crisis_response is False


def test_uses_recent_user_history_for_context() -> None:
    """Deterministic crisis checks should consider recent user history."""

    state = build_initial_state(
        AgentInput(
            message="I keep thinking about it.",
            history=[{"role": "user", "content": "Sometimes I want to kill myself."}],
        ),
        include_input_history=True,
    )

    assessment = assess_crisis_risk_deterministically(state)
    assert assessment.level == 2


def test_detects_imminent_override_separately() -> None:
    """Imminent-risk override detection should work independently of the classifier."""

    state = build_initial_state(
        AgentInput(message="I have a plan to kill myself tonight."),
        include_input_history=True,
    )

    override = detect_crisis_override(state)
    assert override is not None
    kind, assessment = override
    assert kind == "imminent_risk"
    assert assessment.level == 3


def test_detects_idiomatic_safe_override_separately() -> None:
    """Idiomatic-safe override detection should suppress false crisis positives."""

    state = build_initial_state(
        AgentInput(message="Work is killing me this week."),
        include_input_history=True,
    )

    override = detect_crisis_override(state)
    assert override is not None
    kind, assessment = override
    assert kind == "idiomatic_safe"
    assert assessment.level == 0


# ─── v0.9 safety-reorder regression tests ─────────────────────────────────


class _BrokenMemoryStore(OpenCouchMemoryStore):
    """A memory store that raises on every read operation.

    Used to verify that the crisis path completes even when the
    memory store is entirely broken — the v0.9 graph reorder
    ensures crisis_gate runs before load_memory_node, so a broken
    store should never block crisis routing.
    """

    async def asearch(self, *args: Any, **kwargs: Any) -> list[StoreRecord]:
        raise RuntimeError("Store is broken")

    async def asearch_similar(self, *args: Any, **kwargs: Any) -> list[StoreRecord]:
        raise RuntimeError("Store is broken")

    async def arecord_count(self, namespace: Namespace | None = None) -> int:
        raise RuntimeError("Store is broken")

    async def aget(self, *args: Any, **kwargs: Any) -> StoreRecord | None:
        raise RuntimeError("Store is broken")

    async def alatest(self, namespace: Namespace) -> StoreRecord | None:
        raise RuntimeError("Store is broken")


@pytest.mark.asyncio
async def test_crisis_path_succeeds_with_broken_store() -> None:
    """v0.9 safety regression: a crisis message must reach crisis_response
    even when the memory store is entirely broken. The graph reorder
    (START → crisis_gate → crisis_response) ensures the crisis path
    never touches load_memory_node, so store failures are irrelevant."""

    result = await run_agent(
        AgentInput(message="I have pills and I am going to kill myself tonight."),
        memory_store=_BrokenMemoryStore(),
    )

    assert result.crisis.level == 3
    assert result.response_type == ResponseKind.CRISIS
    assert result.crisis.needs_crisis_response


@pytest.mark.asyncio
async def test_crisis_turns_skip_extractors() -> None:
    """v0.9: crisis turns must skip both extract_semantic_facts and
    extract_procedural_rules to avoid delaying crisis response delivery.
    The extractors' diagnostics should report 'skipped: crisis_path'."""

    result = await run_agent(
        AgentInput(message="I've been thinking about ending it all."),
    )

    assert result.crisis.level >= 2
    assert result.response_type == ResponseKind.CRISIS
    # Both extractors should have short-circuited with crisis_path reason
    diag = result.diagnostics
    assert diag.get("extract_facts_reason") == "skipped: crisis_path"
    assert diag.get("extract_procedural_reason") == "skipped: crisis_path"
