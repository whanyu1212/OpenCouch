"""Deterministic crisis-gate tests.

These exercise the rule-based crisis classifier and the override detector
without requiring an LLM client. The end-to-end ``run_agent`` tests have been
removed pending the therapeutic-response rebuild.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.graph import build_initial_state, run_agent
from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.memory.modes import MemoryMode
from agent.memory.store import Namespace, OpenCouchMemoryStore, StoreRecord
from agent.models import AgentInput, ResponseCategory
from agent.nodes.crisis_gate import run_crisis_gate_node
from agent.gates.safety.crisis_rules import (
    assess_crisis_risk_deterministically,
    detect_crisis_override,
)
from agent.gates.safety.service import CrisisAssessmentSchema
from agent.runtime_context import WorkflowContext
from llm.base import BaseLLMClient, StructuredResponseT


class _MockRuntime:
    """Minimal runtime stand-in exposing the workflow context only."""

    def __init__(self, *, llm_client: BaseLLMClient | None = None) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        )


class _CannedStructuredLLM(BaseLLMClient):
    """Fake LLM client returning one canned crisis assessment."""

    def __init__(self, response: CrisisAssessmentSchema) -> None:
        self._response = response

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
        return cast(StructuredResponseT, self._response)


class _FailingStructuredLLM(BaseLLMClient):
    """Fake LLM client that raises on structured generation."""

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
        raise RuntimeError("simulated classifier failure")


@pytest.mark.asyncio
async def test_routes_clear_imminent_risk_to_crisis() -> None:
    """Imminent-risk language should route directly to crisis mode."""

    result = await run_agent(
        AgentInput(message="I have pills and I am going to kill myself tonight.")
    )

    assert result.crisis.level == 3
    assert result.response_type == ResponseCategory.CRISIS
    assert result.crisis.needs_crisis_response


@pytest.mark.asyncio
async def test_routes_clear_ideation_to_crisis() -> None:
    """Clear suicidal ideation should route to crisis mode."""

    result = await run_agent(
        AgentInput(message="I've been thinking about ending it all.")
    )

    assert result.crisis.level == 2
    assert result.response_type == ResponseCategory.CRISIS


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


def test_detects_benign_planning_safe_override_separately() -> None:
    """Benign planning language should not trigger a crisis clarification."""

    state = build_initial_state(
        AgentInput(message="Okay, I've planned it within my head."),
        include_input_history=True,
    )

    override = detect_crisis_override(state)
    assert override is not None
    kind, assessment = override
    assert kind == "idiomatic_safe"
    assert assessment.level == 0
    assert assessment.needs_clarification is False


def test_post_safety_check_deescalates_default_crisis_reply_language() -> None:
    """A post-crisis denial should de-escalate even after the default crisis reply."""

    state = build_initial_state(
        AgentInput(
            message="I'm not going to do anything tonight. I just needed to say it out loud.",
            history=[
                {"role": "user", "content": "I've been thinking about ending it."},
                {
                    "role": "assistant",
                    "content": (
                        "Thank you for telling me this — I'm really glad you reached out. "
                        "Your safety matters most right now. "
                        "If you feel at risk of harming yourself, please contact your local "
                        "emergency services right now or go to the nearest emergency department. "
                        "If possible, move away from anything you could use to hurt yourself and "
                        "contact a trusted person who can stay with you."
                    ),
                },
            ],
        ),
        include_input_history=True,
    )

    override = detect_crisis_override(state)
    assert override is not None
    kind, assessment = override
    assert kind == "idiomatic_safe"
    assert assessment.level == 0
    assert assessment.needs_crisis_response is False
    assert assessment.needs_clarification is False


# ─── Standalone crisis-gate node tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_run_crisis_gate_node_override_path_standalone() -> None:
    """The node should route imminent-risk override turns on its own."""

    state = build_initial_state(
        AgentInput(message="I have pills and I am going to kill myself tonight."),
        include_input_history=True,
    )

    command = await run_crisis_gate_node(state, _MockRuntime())  # type: ignore[arg-type]

    assert command.goto == "crisis_resource_lookup_node"
    assert command.update["crisis"].level == 3
    assert command.update["crisis"].needs_crisis_response is True
    assert command.update["route"] == "crisis"
    assert command.update["crisis_audit"]["crisis_override_kind"] == "imminent_risk"
    assert command.update["crisis_audit"]["crisis_classifier_path"] == "override"
    assert command.update["crisis_audit"]["crisis_llm_failure_occurred"] is False
    assert command.update["response_kind"] == ResponseCategory.CRISIS


@pytest.mark.asyncio
async def test_run_crisis_gate_node_no_llm_path_standalone() -> None:
    """Without an LLM client, the node should use deterministic classification."""

    state = build_initial_state(
        AgentInput(message="I just wish I could disappear for a while."),
        include_input_history=True,
    )

    command = await run_crisis_gate_node(state, _MockRuntime())  # type: ignore[arg-type]

    assert command.goto == "memory_control_gate_node"
    assert command.update["crisis"].level == 1
    assert command.update["crisis"].needs_crisis_response is False
    assert command.update["crisis"].needs_clarification is True
    assert command.update["route"] == "therapeutic"
    assert command.update["crisis_audit"]["crisis_override_kind"] == "none"
    assert command.update["crisis_audit"]["crisis_classifier_path"] == "deterministic"
    assert command.update["crisis_audit"]["crisis_llm_failure_occurred"] is False


@pytest.mark.asyncio
async def test_run_crisis_gate_node_llm_success_path_standalone() -> None:
    """The node should use the primary LLM verdict when the call succeeds."""

    state = build_initial_state(
        AgentInput(message="I just wish I could disappear for a while."),
        include_input_history=True,
    )
    llm = _CannedStructuredLLM(
        CrisisAssessmentSchema(
            level=2,
            confidence="high",
            reason="LLM escalated the message to level 2",
            needs_crisis_response=True,
            needs_clarification=False,
        )
    )

    command = await run_crisis_gate_node(  # type: ignore[arg-type]
        state,
        _MockRuntime(llm_client=llm),
    )

    assert command.goto == "crisis_resource_lookup_node"
    assert command.update["crisis"].level == 2
    assert command.update["route"] == "crisis"
    assert command.update["crisis_audit"]["crisis_override_kind"] == "none"
    assert command.update["crisis_audit"]["crisis_classifier_path"] == "llm_primary"
    assert command.update["crisis_audit"]["crisis_llm_failure_occurred"] is False
    assert command.update["response_kind"] == ResponseCategory.CRISIS


@pytest.mark.asyncio
async def test_run_crisis_gate_node_llm_failure_fallback_standalone() -> None:
    """The node should fall back to deterministic classification on LLM failure."""

    state = build_initial_state(
        AgentInput(message="I just wish I could disappear for a while."),
        include_input_history=True,
    )

    command = await run_crisis_gate_node(  # type: ignore[arg-type]
        state,
        _MockRuntime(llm_client=_FailingStructuredLLM()),
    )

    assert command.goto == "memory_control_gate_node"
    assert command.update["crisis"].level == 1
    assert command.update["crisis"].needs_crisis_response is False
    assert command.update["crisis"].needs_clarification is True
    assert command.update["crisis_audit"]["crisis_override_kind"] == "none"
    assert command.update["crisis_audit"]["crisis_classifier_path"] == "deterministic"
    assert command.update["crisis_audit"]["crisis_llm_failure_occurred"] is True


@pytest.mark.asyncio
async def test_run_crisis_gate_node_enforces_truth_table_on_llm_output() -> None:
    """The node should normalize inconsistent LLM booleans to match the level."""

    state = build_initial_state(
        AgentInput(message="I just wish I could disappear for a while."),
        include_input_history=True,
    )
    llm = _CannedStructuredLLM(
        CrisisAssessmentSchema(
            level=0,
            confidence="medium",
            reason="LLM returned inconsistent booleans",
            needs_crisis_response=True,
            needs_clarification=True,
        )
    )

    command = await run_crisis_gate_node(  # type: ignore[arg-type]
        state,
        _MockRuntime(llm_client=llm),
    )

    assert command.goto == "memory_control_gate_node"
    assert command.update["crisis"].level == 0
    assert command.update["crisis"].needs_crisis_response is False
    assert command.update["crisis"].needs_clarification is False
    assert command.update["crisis_audit"]["crisis_classifier_path"] == "llm_primary"


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
    (START → crisis_gate → crisis_resource_lookup → crisis_response) ensures
    the crisis path never touches load_memory_node, so store failures are
    irrelevant."""

    result = await run_agent(
        AgentInput(message="I have pills and I am going to kill myself tonight."),
        memory_store=_BrokenMemoryStore(),
    )

    assert result.crisis.level == 3
    assert result.response_type == ResponseCategory.CRISIS
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
    assert result.response_type == ResponseCategory.CRISIS
    # Both extractors should have short-circuited with crisis_path reason
    diag = result.diagnostics
    assert diag.get("extract_facts_reason") == "skipped: crisis_path"
    assert diag.get("extract_procedural_reason") == "skipped: crisis_path"
