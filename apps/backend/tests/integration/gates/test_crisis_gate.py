"""LLM-only crisis-gate tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from agent.audit.crisis_log import InMemoryCrisisLogBackend
from agent.guardrails.service import CrisisAssessmentSchema
from agent.guardrails.assessment import assess_crisis_gate
from agent.runtime import build_initial_state, run_agent
from agent.memory.modes import MemoryMode
from agent.memory.store import Namespace, OpenCouchMemoryStore, StoreRecord
from agent.models import AgentInput, ResponseCategory
from agent.runtime_context import WorkflowContext
from llm.base import BaseLLMClient, StructuredResponseT
from tests.support.persistence import FakeCrossRestartLLM


class _MockRuntime:
    """Minimal runtime stand-in exposing the workflow context only."""

    def __init__(self, *, llm_client: BaseLLMClient | None = None) -> None:
        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        )


class _CannedCrisisLLM(BaseLLMClient):
    """Fake LLM client returning one canned crisis assessment."""

    def __init__(
        self,
        response: CrisisAssessmentSchema,
        *,
        stream_text: str = "fake crisis response",
    ) -> None:
        self._response = response
        self._stream_text = stream_text
        self.structured_calls: list[str] = []

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
        yield self._stream_text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        schema_name = response_schema.__name__
        self.structured_calls.append(schema_name)

        if schema_name == "CrisisAssessmentSchema":
            return cast(StructuredResponseT, self._response)
        if schema_name == "CrisisLocationDecision":
            return response_schema(  # type: ignore[call-arg,return-value]
                status="not_provided",
                location="",
                reasoning="No user location in this test fixture.",
            )

        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")


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


def _crisis_schema(
    *,
    level: int,
    reason: str,
    confidence: str = "high",
    needs_crisis_response: bool | None = None,
    needs_clarification: bool | None = None,
) -> CrisisAssessmentSchema:
    """Build a crisis classifier response for tests."""

    return CrisisAssessmentSchema(
        level=level,
        confidence=confidence,  # type: ignore[arg-type]
        reason=reason,
        needs_crisis_response=level >= 2
        if needs_crisis_response is None
        else needs_crisis_response,
        needs_clarification=level == 1
        if needs_clarification is None
        else needs_clarification,
    )


def _crisis_goto(update: dict[str, Any]) -> str:
    crisis = update["crisis"]
    return (
        "crisis_resource_lookup_node"
        if crisis.needs_crisis_response
        else "safe_runtime"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "expected_goto", "expected_route", "expected_decision"),
    [
        (3, "crisis_resource_lookup_node", "crisis", "crisis"),
        (2, "crisis_resource_lookup_node", "crisis", "crisis"),
        (1, "safe_runtime", "therapeutic", "check"),
        (0, "safe_runtime", "therapeutic", "normal"),
    ],
)
async def test_assess_crisis_gate_uses_llm_level_for_routing(
    level: int,
    expected_goto: str,
    expected_route: str,
    expected_decision: str,
) -> None:
    """The node should route from the structured LLM verdict only."""

    state = build_initial_state(
        AgentInput(message="Current user message for crisis classification."),
        include_input_history=True,
    )
    llm = _CannedCrisisLLM(
        _crisis_schema(level=level, reason=f"LLM classified level {level}")
    )

    result = await assess_crisis_gate(
        state,
        llm_client=_MockRuntime(llm_client=llm).context.llm_client,
    )
    update = result.delta

    assert _crisis_goto(update) == expected_goto
    assert update["crisis"].level == level
    assert update["crisis"].needs_crisis_response is (level >= 2)
    assert update["crisis"].needs_clarification is (level == 1)
    assert update["route"] == expected_route
    assert update["crisis_audit"]["crisis_override_kind"] == "none"
    assert update["crisis_audit"]["crisis_classifier_path"] == "llm_primary"
    assert update["crisis_audit"]["crisis_llm_failure_occurred"] is False
    trace = update["diagnostics"]["routing_trace"]
    assert trace[-1]["stage"] == "safety"
    assert trace[-1]["decision"] == expected_decision
    assert trace[-1]["source"] == "llm_primary"
    assert trace[-1]["reason"] == f"LLM classified level {level}"


@pytest.mark.asyncio
async def test_assess_crisis_gate_requires_llm_client() -> None:
    """Without an LLM client, crisis classification should fail visibly."""

    state = build_initial_state(
        AgentInput(message="I just wish I could disappear for a while."),
        include_input_history=True,
    )

    with pytest.raises(RuntimeError, match="requires an LLM client"):
        await assess_crisis_gate(
            state,
            llm_client=_MockRuntime().context.llm_client,
        )


@pytest.mark.asyncio
async def test_assess_crisis_gate_propagates_llm_failure() -> None:
    """LLM errors should be left to the caller's retry policy, not hidden."""

    state = build_initial_state(
        AgentInput(message="I just wish I could disappear for a while."),
        include_input_history=True,
    )

    with pytest.raises(RuntimeError, match="simulated classifier failure"):
        await assess_crisis_gate(
            state,
            llm_client=_MockRuntime(
                llm_client=_FailingStructuredLLM()
            ).context.llm_client,
        )


@pytest.mark.asyncio
async def test_assess_crisis_gate_enforces_truth_table_on_llm_output() -> None:
    """The node should normalize inconsistent LLM booleans to match the level."""

    state = build_initial_state(
        AgentInput(message="I just wish I could disappear for a while."),
        include_input_history=True,
    )
    llm = _CannedCrisisLLM(
        _crisis_schema(
            level=0,
            confidence="medium",
            reason="LLM returned inconsistent booleans",
            needs_crisis_response=True,
            needs_clarification=True,
        )
    )

    result = await assess_crisis_gate(
        state,
        llm_client=_MockRuntime(llm_client=llm).context.llm_client,
    )
    update = result.delta

    assert _crisis_goto(update) == "safe_runtime"
    assert update["crisis"].level == 0
    assert update["crisis"].needs_crisis_response is False
    assert update["crisis"].needs_clarification is False
    assert update["crisis_audit"]["crisis_classifier_path"] == "llm_primary"


@pytest.mark.asyncio
async def test_run_agent_does_not_overtrigger_common_idiom_when_llm_marks_safe() -> (
    None
):
    """Common idioms should stay non-crisis when the LLM classifies them safe."""

    result = await run_agent(
        AgentInput(message="Work is killing me lately."),
        llm_client=FakeCrossRestartLLM(),
    )

    assert result.crisis.level == 0
    assert result.crisis.needs_crisis_response is False
    assert result.response_type == ResponseCategory.THERAPEUTIC


# ─── v0.9 safety-reorder regression tests ─────────────────────────────────


class _BrokenMemoryStore(OpenCouchMemoryStore):
    """A memory store that raises on every read operation."""

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
    """A crisis turn should not touch memory retrieval before crisis response."""

    result = await run_agent(
        AgentInput(message="I have pills and I am going to kill myself tonight."),
        llm_client=_CannedCrisisLLM(
            _crisis_schema(level=3, reason="LLM classified imminent risk")
        ),
        memory_store=_BrokenMemoryStore(),
    )

    assert result.crisis.level == 3
    assert result.response_type == ResponseCategory.CRISIS
    assert result.crisis.needs_crisis_response


@pytest.mark.asyncio
async def test_crisis_turns_do_not_emit_extraction_diagnostics() -> None:
    """Crisis turns should not carry removed extraction diagnostics."""

    result = await run_agent(
        AgentInput(message="I've been thinking about ending it all."),
        llm_client=_CannedCrisisLLM(
            _crisis_schema(level=2, reason="LLM classified suicidal ideation")
        ),
    )

    assert result.crisis.level == 2
    assert result.response_type == ResponseCategory.CRISIS
    diag = result.diagnostics
    assert "extract_facts_reason" not in diag
    assert "extract_procedural_reason" not in diag
