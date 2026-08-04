"""Direct characterization tests for app-owned text turn routing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.runtime.text_turn_graph import (
    PreparedTurn,
    TextRouteKind,
    TextRoutePlan,
    TextTurnGraph,
    TextTurnGraphResult,
    grounded_lookup_query_for_state,
)
from agent.runtime.types import TextRuntimeConfig
from agent.runtime.workflow_context import WorkflowContext
from agent.specialists.crisis import CRISIS_AGENT_NAME
from agent.specialists.guided_exercise import GUIDED_EXERCISE_AGENT_NAME
from agent.specialists.therapeutic import THERAPEUTIC_AGENT_NAME
from agent.state import AgentState, AgentTurnInputState


def _state(**values: Any) -> AgentState:
    return cast(AgentState, {"message": "fallback message", **values})


def _graph(
    prepared: PreparedTurn,
    *,
    routed_state: AgentState | None = None,
    guided_exercise: bool = False,
) -> tuple[TextTurnGraph, dict[str, int]]:
    calls = {"prepare": 0, "guided_exercise": 0}

    async def prepare_turn(*args: Any, **kwargs: Any) -> PreparedTurn:
        del args, kwargs
        calls["prepare"] += 1
        return prepared

    async def load_and_prepare_guided_exercise(
        state: AgentState,
        context: WorkflowContext,
    ) -> tuple[AgentState, bool]:
        del context
        calls["guided_exercise"] += 1
        return routed_state if routed_state is not None else state, guided_exercise

    return (
        TextTurnGraph(
            prepare_turn=prepare_turn,
            load_and_prepare_guided_exercise=load_and_prepare_guided_exercise,
        ),
        calls,
    )


async def _resolve(graph: TextTurnGraph) -> TextTurnGraphResult:
    return await graph.resolve(
        cast(AgentTurnInputState, {"message": "new message"}),
        config=cast(TextRuntimeConfig, {}),
        context=cast(WorkflowContext, object()),
        prior_state=cast(AgentState, {"message": "prior message"}),
    )


@pytest.mark.asyncio
async def test_ineligible_turn_has_no_plan_or_guided_exercise_load() -> None:
    prepared = PreparedTurn(state=_state(), eligible=False)
    graph, calls = _graph(prepared)

    result = await _resolve(graph)

    assert result.prepared is prepared
    assert result.plan is None
    assert calls == {"prepare": 1, "guided_exercise": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("crisis", "kind", "response_style", "stream_status_stages"),
    [
        (
            SimpleNamespace(needs_crisis_response=True, level=0),
            "crisis_response",
            "crisis_response",
            ("crisis_resource_lookup",),
        ),
        (
            SimpleNamespace(needs_clarification=True, level=0),
            "crisis_clarification",
            "clarifying",
            ("load_memory",),
        ),
    ],
)
async def test_crisis_turns_use_characterized_route_metadata(
    crisis: object,
    kind: TextRouteKind,
    response_style: str,
    stream_status_stages: tuple[str, ...],
) -> None:
    prepared = PreparedTurn(state=_state(crisis=crisis), eligible=True)
    graph, calls = _graph(prepared)

    result = await _resolve(graph)

    assert result.plan is not None
    assert result.plan.kind == kind
    assert result.plan.state is prepared.state
    assert result.plan.runtime_mode == kind
    assert result.plan.response_style == response_style
    assert result.plan.selected_agent == CRISIS_AGENT_NAME
    assert result.plan.query == ""
    assert result.plan.stream_status_stages == stream_status_stages
    assert calls == {"prepare": 1, "guided_exercise": 0}


@pytest.mark.asyncio
async def test_grounded_lookup_plan_uses_normalized_lookup_query() -> None:
    prepared = PreparedTurn(
        state=_state(
            route="grounded_lookup",
            grounded_lookup={"query": "  medication side effects  "},
        ),
        eligible=True,
    )
    graph, calls = _graph(prepared)

    result = await _resolve(graph)

    assert result.plan is not None
    assert result.plan.kind == "grounded_lookup"
    assert result.plan.runtime_mode == "grounded_lookup"
    assert result.plan.response_style == "grounded_lookup"
    assert result.plan.selected_agent == THERAPEUTIC_AGENT_NAME
    assert result.plan.query == "medication side effects"
    assert result.plan.stream_status_stages == ("grounded_lookup",)
    assert calls == {"prepare": 1, "guided_exercise": 0}
    assert grounded_lookup_query_for_state(_state(message="  fallback query  ")) == (
        "fallback query"
    )


@pytest.mark.asyncio
async def test_guided_exercise_plan_uses_routed_prepared_state() -> None:
    prepared = PreparedTurn(
        state=_state(route="therapeutic"),
        eligible=True,
        fallback_reason="prior fallback",
    )
    routed_state = _state(route="guided_exercise", message="routed message")
    graph, calls = _graph(
        prepared,
        routed_state=routed_state,
        guided_exercise=True,
    )

    result = await _resolve(graph)

    assert result.prepared is prepared
    assert result.plan is not None
    assert result.plan.kind == "guided_exercise"
    assert result.plan.prepared.state is routed_state
    assert result.plan.prepared.fallback_reason == "prior fallback"
    assert result.plan.state is routed_state
    assert result.plan.runtime_mode == "guided_exercise"
    assert result.plan.response_style == "guided_exercise"
    assert result.plan.selected_agent == GUIDED_EXERCISE_AGENT_NAME
    assert result.plan.query == ""
    assert result.plan.stream_status_stages == ("load_memory",)
    assert calls == {"prepare": 1, "guided_exercise": 1}


@pytest.mark.asyncio
async def test_memory_control_plan_uses_shared_therapeutic_agent() -> None:
    prepared = PreparedTurn(state=_state(), eligible=True)
    routed_state = _state(route="memory_control")
    graph, calls = _graph(prepared, routed_state=routed_state)

    result = await _resolve(graph)

    assert result.plan is not None
    assert result.plan.kind == "memory_control"
    assert result.plan.state is routed_state
    assert result.plan.runtime_mode == "memory_control"
    assert result.plan.response_style == "memory_control"
    assert result.plan.selected_agent == THERAPEUTIC_AGENT_NAME
    assert result.plan.query == ""
    assert result.plan.stream_status_stages == ("load_memory",)
    assert calls == {"prepare": 1, "guided_exercise": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_style", "expected_response_style"),
    [("reflective", "reflective"), (None, "supportive")],
)
async def test_therapeutic_fallback_uses_routed_response_style(
    response_style: str | None,
    expected_response_style: str,
) -> None:
    prepared = PreparedTurn(state=_state(), eligible=True)
    routed_state = _state(response_style=response_style)
    graph, calls = _graph(prepared, routed_state=routed_state)

    result = await _resolve(graph)

    assert result.plan is not None
    assert result.plan.kind == "therapeutic"
    assert result.plan.state is routed_state
    assert result.plan.runtime_mode == "safe_therapeutic"
    assert result.plan.response_style == expected_response_style
    assert result.plan.selected_agent == THERAPEUTIC_AGENT_NAME
    assert result.plan.query == ""
    assert result.plan.stream_status_stages == ("load_memory",)
    assert calls == {"prepare": 1, "guided_exercise": 1}


def test_plan_snapshots_dynamic_metadata_from_prepared_state() -> None:
    therapeutic_state = _state(response_style="reflective")
    therapeutic_plan = TextRoutePlan(
        kind="therapeutic",
        prepared=PreparedTurn(state=therapeutic_state, eligible=True),
    )
    lookup_state = _state(
        route="grounded_lookup",
        grounded_lookup={"query": "initial query"},
    )
    lookup_plan = TextRoutePlan(
        kind="grounded_lookup",
        prepared=PreparedTurn(state=lookup_state, eligible=True),
    )

    therapeutic_state["response_style"] = "closing"
    lookup_state["grounded_lookup"] = {"query": "changed query"}

    assert therapeutic_plan.response_style == "reflective"
    assert lookup_plan.query == "initial query"
