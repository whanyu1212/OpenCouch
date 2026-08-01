"""Tests for explicit OpenAI text-route registry behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from agent.runtime.route_registry import (
    TextRouteRegistry,
    build_default_text_route_registry,
)
from agent.runtime.text_turn_graph import (
    PreparedTurn,
    TextRouteKind,
    TextRoutePlan,
)
from agent.runtime.types import (
    RouteHandler,
    TextRuntimeStateEvent,
    TextRuntimeStreamEvent,
)


async def _execute_handler(
    plan: TextRoutePlan,
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    return plan.state


async def _stream_handler(
    plan: TextRoutePlan,
    **kwargs: Any,
) -> AsyncIterator[TextRuntimeStreamEvent]:
    del kwargs
    yield TextRuntimeStateEvent(state=plan.state)


def _handler() -> RouteHandler:
    return RouteHandler(execute=_execute_handler, stream=_stream_handler)


def _plan(kind: TextRouteKind = "therapeutic") -> TextRoutePlan:
    state: dict[str, Any] = {"route": kind}
    prepared = PreparedTurn(state=state, eligible=True)
    return TextRoutePlan(kind=kind, prepared=prepared)


def test_route_registry_resolves_explicit_handlers_without_mutating_input() -> None:
    crisis = _handler()
    guided = _handler()
    therapeutic = _handler()
    handlers = {
        "crisis_response": crisis,
        "crisis_clarification": crisis,
        "guided_exercise": guided,
        "therapeutic": therapeutic,
    }

    registry = TextRouteRegistry(handlers)
    handlers["crisis_response"] = guided

    assert registry.handler_for("crisis_response") is crisis
    assert registry.handler_for("crisis_clarification") is crisis
    assert registry.handler_for("guided_exercise") is guided
    assert registry.handler_for("therapeutic") is therapeutic
    with pytest.raises(KeyError, match="unknown"):
        registry.handler_for("unknown")


def test_route_registry_handlers_mapping_is_immutable() -> None:
    registry = TextRouteRegistry({})

    with pytest.raises(TypeError):
        registry.handlers["therapeutic"] = _handler()  # type: ignore[index]


def test_default_registry_is_complete_for_known_text_route_kinds() -> None:
    registry = build_default_text_route_registry(SimpleNamespace())  # type: ignore[arg-type]

    assert registry.handler_for("crisis_response") is registry.handler_for(
        "crisis_clarification"
    )
    assert registry.handler_for("therapeutic") is registry.handler_for("memory_control")
    with pytest.raises(KeyError, match="unknown"):
        registry.handler_for("unknown")

    # Intentional change gate: add any new route kind here when wiring its handler.
    assert set(registry.handlers) == {
        "crisis_response",
        "crisis_clarification",
        "grounded_lookup",
        "guided_exercise",
        "memory_control",
        "therapeutic",
    }


@pytest.mark.asyncio
async def test_openai_text_runtime_uses_injected_registry_for_execute_and_stream() -> (
    None
):
    from agent.runtime.openai_text_runtime import OpenAITextRuntime

    calls: list[tuple[str, object, object, object]] = []
    final_state: dict[str, Any] = {"response_text": "registry response"}

    async def execute(
        plan: TextRoutePlan,
        *,
        config: object,
        context: object,
        session: object,
    ) -> dict[str, Any]:
        calls.append(("execute", config, context, session))
        assert plan.kind == "therapeutic"
        return final_state

    async def stream(
        plan: TextRoutePlan,
        *,
        config: object,
        context: object,
        session: object,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        calls.append(("stream", config, context, session))
        assert plan.kind == "therapeutic"
        yield TextRuntimeStateEvent(state=final_state)

    registry = TextRouteRegistry(
        {"therapeutic": RouteHandler(execute=execute, stream=stream)}
    )
    runtime = OpenAITextRuntime(
        model="gpt-test",
        route_registry_factory=lambda services: registry,
    )
    plan = _plan()
    config = {"configurable": {"thread_id": "thread-1"}}
    context = SimpleNamespace()
    session = object()

    executed = await runtime._execute_route_plan(
        plan,
        config=config,
        context=context,
        session=session,
    )
    streamed = [
        event
        async for event in runtime._stream_route_plan(
            plan,
            config=config,
            context=context,
            session=session,
        )
    ]

    assert executed is final_state
    assert streamed == [TextRuntimeStateEvent(state=final_state)]
    assert calls == [
        ("execute", config, context, session),
        ("stream", config, context, session),
    ]


@pytest.mark.asyncio
async def test_flow_handler_resolves_fresh_services_for_each_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.flows import crisis as crisis_flow

    service_snapshots: list[object] = []

    def services_factory() -> object:
        services = object()
        service_snapshots.append(services)
        return services

    async def run_crisis_turn(
        services: object,
        state: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        assert services is service_snapshots[-1]
        return state

    monkeypatch.setattr(crisis_flow, "run_crisis_turn", run_crisis_turn)
    handler = crisis_flow.build_crisis_route_handler(services_factory)  # type: ignore[arg-type]
    plan = _plan("crisis_response")

    await handler.execute(
        plan,
        config={},
        context=SimpleNamespace(),
        session=None,
    )
    await handler.execute(
        plan,
        config={},
        context=SimpleNamespace(),
        session=None,
    )

    assert len(service_snapshots) == 2
    assert service_snapshots[0] is not service_snapshots[1]
