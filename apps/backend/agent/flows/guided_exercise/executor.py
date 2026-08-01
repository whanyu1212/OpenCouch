"""Guided exercise execution helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from llm.base import BaseLLMClient

from agent.flows.guided_exercise.adapters import guided_exercise_response_llm
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.services import TextRuntimeServices, TextRuntimeServicesFactory
from agent.runtime.state_ops import apply_state_delta
from agent.runtime.text_turn_graph import TextRoutePlan
from agent.runtime.types import (
    RouteHandler,
    TextRuntimeChunkEvent,
    TextRuntimeConfig,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.runtime.workflow_context import WorkflowContext
from agent.skills.guided_exercises.lifecycle.service import GuidedExerciseSkillService
from agent.specialists.guided_exercise import GUIDED_EXERCISE_AGENT_NAME
from agent.state import AgentState


def guided_exercise_skill_service(
    context: WorkflowContext,
    *,
    response_llm: BaseLLMClient,
    stream_writer_factory: Any | None = None,
) -> GuidedExerciseSkillService:
    kwargs: dict[str, Any] = {}
    if stream_writer_factory is not None:
        kwargs["stream_writer_factory"] = stream_writer_factory
    return GuidedExerciseSkillService(
        classifier_llm=context.llm_client,
        response_llm=response_llm,
        memory_store=context.memory_store,
        memory_mode=context.memory_mode,
        prompt_appendix=context.prompt_appendix,
        **kwargs,
    )


async def run_guided_exercise_turn(
    services: TextRuntimeServices,
    state: Any,
    *,
    config: Any,
    context: WorkflowContext,
    streamed: bool,
    session: Any | None = None,
) -> Any:
    response_llm = guided_exercise_response_llm(
        services,
        state,
        config,
        context,
        session=session,
    )
    skill_service = guided_exercise_skill_service(
        context,
        response_llm=response_llm,
    )
    delta = await skill_service.run_turn(state)
    apply_state_delta(state, dict(delta))
    _apply_guided_exercise_tool_diagnostics(
        state,
        response_llm.run_context,
        fallback=response_llm.used_skill_tool_fallback,
    )
    response_text = str(state.get("response_text") or "")
    if not response_text:
        raise ValueError("guided_exercise returned an empty response.")
    return await services.finalize_turn(
        state,
        response_text=response_text,
        config=config,
        runtime_mode="guided_exercise",
        response_style=str(state.get("response_style") or "guided_exercise"),
        selected_agent=GUIDED_EXERCISE_AGENT_NAME,
        sdk_duration_ms=response_llm.last_duration_ms,
        streamed=streamed,
    )


async def run_guided_exercise_turn_stream(
    services: TextRuntimeServices,
    state: Any,
    *,
    config: Any,
    context: WorkflowContext,
    session: Any | None = None,
) -> AsyncIterator[TextRuntimeStreamEvent]:
    yield TextRuntimeStatusEvent(stage="guided_exercise")
    queue: asyncio.Queue[str] = asyncio.Queue()

    def writer_factory() -> Any:
        def writer(payload: dict[str, str]) -> None:
            if payload.get("type") == "chunk":
                queue.put_nowait(str(payload.get("text") or ""))

        return writer

    response_llm = guided_exercise_response_llm(
        services,
        state,
        config,
        context,
        session=session,
    )
    skill_service = guided_exercise_skill_service(
        context,
        response_llm=response_llm,
        stream_writer_factory=writer_factory,
    )
    task = asyncio.create_task(skill_service.run_turn(state))
    try:
        while not task.done() or not queue.empty():
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if chunk:
                yield TextRuntimeChunkEvent(text=chunk)

        # Only on normal completion: collect the result and finalize. If the
        # consumer abandoned the stream (GeneratorExit at a yield above), control
        # jumps straight to the finally and none of this runs.
        delta = await task
        apply_state_delta(state, dict(delta))
        _apply_guided_exercise_tool_diagnostics(
            state,
            response_llm.run_context,
            fallback=response_llm.used_skill_tool_fallback,
        )
        response_text = str(state.get("response_text") or "")
        if not response_text:
            raise ValueError("guided_exercise returned an empty response.")
        final_state = await services.finalize_turn(
            state,
            response_text=response_text,
            config=config,
            runtime_mode="guided_exercise",
            response_style=str(state.get("response_style") or "guided_exercise"),
            selected_agent=GUIDED_EXERCISE_AGENT_NAME,
            sdk_duration_ms=response_llm.last_duration_ms,
            streamed=True,
        )
        yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
        yield TextRuntimeStateEvent(state=final_state)
    finally:
        # Guarantee the producer never outlives this generator: on early consumer
        # abandonment the drain loop exits before `await task`, orphaning it.
        # Cancelling and draining a cancelled task completes promptly, so this is
        # safe to await even while unwinding a GeneratorExit.
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _apply_guided_exercise_tool_diagnostics(
    state: Any,
    run_context: OpenAITextRunContext,
    *,
    fallback: bool,
) -> None:
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_guided_exercise_tool_expected": "load_guided_exercise_skill",
        "openai_guided_exercise_tool_calls": [
            call.tool_name for call in run_context.guided_exercise_skill_tool_calls
        ],
        "openai_guided_exercise_tool_fallback": fallback,
    }
    latest = run_context.latest_guided_exercise_skill_tool_result()
    if latest is not None:
        diagnostics.update(
            {
                "openai_guided_exercise_tool_exercise_type": latest.exercise_type,
                "openai_guided_exercise_tool_runtime_action": latest.runtime_action,
                "openai_guided_exercise_tool_step": latest.current_step_index,
            }
        )
    apply_state_delta(state, {"diagnostics": diagnostics})


def build_guided_exercise_route_handler(
    services_factory: TextRuntimeServicesFactory,
) -> RouteHandler:
    """Build the guided-exercise route handler."""

    async def execute(
        plan: TextRoutePlan,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AgentState:
        return await run_guided_exercise_turn(
            services_factory(),
            plan.state,
            config=config,
            context=context,
            streamed=False,
            session=session,
        )

    async def stream(
        plan: TextRoutePlan,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        async for event in run_guided_exercise_turn_stream(
            services_factory(),
            plan.state,
            config=config,
            context=context,
            session=session,
        ):
            yield event

    return RouteHandler(execute=execute, stream=stream)


__all__ = [
    "build_guided_exercise_route_handler",
    "guided_exercise_skill_service",
    "run_guided_exercise_turn",
    "run_guided_exercise_turn_stream",
]
