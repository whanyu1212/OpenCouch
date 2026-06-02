"""Guided exercise execution helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from llm.base import BaseLLMClient

from agent.flows.guided_exercise.adapters import guided_exercise_response_llm
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.services import TextRuntimeServices
from agent.runtime.state_ops import apply_state_delta
from agent.runtime.types import (
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.runtime.workflow_context import WorkflowContext
from agent.skills.guided_exercises.engine.lifecycle import GuidedExerciseSkillService
from agent.specialists.guided_exercise import GUIDED_EXERCISE_AGENT_NAME


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
    while not task.done() or not queue.empty():
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        if chunk:
            yield TextRuntimeChunkEvent(text=chunk)

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


__all__ = [
    "guided_exercise_skill_service",
    "run_guided_exercise_turn",
    "run_guided_exercise_turn_stream",
]
