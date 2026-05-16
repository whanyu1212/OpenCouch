"""LangGraph adapter for guided therapeutic exercises."""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.exercises.responses import StreamWriterFactory
from agent.therapeutic.exercises.runner import ExerciseRunner


async def run_guided_exercise_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
    *,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> dict[str, Any]:
    """Drive a multi-turn guided exercise.

    Args:
        state: Current graph state.
        runtime: LangGraph runtime carrying configured dependencies.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Response and state delta for the exercise turn.
    """

    control_llm = runtime.context.llm_client
    runner = ExerciseRunner(
        classifier_llm=control_llm,
        response_llm=runtime.context.response_llm or control_llm,
        memory_store=runtime.context.memory_store,
        memory_mode=runtime.context.memory_mode,
        stream_writer_factory=stream_writer_factory,
    )
    return await runner.run(state)
