"""Compatibility adapter for guided therapeutic exercises."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from agent.therapeutic.exercises.responses import (
    StreamWriterFactory,
    _noop_stream_writer_factory,
)
from agent.therapeutic.exercises.runner import ExerciseRunner


async def run_guided_exercise_response_node(
    state: AgentState,
    runtime: Any,
    *,
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Drive a multi-turn guided exercise.

    Args:
        state: Current graph state.
        runtime: Runtime object carrying configured dependencies.
        stream_writer_factory: Factory that returns the current stream writer.

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
