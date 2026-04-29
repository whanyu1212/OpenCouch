"""Guided exercise response style - public compatibility surface."""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.exercises.node import (
    run_guided_exercise_response_node as _run_guided_exercise_response_node,
)

__all__ = ["get_stream_writer", "run_guided_exercise_response_node"]


async def run_guided_exercise_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Drive a multi-turn guided exercise.

    Args:
        state: Current graph state.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response and state delta for the exercise turn.
    """

    return await _run_guided_exercise_response_node(
        state,
        runtime,
        stream_writer_factory=get_stream_writer,
    )
