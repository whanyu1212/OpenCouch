"""Memory-control node for explicit user memory commands."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.gates.memory_control import execute_memory_control_action
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


def _base_delta(response_text: str, *, started_at: float) -> dict[str, Any]:
    """Return the shared response delta for memory-control turns.

    Args:
        response_text (str): User-facing operational reply.
        started_at (float): Monotonic start timestamp.

    Returns:
        dict[str, Any]: Partial graph state update for memory-control turns.
    """

    return {
        "route": "memory_control",
        "response_style": "memory_control",
        "response_text": response_text,
        "diagnostics": {"memory_control_ms": elapsed_ms(started_at)},
    }


async def run_memory_control_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Execute an explicit memory-control action.

    Args:
        state (AgentState): Current graph state with ``memory_control.action`` set
            by the gate.
        runtime (Runtime[WorkflowContext]): LangGraph runtime carrying memory
            dependencies.

    Returns:
        dict[str, Any]: Partial state update containing an operational reply and
            any memory-control state changes.
    """

    started_at = time.monotonic()
    result = await execute_memory_control_action(state, runtime.context)
    delta = _base_delta(result.response_text, started_at=started_at)
    delta["memory_control"] = result.memory_control
    if result.procedural_profile is not None:
        delta["procedural_profile"] = result.procedural_profile
    return delta
