"""Memory-control node for explicit user memory commands."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from agent.turn_branches import build_memory_control_delta


async def run_memory_control_node(
    state: AgentState,
    runtime: Any,
) -> dict[str, Any]:
    """Execute an explicit memory-control action.

    Args:
        state (AgentState): Current graph state with ``memory_control.action`` set
            by the gate.
        runtime: Runtime object carrying memory dependencies.

    Returns:
        dict[str, Any]: Partial state update containing an operational reply and
            any memory-control state changes.
    """

    return await build_memory_control_delta(state, runtime.context)
