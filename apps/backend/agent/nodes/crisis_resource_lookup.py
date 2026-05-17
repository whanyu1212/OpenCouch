"""Crisis resource lookup node for the OpenCouch graph."""

from __future__ import annotations

from typing import Any

from agent.crisis_branch import build_crisis_resource_lookup_delta
from agent.state import AgentState


async def run_crisis_resource_lookup_node(
    state: AgentState,
    runtime: Any,
) -> dict[str, Any]:
    """Resolve local crisis-resource state for the current crisis turn.

    Args:
        state: Current graph state after crisis classification.
        runtime: Runtime object carrying the workflow context.

    Returns:
        A partial state update containing inferred location, verified resources,
        and lookup status.
    """

    return await build_crisis_resource_lookup_delta(state, runtime.context)
