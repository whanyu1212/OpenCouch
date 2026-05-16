"""Grounded factual answer node for explicit lookup requests."""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.turn_branches import build_grounded_lookup_delta


async def run_grounded_answer_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Answer an explicit factual lookup request with search grounding.

    Args:
        state: Current graph state with ``grounded_lookup.query`` set by the
            grounded lookup gate.
        runtime: LangGraph runtime carrying the provider client.

    Returns:
        Partial state update containing an operational response and lookup
        status.
    """

    return await build_grounded_lookup_delta(state, runtime.context)
