"""Turn-level dispatch node for non-crisis turns."""

from __future__ import annotations

import time
from typing import Any

from agent.graph_constants import (
    GROUNDED_ANSWER_NODE,
    LOAD_MEMORY_NODE,
    MEMORY_CONTROL_NODE,
    TurnDispatchNextNode,
)
from agent.observability.timing import elapsed_ms
from agent.runtime.command import RuntimeCommand
from agent.state import AgentState
from agent.turn_dispatch import build_turn_dispatch_update, plan_turn_route


async def run_turn_dispatch_node(
    state: AgentState,
    runtime: Any,
) -> RuntimeCommand[TurnDispatchNextNode]:
    """Route a safe user turn to the next lifecycle node.

    Args:
        state (AgentState): Current graph state after crisis classification.
        runtime: Runtime object carrying workflow dependencies.

    Returns:
        RuntimeCommand[TurnDispatchNextNode]: State update plus the next node to run.
    """

    start = time.monotonic()
    plan = await plan_turn_route(
        state,
        llm_client=runtime.context.llm_client,
    )
    update = build_turn_dispatch_update(
        state,
        plan,
        duration_ms=elapsed_ms(start),
    )

    if plan.route == "memory_control":
        next_node: TurnDispatchNextNode = MEMORY_CONTROL_NODE
    elif plan.route == "grounded_lookup":
        next_node = GROUNDED_ANSWER_NODE
    else:
        next_node = LOAD_MEMORY_NODE

    return RuntimeCommand(update=update, goto=next_node)
