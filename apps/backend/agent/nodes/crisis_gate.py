"""LLM-only crisis gate compatibility adapter."""

from __future__ import annotations

from typing import Any

from agent.gates.safety.turn_gate import assess_crisis_gate
from agent.graph_constants import (
    CRISIS_RESOURCE_LOOKUP_NODE,
    TURN_DISPATCH_NODE,
    CrisisGateNextNode,
)
from agent.runtime.command import RuntimeCommand
from agent.state import AgentState


async def run_crisis_gate_node(
    state: AgentState,
    runtime: Any,
) -> RuntimeCommand[CrisisGateNextNode]:
    """Run the crisis gate for the current turn.

    Args:
        state: Current graph state for the turn being processed.
        runtime: Runtime object carrying the workflow context.

    Returns:
        State update plus the next node to run.
    """

    result = await assess_crisis_gate(
        state,
        llm_client=runtime.context.llm_client,
    )
    assessment = result.assessment
    next_node: CrisisGateNextNode = (
        CRISIS_RESOURCE_LOOKUP_NODE
        if assessment.needs_crisis_response
        else TURN_DISPATCH_NODE
    )
    return RuntimeCommand(update=result.delta, goto=next_node)
