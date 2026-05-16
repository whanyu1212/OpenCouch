"""LLM-only crisis gate node for the OpenCouch graph."""

from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.gates.safety.turn_gate import assess_crisis_gate
from agent.graph_constants import (
    CRISIS_RESOURCE_LOOKUP_NODE,
    TURN_DISPATCH_NODE,
    CrisisGateNextNode,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_crisis_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[CrisisGateNextNode]:
    """Run the crisis gate for the current turn.

    Args:
        state: Current graph state for the turn being processed.
        runtime: LangGraph runtime carrying the workflow context.

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
    return Command(update=result.delta, goto=next_node)
