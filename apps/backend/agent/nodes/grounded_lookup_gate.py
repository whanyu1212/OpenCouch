"""Grounded-lookup routing gate for explicit factual search requests."""

from __future__ import annotations

import time

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.graph_constants import (
    GROUNDED_ANSWER_NODE,
    LOAD_MEMORY_NODE,
    GroundedLookupGateNextNode,
)
from agent.grounded_lookup.router import resolve_grounded_lookup_action
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_grounded_lookup_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[GroundedLookupGateNextNode]:
    """Route explicit factual lookup requests before therapeutic generation.

    Args:
        state (AgentState): Current graph state after memory-control routing.
        runtime (Runtime[WorkflowContext]): LangGraph runtime carrying workflow
            dependencies.

    Returns:
        Command[GroundedLookupGateNextNode]: State update plus the next node to
            run.
    """

    start = time.monotonic()
    (
        action,
        classifier_path,
        llm_failure_occurred,
    ) = await resolve_grounded_lookup_action(
        state,
        llm_client=runtime.context.llm_client,
    )
    diagnostics = {
        "grounded_lookup_gate_ms": elapsed_ms(start),
        "grounded_lookup_classifier_path": classifier_path,
        "grounded_lookup_llm_failure_occurred": llm_failure_occurred,
    }

    if action is None:
        return Command(
            update={
                "grounded_lookup": {"query": "", "status": "not_attempted"},
                "diagnostics": diagnostics,
            },
            goto=LOAD_MEMORY_NODE,
        )

    return Command(
        update={
            "route": "grounded_lookup",
            "grounded_lookup": {"query": action["query"], "status": "not_attempted"},
            "diagnostics": diagnostics,
        },
        goto=GROUNDED_ANSWER_NODE,
    )
