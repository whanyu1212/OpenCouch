"""Crisis resource lookup node for the OpenCouch graph."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded_search import (
    CrisisResourceLookupStatus,
    find_crisis_resources,
)

logger = logging.getLogger(__name__)


async def run_crisis_resource_lookup_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Resolve local crisis-resource state for the current crisis turn.

    Args:
        state: Current graph state after crisis classification.
        runtime: LangGraph runtime carrying the workflow context.

    Returns:
        A partial state update containing inferred location, verified resources,
        and lookup status.
    """

    inferred_location = ""
    found_resources: list[dict[str, str]] = []
    resource_lookup_status: CrisisResourceLookupStatus = "not_attempted"

    llm_client = runtime.context.llm_client
    if llm_client is not None:
        try:
            (
                inferred_location,
                found_resources,
                resource_lookup_status,
            ) = await find_crisis_resources(state, llm_client=llm_client)
        except Exception:
            logger.warning(
                "Crisis resource lookup failed; continuing without resources.",
                exc_info=True,
            )
            resource_lookup_status = "search_failed"

    return {
        "inferred_location": inferred_location,
        "found_resources": found_resources,
        "resource_lookup_status": resource_lookup_status,
    }
