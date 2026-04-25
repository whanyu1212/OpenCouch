"""Grounded factual answer node for explicit lookup requests."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseCategory
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded_lookup import GroundedLookupStatus, answer_grounded_lookup


def _base_delta(
    response_text: str,
    *,
    status: GroundedLookupStatus,
    started_at: float,
) -> dict[str, Any]:
    """Return the shared response delta for grounded lookup turns.

    Args:
        response_text: User-facing answer text.
        status: Grounded lookup status for diagnostics and observability.
        started_at: Monotonic start timestamp.

    Returns:
        Partial graph state update for the grounded answer node.
    """

    return {
        "route": "grounded_lookup",
        "grounded_lookup_status": status,
        "response_style": "grounded_lookup",
        "response_style_source": "grounded_lookup_gate",
        "response_style_type": ModeType.OPERATIONAL,
        "response_kind": ResponseCategory.THERAPEUTIC,
        "response_text": response_text,
        "diagnostics": {
            "grounded_lookup_ms": round((time.monotonic() - started_at) * 1000, 2)
        },
    }


async def run_grounded_answer_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Answer an explicit factual lookup request with search grounding.

    Args:
        state: Current graph state with ``grounded_lookup_query`` set by the
            grounded lookup gate.
        runtime: LangGraph runtime carrying the provider client.

    Returns:
        Partial state update containing an operational response and lookup
        status.
    """

    started_at = time.monotonic()
    query = (state.get("grounded_lookup_query") or state.get("message", "")).strip()
    llm_client = runtime.context.llm_client

    if llm_client is None:
        return _base_delta(
            "I can't look that up from here right now, so I don't want to guess.",
            status="search_unavailable",
            started_at=started_at,
        )

    answer, status = await answer_grounded_lookup(
        state,
        llm_client=llm_client,
        query=query,
    )
    if status == "answered":
        return _base_delta(answer, status=status, started_at=started_at)
    if answer:
        return _base_delta(answer, status=status, started_at=started_at)
    if status == "search_failed":
        text = "I couldn't complete the lookup right now, so I don't want to guess."
    else:
        text = "I couldn't verify that from reliable sources, so I don't want to guess."
    return _base_delta(text, status=status, started_at=started_at)
