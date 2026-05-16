"""Grounded factual answer node for explicit lookup requests."""

from __future__ import annotations

import time
from typing import Any

from langgraph.runtime import Runtime

from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded_search import FactualLookupStatus, answer_factual_lookup


def _base_delta(
    response_text: str,
    *,
    status: FactualLookupStatus,
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
        "grounded_lookup": {"status": status},
        "response_style": "grounded_lookup",
        "response_text": response_text,
        "diagnostics": {"grounded_lookup_ms": elapsed_ms(started_at)},
    }


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

    started_at = time.monotonic()
    grounded_lookup = state.get("grounded_lookup", {}) or {}
    query = str(grounded_lookup.get("query") or "").strip()
    if not query:
        raise ValueError("grounded_answer_node requires grounded_lookup.query.")
    llm_client = runtime.context.llm_client

    if llm_client is None:
        raise RuntimeError("grounded_answer_node requires an LLM client.")

    answer, status = await answer_factual_lookup(
        state,
        llm_client=llm_client,
        query=query,
    )
    if answer:
        return _base_delta(answer, status=status, started_at=started_at)
    text = "I couldn't verify that from reliable sources, so I don't want to guess."
    return _base_delta(text, status="no_verified_answer", started_at=started_at)
