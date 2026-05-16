"""Shared execution for operational turn branches."""

from __future__ import annotations

import time
from typing import Any

from agent.gates.memory_control import execute_memory_control_action
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.tools.grounded_search import FactualLookupStatus, answer_factual_lookup


def memory_control_response_delta(
    response_text: str,
    *,
    started_at: float,
) -> dict[str, Any]:
    """Return the shared response delta for memory-control turns."""

    return {
        "route": "memory_control",
        "response_style": "memory_control",
        "response_text": response_text,
        "diagnostics": {"memory_control_ms": elapsed_ms(started_at)},
    }


async def build_memory_control_delta(
    state: AgentState,
    context: WorkflowContext,
) -> dict[str, Any]:
    """Execute a memory-control action and return its state delta."""

    started_at = time.monotonic()
    result = await execute_memory_control_action(state, context)
    delta = memory_control_response_delta(result.response_text, started_at=started_at)
    delta["memory_control"] = result.memory_control
    if result.procedural_profile is not None:
        delta["procedural_profile"] = result.procedural_profile
    return delta


def grounded_lookup_response_delta(
    response_text: str,
    *,
    status: FactualLookupStatus,
    started_at: float,
) -> dict[str, Any]:
    """Return the shared response delta for grounded lookup turns."""

    return {
        "route": "grounded_lookup",
        "grounded_lookup": {"status": status},
        "response_style": "grounded_lookup",
        "response_text": response_text,
        "diagnostics": {"grounded_lookup_ms": elapsed_ms(started_at)},
    }


async def build_grounded_lookup_delta(
    state: AgentState,
    context: WorkflowContext,
) -> dict[str, Any]:
    """Answer a grounded factual lookup and return its state delta."""

    started_at = time.monotonic()
    grounded_lookup = state.get("grounded_lookup", {}) or {}
    query = str(grounded_lookup.get("query") or "").strip()
    if not query:
        raise ValueError("grounded lookup requires grounded_lookup.query.")
    llm_client = context.llm_client

    if llm_client is None:
        raise RuntimeError("grounded lookup requires an LLM client.")

    answer, status = await answer_factual_lookup(
        state,
        llm_client=llm_client,
        query=query,
    )
    if answer:
        return grounded_lookup_response_delta(
            answer,
            status=status,
            started_at=started_at,
        )
    text = "I couldn't verify that from reliable sources, so I don't want to guess."
    return grounded_lookup_response_delta(
        text,
        status="no_verified_answer",
        started_at=started_at,
    )
