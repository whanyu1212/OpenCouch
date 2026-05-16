"""Load-memory node for the OpenCouch graph.

Thin LangGraph wrapper around turn-level memory retrieval services.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.runtime import Runtime

from agent.memory.recall import LoadMemoryResult, load_memory_for_turn
from agent.memory.modes import MemoryMode
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id

logger = logging.getLogger(__name__)


async def run_load_memory_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Retrieve turn memory and shape the graph state delta.

    Prefers awaiting the runtime's speculative pre-fetch when present so this
    node can return without paying memory-retrieval latency on the critical
    path. Falls back to a fresh retrieval call when no pre-fetch was scheduled
    or the pre-fetch raised — speculation must never fail the turn.

    Args:
        state (AgentState): Current workflow state.
        runtime (Runtime[WorkflowContext]): LangGraph runtime with memory dependencies.

    Returns:
        dict[str, Any]: State delta with working memory, session/procedural
            memory metadata, and diagnostics.
    """

    if runtime.context.memory_mode == MemoryMode.INCOGNITO:
        return {
            "working_memory": [],
            "session_memory": {
                **state.get("session_memory", {}),
                "summary": "Guest session without long-term memory.",
            },
            "procedural_profile": {
                "procedural_rules": [],
                "proactive_recall_enabled": False,
            },
        }

    transcript = state.get("transcript", [])
    owner_id = resolve_owner_id(state)
    result, speculation_used, speculation_wait_ms = await _resolve_memory_result(
        runtime=runtime,
        owner_id=owner_id,
        query=state["message"],
        is_first_turn=len(transcript) == 1,
    )

    diagnostics = dict(result.diagnostics)
    diagnostics["load_memory_speculation_used"] = speculation_used
    diagnostics["load_memory_speculation_wait_ms"] = round(speculation_wait_ms, 2)

    return {
        "working_memory": list(result.working_memory),
        "session_memory": {
            **state.get("session_memory", {}),
            "summary": result.summary,
        },
        "procedural_profile": {
            "procedural_rules": result.procedural_rules,
            "proactive_recall_enabled": result.proactive_recall_enabled,
        },
        "diagnostics": diagnostics,
    }


async def _resolve_memory_result(
    *,
    runtime: Runtime[WorkflowContext],
    owner_id: str,
    query: str,
    is_first_turn: bool,
) -> tuple[LoadMemoryResult, bool, float]:
    """Resolve the turn memory result, preferring the speculative pre-fetch.

    Args:
        runtime: LangGraph runtime carrying the workflow context.
        owner_id: Resolved memory owner.
        query: Current user message used as the retrieval query.
        is_first_turn: Whether this is the first turn of the session.

    Returns:
        Tuple of ``(result, speculation_used, speculation_wait_ms)``. The
        ``speculation_used`` flag indicates whether the pre-fetched task was
        successfully consumed; ``speculation_wait_ms`` is the wall-clock spent
        awaiting it (near zero when the pre-fetch finished before this node
        ran, larger when the gates outran the pre-fetch).
    """

    pre_fetched = runtime.context.pre_fetched_memory
    if pre_fetched is not None:
        await_start = time.monotonic()
        try:
            result = await pre_fetched
            return result, True, elapsed_ms(await_start)
        except Exception:
            logger.warning(
                "load_memory_node: speculative pre-fetch failed; falling back "
                "to fresh retrieval for owner=%r",
                owner_id,
                exc_info=True,
            )

    result = await load_memory_for_turn(
        memory_store=runtime.context.memory_store,
        embedding_provider=runtime.context.embedding_provider,
        owner_id=owner_id,
        query=query,
        is_first_turn=is_first_turn,
    )
    return result, False, 0.0
