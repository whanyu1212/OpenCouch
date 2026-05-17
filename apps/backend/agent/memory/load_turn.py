"""Turn-level memory retrieval helpers for text runtimes."""

from __future__ import annotations

import logging
import time
from typing import Any

from agent.memory.modes import MemoryMode
from agent.memory.recall import LoadMemoryResult, load_memory_for_turn
from agent.observability.timing import elapsed_ms
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id

logger = logging.getLogger(__name__)


async def build_load_memory_delta(
    state: AgentState,
    context: WorkflowContext,
) -> dict[str, Any]:
    """Retrieve turn memory and shape a state delta for any text runtime."""

    if context.memory_mode == MemoryMode.INCOGNITO:
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
        context=context,
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
    context: WorkflowContext,
    owner_id: str,
    query: str,
    is_first_turn: bool,
) -> tuple[LoadMemoryResult, bool, float]:
    """Resolve turn memory, preferring speculative prefetch when available."""

    pre_fetched = context.pre_fetched_memory
    if pre_fetched is not None:
        await_start = time.monotonic()
        try:
            result = await pre_fetched
            return result, True, elapsed_ms(await_start)
        except Exception:
            logger.warning(
                "turn memory prefetch failed; falling back to fresh retrieval "
                "for owner=%r",
                owner_id,
                exc_info=True,
            )

    result = await load_memory_for_turn(
        memory_store=context.memory_store,
        embedding_provider=context.embedding_provider,
        owner_id=owner_id,
        query=query,
        is_first_turn=is_first_turn,
    )
    return result, False, 0.0


__all__ = ["build_load_memory_delta"]
