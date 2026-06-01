"""Runner-turn memory context helpers for the OpenAI text runtime."""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

from agent.memory.modes import MemoryMode
from agent.memory.retrieval.service import LoadMemoryResult, load_memory_for_turn
from agent.observability.timing import elapsed_ms
from agent.runtime_context import PrefetchedTurnMemory, WorkflowContext
from agent.state import AgentState, resolve_owner_id

logger = logging.getLogger(__name__)

MemorySpeculationStatus = Literal[
    "not_scheduled",
    "used",
    "fallback_after_error",
    "discarded_mismatch",
]


async def build_turn_memory_delta(
    state: AgentState,
    context: WorkflowContext,
) -> dict[str, Any]:
    """Retrieve durable memory and shape the runner-turn state delta."""

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
    result, speculation_status, speculation_wait_ms = await _resolve_memory_result(
        context=context,
        owner_id=owner_id,
        query=state["message"],
        is_first_turn=len(transcript) == 1,
    )

    diagnostics = dict(result.diagnostics)
    diagnostics["load_memory_speculation_used"] = speculation_status == "used"
    diagnostics["load_memory_speculation_status"] = speculation_status
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
) -> tuple[LoadMemoryResult, MemorySpeculationStatus, float]:
    """Resolve turn memory, preferring speculative prefetch when available."""

    pre_fetched = context.pre_fetched_memory
    if pre_fetched is not None:
        if not _prefetch_matches(
            pre_fetched,
            owner_id=owner_id,
            query=query,
            is_first_turn=is_first_turn,
        ):
            pre_fetched.cancel_if_pending()
            result = await _load_memory_fresh(
                context=context,
                owner_id=owner_id,
                query=query,
                is_first_turn=is_first_turn,
            )
            return result, "discarded_mismatch", 0.0

        await_start = time.monotonic()
        try:
            result = await pre_fetched.task
            return result, "used", elapsed_ms(await_start)
        except Exception:
            logger.warning(
                "turn memory prefetch failed; falling back to fresh retrieval "
                "for owner=%r",
                owner_id,
                exc_info=True,
            )

            result = await _load_memory_fresh(
                context=context,
                owner_id=owner_id,
                query=query,
                is_first_turn=is_first_turn,
            )
            return result, "fallback_after_error", 0.0

    result = await _load_memory_fresh(
        context=context,
        owner_id=owner_id,
        query=query,
        is_first_turn=is_first_turn,
    )
    return result, "not_scheduled", 0.0


def _prefetch_matches(
    pre_fetched: PrefetchedTurnMemory,
    *,
    owner_id: str,
    query: str,
    is_first_turn: bool,
) -> bool:
    """Return whether the speculative result belongs to this turn."""

    return pre_fetched.matches(
        owner_id=owner_id,
        query=query,
        is_first_turn=is_first_turn,
    )


async def _load_memory_fresh(
    *,
    context: WorkflowContext,
    owner_id: str,
    query: str,
    is_first_turn: bool,
) -> LoadMemoryResult:
    """Load turn memory without using speculative state."""

    return await load_memory_for_turn(
        memory_store=context.memory_store,
        embedding_provider=context.embedding_provider,
        owner_id=owner_id,
        query=query,
        is_first_turn=is_first_turn,
    )


__all__ = ["build_turn_memory_delta"]
