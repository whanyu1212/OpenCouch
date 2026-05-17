"""Compatibility wrapper for turn-level memory retrieval."""

from __future__ import annotations

from typing import Any

from agent.memory.load_turn import build_load_memory_delta
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_load_memory_node(
    state: AgentState,
    runtime: Any,
) -> dict[str, Any]:
    """Retrieve turn memory using a runtime object with a ``context`` field."""

    context = getattr(runtime, "context")
    if not isinstance(context, WorkflowContext):
        raise TypeError("run_load_memory_node requires runtime.context.")
    return await build_load_memory_delta(state, context)


__all__ = ["build_load_memory_delta", "run_load_memory_node"]
