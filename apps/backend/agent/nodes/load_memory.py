"""Load-memory node for the OpenCouch graph.

Thin LangGraph wrapper around turn-level memory retrieval services.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.memory.load_memory_service import load_memory_for_turn
from agent.memory.modes import MemoryMode
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id


async def run_load_memory_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Retrieve turn memory and shape the graph state delta.

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
    result = await load_memory_for_turn(
        memory_store=runtime.context.memory_store,
        embedding_provider=runtime.context.embedding_provider,
        owner_id=resolve_owner_id(state),
        query=state["message"],
        is_first_turn=len(transcript) == 1,
    )

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
        "diagnostics": result.diagnostics,
    }
