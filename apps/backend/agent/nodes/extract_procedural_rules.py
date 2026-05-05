"""LangGraph adapter for the procedural-rule extraction service.

Thin wrapper around
:func:`agent.memory.extraction_service.extract_procedural_rules` that
translates between the LangGraph runtime context and the service
inputs, then formats the outcome into the per-turn diagnostics state
delta. Per AGENTS.md §6, all extraction logic — LLM call, prompt
building, policy dispatch — lives in the service module rather than
here.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.memory.extraction_service import extract_procedural_rules
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_extract_procedural_rules_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Run the procedural-rule extractor and return its diagnostics delta.

    Args:
        state (AgentState): Current graph state after response generation.
        runtime (Runtime[WorkflowContext]): LangGraph runtime carrying memory
            dependencies.

    Returns:
        dict[str, Any]: State delta containing procedural-writer diagnostics.
    """

    outcome = await extract_procedural_rules(
        state,
        llm_client=runtime.context.llm_client,
        memory_store=runtime.context.memory_store,
        memory_mode=runtime.context.memory_mode,
        session_buffer=runtime.context.session_memory_buffer,
    )
    return {
        "diagnostics": {
            "extract_procedural_ms": outcome.duration_ms,
            "procedural_writes": outcome.procedural_writes,
            "procedural_candidates": outcome.procedural_candidates,
            "procedural_commit_now_candidates": outcome.procedural_commit_now_candidates,
            "procedural_session_end_holds": outcome.procedural_session_end_holds,
            "procedural_policy_drops": outcome.procedural_policy_drops,
            "extract_procedural_reason": outcome.reason,
        }
    }
