"""LangGraph adapter for the semantic-fact extraction service.

Thin wrapper around :func:`agent.memory.extraction_service.extract_semantic_facts`
that translates between the LangGraph runtime context and the service
inputs, then formats the outcome into the per-turn diagnostics state
delta. Per AGENTS.md §6, all extraction logic — LLM call, prompt
building, deterministic backstops, policy/dedup dispatch — lives in
the service module rather than here.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from agent.memory.extraction_service import extract_semantic_facts
from agent.runtime_context import WorkflowContext
from agent.state import AgentState


async def run_extract_semantic_facts_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Run the semantic-fact extractor and return its diagnostics delta.

    Args:
        state (AgentState): Current graph state after response generation.
        runtime (Runtime[WorkflowContext]): LangGraph runtime carrying memory
            dependencies.

    Returns:
        dict[str, Any]: State delta containing extractor diagnostics.
    """

    outcome = await extract_semantic_facts(
        state,
        llm_client=runtime.context.llm_client,
        memory_store=runtime.context.memory_store,
        memory_mode=runtime.context.memory_mode,
        embedding_provider=runtime.context.embedding_provider,
        session_buffer=runtime.context.session_memory_buffer,
    )
    return {
        "diagnostics": {
            "extract_facts_ms": outcome.duration_ms,
            "semantic_writes": outcome.semantic_writes,
            "semantic_bumps": outcome.semantic_bumps,
            "semantic_candidates": outcome.semantic_candidates,
            "semantic_commit_now_candidates": outcome.semantic_commit_now_candidates,
            "semantic_session_end_holds": outcome.semantic_session_end_holds,
            "semantic_repeat_required": outcome.semantic_repeat_required,
            "semantic_policy_drops": outcome.semantic_policy_drops,
            "extract_facts_reason": outcome.reason,
        }
    }
