"""Realignment response node."""

from __future__ import annotations

from agent.nodes.therapeutic_mode_registry import (
    run_registered_therapeutic_mode_response,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient


async def run_realignment_response(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Generate a realignment-style reply.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated agent state with a realignment reply.
    """

    return await run_registered_therapeutic_mode_response(
        state,
        mode="realignment",
        llm_client=llm_client,
    )
