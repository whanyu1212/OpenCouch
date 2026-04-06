"""Realignment response node."""

from __future__ import annotations

from agent.models import ResponseKind
from agent.prompts import (
    build_realignment_response_prompt,
    build_realignment_system_prompt,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient


def _fallback_realignment_response() -> str:
    """Return a deterministic realignment reply."""

    return (
        "You're right, that missed the point. Let me slow down and try again more carefully. "
        "What feels most important or most off about what I just said?"
    )


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

    state["response_type"] = ResponseKind.THERAPEUTIC
    state["mode"] = "realignment"

    if llm_client is not None:
        try:
            state["response_text"] = await llm_client.generate_text(
                prompt=build_realignment_response_prompt(state),
                system_instruction=build_realignment_system_prompt(),
                temperature=0.3,
            )
            return state
        except Exception:
            pass

    state["response_text"] = _fallback_realignment_response()
    return state
