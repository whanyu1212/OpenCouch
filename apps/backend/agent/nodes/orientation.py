"""Orientation response node."""

from __future__ import annotations

from agent.models import ResponseKind
from agent.prompts import (
    build_orientation_response_prompt,
    build_orientation_system_prompt,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient


def _fallback_orientation_response() -> str:
    """Return a deterministic orientation reply."""

    return (
        "I can help you talk through difficult moments, reflect on patterns, and try simple "
        "self-help exercises when useful. I’m not a therapist or emergency service, but I "
        "can be a steady place to start. What feels most useful to focus on today?"
    )


async def run_orientation_response(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Generate an orientation-style opening reply.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated agent state with an orientation reply.
    """

    state["response_type"] = ResponseKind.THERAPEUTIC
    state["mode"] = "orientation"

    if llm_client is not None:
        try:
            state["response_text"] = await llm_client.generate_text(
                prompt=build_orientation_response_prompt(state),
                system_instruction=build_orientation_system_prompt(),
                temperature=0.3,
            )
            return state
        except Exception:
            pass

    state["response_text"] = _fallback_orientation_response()
    return state
