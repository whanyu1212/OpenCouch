"""Crisis response node for the MVP graph."""

from __future__ import annotations

from agent.models import ModeType, ResponseKind
from agent.prompts import (
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient


def _fallback_crisis_response(state: AgentState) -> str:
    """Return a deterministic crisis message when no model reply is available."""

    crisis = state["crisis"]

    if crisis.level >= 3:
        return (
            "I’m really glad you said this. It sounds like you may be in immediate danger. "
            "Please contact emergency services or a crisis hotline right now, or reach out "
            "to someone nearby who can stay with you."
        )

    return (
        "I’m sorry you’re carrying this right now. What you said sounds serious, and I "
        "want to respond carefully. Please reach out to a crisis hotline or a trusted "
        "person who can be with you while we focus on your safety."
    )


async def run_crisis_response(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Return an empathetic interruption when the crisis gate detects risk.

    Args:
        state: Shared agent state after crisis routing.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated agent state with a crisis reply.
    """

    state["mode"] = "crisis_response"
    state["mode_type"] = ModeType.CRISIS
    state["response_guidance"] = ""
    state["response_type"] = ResponseKind.CRISIS

    if llm_client is not None:
        try:
            state["response_text"] = await llm_client.generate_text(
                prompt=build_crisis_response_prompt(state),
                system_instruction=build_crisis_response_system_prompt(),
                temperature=0.2,
            )
            return state
        except Exception:
            # Fall back cleanly so the safety path does not fail closed on provider issues.
            pass

    state["response_text"] = _fallback_crisis_response(state)

    return state
