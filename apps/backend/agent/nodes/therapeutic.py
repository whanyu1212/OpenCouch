"""Therapeutic response node for the MVP graph."""

from __future__ import annotations

from agent.models import ModeType, ResponseKind
from agent.nodes.therapeutic_mode_registry import (
    run_registered_therapeutic_mode_response,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient

CLARIFICATION_TEMPLATES = {
    "high_distress": (
        "I want to pause and check on your safety before we go further. "
        "Are you feeling unsafe or thinking about hurting yourself right now?"
    ),
    "passive_ideation": (
        "I want to check something important with you directly. "
        "When you say that, are you thinking about hurting yourself or not wanting to be alive right now?"
    ),
    "general": (
        "I want to check something important before we keep going. "
        "Are you feeling unsafe or thinking about hurting yourself right now?"
    ),
}


def _select_safety_check_message(state: AgentState) -> str:
    """Select a bounded safety-check template based on crisis context."""

    reason = state["crisis"].reason.lower()
    if "high-distress" in reason or "high distress" in reason:
        return CLARIFICATION_TEMPLATES["high_distress"]
    if "self-harm-adjacent" in reason or "passive" in reason:
        return CLARIFICATION_TEMPLATES["passive_ideation"]
    return CLARIFICATION_TEMPLATES["general"]


async def run_therapeutic_response(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Return a supportive therapeutic response.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated agent state with a therapeutic reply.
    """

    crisis = state["crisis"]
    state["response_type"] = ResponseKind.THERAPEUTIC

    if crisis.needs_clarification:
        state["mode"] = "safety_check"
        state["mode_type"] = ModeType.OPERATIONAL
        state["response_text"] = _select_safety_check_message(state)
        return state

    return await run_registered_therapeutic_mode_response(
        state,
        mode="supportive_conversation",
        llm_client=llm_client,
    )
