"""Therapeutic response node for the MVP graph."""

from __future__ import annotations

from typing import cast

from agent.models import ResponseKind
from agent.prompts import (
    build_therapeutic_response_prompt,
    build_therapeutic_system_prompt,
)
from agent.prompts.catalog import Modality
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


def _fallback_supportive_response(state: AgentState) -> str:
    """Return the deterministic fallback for normal therapeutic replies."""

    if state.get("session_stage") == "closing":
        return (
            "It sounds like the most important thing from this conversation is that what you’re carrying has felt heavy, "
            "and you’ve started putting a little more shape around what you need. If it helps, the next step is to stay "
            "with one small thing that felt most grounding or clarifying today. We can pick this up again whenever you want."
        )
    return "I’m here with you. Tell me a bit more about what feels hardest right now."


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
        state["response_text"] = _select_safety_check_message(state)
        return state

    state["mode"] = "support"
    modalities = cast(
        tuple[Modality, ...],
        tuple(state.get("active_modalities", ["motivational_interviewing"])),
    )

    if llm_client is not None:
        try:
            state["response_text"] = await llm_client.generate_text(
                prompt=build_therapeutic_response_prompt(state),
                system_instruction=build_therapeutic_system_prompt(
                    modalities=modalities,
                ),
                temperature=0.4,
            )
            return state
        except Exception:
            # Fall back cleanly so one provider failure does not break the local workflow.
            pass

    state["response_text"] = _fallback_supportive_response(state)
    return state
