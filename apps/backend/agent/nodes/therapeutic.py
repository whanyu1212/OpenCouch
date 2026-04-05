"""Minimal therapeutic response node for the MVP graph."""

from agent.models import ResponseKind
from agent.state import AgentState

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


def _select_clarification_message(state: AgentState) -> str:
    """Select a bounded clarification template based on crisis context."""

    reason = state["crisis"].reason.lower()
    if "high-distress" in reason or "high distress" in reason:
        return CLARIFICATION_TEMPLATES["high_distress"]
    if "self-harm-adjacent" in reason or "passive" in reason:
        return CLARIFICATION_TEMPLATES["passive_ideation"]
    return CLARIFICATION_TEMPLATES["general"]


async def run_therapeutic_response(state: AgentState) -> AgentState:
    """Return a placeholder supportive response.

    If the crisis gate flagged ambiguity, start with a safety-oriented clarifying question.
    """

    crisis = state["crisis"]
    state["response_kind"] = ResponseKind.THERAPEUTIC

    if crisis.needs_clarification:
        state["response_text"] = _select_clarification_message(state)
        return state

    state["response_text"] = (
        "I’m here with you. Tell me a bit more about what feels hardest right now."
    )
    return state
