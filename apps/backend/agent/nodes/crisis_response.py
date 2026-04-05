"""Crisis response node for the MVP graph."""

from agent.models import ResponseKind
from agent.state import AgentState


async def run_crisis_response(state: AgentState) -> AgentState:
    """Return an empathetic interruption when the crisis gate detects risk."""

    crisis = state["crisis"]
    state["response_kind"] = ResponseKind.CRISIS

    if crisis.level >= 3:
        state["response_text"] = (
            "I’m really glad you said this. It sounds like you may be in immediate danger. "
            "Please contact emergency services or a crisis hotline right now, or reach out "
            "to someone nearby who can stay with you."
        )
    else:
        state["response_text"] = (
            "I’m sorry you’re carrying this right now. What you said sounds serious, and I "
            "want to respond carefully. Please reach out to a crisis hotline or a trusted "
            "person who can be with you while we focus on your safety."
        )

    return state
