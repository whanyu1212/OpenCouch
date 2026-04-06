"""Guided exercise response node."""

from __future__ import annotations

from agent.models import ResponseKind
from agent.prompts import (
    build_guided_exercise_response_prompt,
    build_guided_exercise_system_prompt,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient


def _fallback_guided_exercise_response(state: AgentState) -> str:
    """Return a deterministic guided exercise reply."""

    if state.get("session_stage") == "closing":
        return (
            "Before we end, keep this very simple: hold onto one sentence or one step from today that felt most grounding or useful, "
            "and come back to it the next time this feeling shows up. You do not need to do a full exercise right now. The goal is just "
            "to leave with one small thing you can actually reuse."
        )

    message = state["message"].lower()
    if any(term in message for term in ("breathe", "breathing", "calm down", "grounding")):
        return (
            "Let's try one short grounding reset. Look around and name 5 things you can see, 4 things you can feel, "
            "3 things you can hear, 2 things you can smell, and 1 thing you can taste or imagine tasting. "
            "Take it slowly and just move through the list without trying to do it perfectly."
        )

    return (
        "Let's keep it simple and structured. Write down the situation, the main thought that showed up, "
        "the emotion you felt most strongly, and one alternative way to look at the same situation. "
        "You do not need to force a positive answer, just something a little more balanced."
    )


async def run_guided_exercise_response(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Generate a guided-exercise reply.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated agent state with a guided-exercise reply.
    """

    state["response_type"] = ResponseKind.THERAPEUTIC
    state["mode"] = "guided_exercise"

    if llm_client is not None:
        try:
            state["response_text"] = await llm_client.generate_text(
                prompt=build_guided_exercise_response_prompt(state),
                system_instruction=build_guided_exercise_system_prompt(),
                temperature=0.3,
            )
            return state
        except Exception:
            pass

    state["response_text"] = _fallback_guided_exercise_response(state)
    return state
