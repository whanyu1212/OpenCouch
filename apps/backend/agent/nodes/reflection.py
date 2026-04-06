"""Reflection response node."""

from __future__ import annotations

from typing import cast

from agent.models import ResponseKind
from agent.prompts import (
    build_reflection_response_prompt,
    build_reflection_system_prompt,
)
from agent.prompts.catalog import Modality
from agent.state import AgentState
from services.llm.base import BaseLLMClient


def _fallback_reflection_response(state: AgentState) -> str:
    """Return a deterministic reflection reply."""

    if state.get("session_stage") == "closing":
        return (
            "What seems most important from this session is that this pattern has been pulling on you in a recurring way, "
            "and you’ve started naming it more clearly instead of just sitting inside it. A useful next step may be to notice "
            "the first moment this pattern shows up again and pause there. We can return to it when you’re ready."
        )

    if "grief_support" in state.get("active_modalities", []):
        return (
            "What stands out is how much this loss is still shaping the way things feel day to day. "
            "There seems to be both pain and a sense of being stuck with it, which can make grief feel very heavy. "
            "Does that feel close to what you mean?"
        )

    return (
        "A pattern I notice is that this seems to keep pulling you into the same emotional loop, even when part of you wants it to change. "
        "It sounds like the feeling itself and the meaning you attach to it may both be weighing on you. "
        "Does that fit, or is there a different pattern that feels more true to you?"
    )


async def run_reflection_response(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Generate a reflection-style reply.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated agent state with a reflection reply.
    """

    state["response_type"] = ResponseKind.THERAPEUTIC
    state["mode"] = "reflection"
    modalities = cast(
        tuple[Modality, ...],
        tuple(state.get("active_modalities", ["motivational_interviewing"])),
    )

    if llm_client is not None:
        try:
            state["response_text"] = await llm_client.generate_text(
                prompt=build_reflection_response_prompt(state),
                system_instruction=build_reflection_system_prompt(
                    modalities=modalities,
                ),
                temperature=0.4,
            )
            return state
        except Exception:
            pass

    state["response_text"] = _fallback_reflection_response(state)
    return state
