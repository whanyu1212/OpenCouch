"""Out-of-scope response node."""

from __future__ import annotations

from agent.models import ResponseKind
from agent.prompts import (
    build_out_of_scope_response_prompt,
    build_out_of_scope_system_prompt,
)
from agent.state import AgentState
from services.llm.base import BaseLLMClient


def _fallback_out_of_scope_response() -> str:
    """Return a deterministic out-of-scope reply."""

    return (
        "I can't diagnose that or give medication or legal guidance. "
        "If you want, I can help you describe what you're experiencing, think through questions to ask a licensed professional, "
        "or stay with the emotional side of what this brings up."
    )


async def run_out_of_scope_response(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Generate an out-of-scope boundary reply.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated agent state with an out-of-scope reply.
    """

    state["response_type"] = ResponseKind.THERAPEUTIC
    state["mode"] = "out_of_scope"

    if llm_client is not None:
        try:
            state["response_text"] = await llm_client.generate_text(
                prompt=build_out_of_scope_response_prompt(state),
                system_instruction=build_out_of_scope_system_prompt(),
                temperature=0.2,
            )
            return state
        except Exception:
            pass

    state["response_text"] = _fallback_out_of_scope_response()
    return state
