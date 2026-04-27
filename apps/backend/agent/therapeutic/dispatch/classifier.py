"""LLM classifier wrapper for therapeutic dispatch."""

from __future__ import annotations

from agent.memory.models import DispatchDecision
from agent.state import AgentState
from agent.therapeutic.dispatch.prompt import (
    build_therapeutic_dispatch_prompt,
    build_therapeutic_dispatch_system_prompt,
)


async def _pick_mode_and_modality_with_llm(
    state: AgentState,
    llm_client,
) -> tuple[str, str]:
    """Call the structured-output classifier for mode and modality.

    Args:
        state: The current agent state.
        llm_client: The configured control-plane LLM client.

    Returns:
        A ``(mode, modality)`` tuple from the structured classifier response.

    Raises:
        Exception: Propagates any classifier error to the caller.
    """

    raw: DispatchDecision = await llm_client.generate_structured(
        prompt=build_therapeutic_dispatch_prompt(state),
        response_schema=DispatchDecision,
        system_instruction=build_therapeutic_dispatch_system_prompt(),
    )

    return raw.response_style, raw.therapeutic_approach  # type: ignore[return-value]
