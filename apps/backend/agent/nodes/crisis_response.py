"""Compatibility adapter for crisis responses."""

from __future__ import annotations

from typing import Any

from agent.gates.safety.prompts import (
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)
from agent.crisis_branch import crisis_response_delta
from agent.state import AgentState


async def run_crisis_response_node(
    state: AgentState,
    runtime: Any,
) -> dict[str, Any]:
    """Generate a crisis-mode response and return the resulting state delta.

    Args:
        state: Current graph state after crisis classification and optional
            resource lookup.
        runtime: Runtime object carrying the workflow context.

    Returns:
        A partial state update containing the crisis response fields.
    """

    llm_client = runtime.context.response_llm or runtime.context.llm_client
    if llm_client is None:
        raise RuntimeError("crisis_response_node requires an LLM client.")

    chunks: list[str] = []
    async for chunk in llm_client.generate_text_stream(
        prompt=build_crisis_response_prompt(state),
        system_instruction=build_crisis_response_system_prompt(),
    ):
        chunks.append(chunk)
    response_text = "".join(chunks)

    return crisis_response_delta(response_text)
