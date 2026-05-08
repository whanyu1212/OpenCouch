"""Crisis response node for the OpenCouch graph."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.gates.safety.prompts import (
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)


def _default_crisis_reply(state: AgentState) -> str:
    """Return a deterministic fallback crisis reply.

    Args:
        state: Current graph state containing optional crisis metadata.

    Returns:
        User-facing crisis fallback response.
    """

    crisis = state.get("crisis")
    level = crisis.level if crisis is not None else 0
    urgency = (
        "If you might act on these thoughts soon, please contact your local emergency services right now or go to the nearest emergency department."
        if level >= 3
        else "If you feel at risk of harming yourself, please contact your local emergency services right now or go to the nearest emergency department."
    )
    location_prompt = ""
    if state.get("resource_lookup_status") != "location_refused":
        location_prompt = (
            " If you're comfortable, you can share your country or region and "
            "I can help look up the most relevant local crisis line."
        )
    return (
        "Thank you for telling me this — I'm really glad you reached out. "
        "Your safety matters most right now. "
        f"{urgency} "
        "If possible, move away from anything you could use to hurt yourself and contact a trusted person who can stay with you. "
        f"{location_prompt}"
    )


async def run_crisis_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a crisis-mode response and return the resulting state delta.

    Args:
        state: Current graph state after crisis classification and optional
            resource lookup.
        runtime: LangGraph runtime carrying the workflow context.

    Returns:
        A partial state update containing the crisis response fields.
    """

    llm_client = runtime.context.llm_client

    response_text = _default_crisis_reply(state)
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_crisis_response_prompt(state),
                system_instruction=build_crisis_response_system_prompt(),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Crisis LLM response generation failed; using deterministic reply.",
                exc_info=True,
            )
            response_text = _default_crisis_reply(state)

    return {
        "route": "crisis",
        "response_style": "crisis_response",
        "response_text": response_text,
    }
