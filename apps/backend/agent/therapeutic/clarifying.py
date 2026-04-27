"""Clarifying response mode - ask one focused question."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import build_clarifying_system_prompt
from agent.therapeutic.response_modes.common import run_streamed_mode_response

logger = logging.getLogger(__name__)

_DEFAULT_CLARIFYING_REPLY = (
    "It sounds like something's on your mind. "
    "Can you help me understand a bit more about what brought this up today?"
)


async def run_clarifying_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a single clarifying question for an ambiguous message.

    Activated when the user's message is too short, too ambiguous, or too
    out-of-context to respond to well. Rather than guessing wrong, the
    agent asks ONE focused question.

    Falls back to a deterministic template when no LLM client is available.

    Args:
        state: Current graph state for the turn.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response delta for the parent graph.
    """

    return await run_streamed_mode_response(
        state,
        runtime,
        mode="clarifying",
        system_prompt_builder=build_clarifying_system_prompt,
        fallback_text=_DEFAULT_CLARIFYING_REPLY,
        logger=logger,
        failure_message=(
            "Clarifying response LLM call failed; using deterministic fallback."
        ),
        stream_writer_factory=get_stream_writer,
    )
