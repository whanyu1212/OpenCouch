"""Supportive response mode - warm validation and gentle reflection."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import build_supportive_system_prompt
from agent.therapeutic.response_modes.common import run_streamed_mode_response

logger = logging.getLogger(__name__)

_DEFAULT_SUPPORTIVE_REPLY = (
    "It sounds like there's a lot on your mind right now, "
    "and what you're feeling makes sense. "
    "Take your time — I'm here whenever you're ready to say more."
)


async def run_supportive_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a supportive, validating therapeutic response.

    The default mode. Covers the majority of turns: the user is sharing,
    and the agent's job is to listen well, validate the feeling, and
    leave room for the user to continue.

    Falls back to a deterministic template when no LLM client is
    available (common in tests and deterministic-mode CLI runs).

    Args:
        state: Current graph state for the turn.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response delta for the parent graph.
    """

    return await run_streamed_mode_response(
        state,
        runtime,
        mode="supportive",
        system_prompt_builder=build_supportive_system_prompt,
        fallback_text=_DEFAULT_SUPPORTIVE_REPLY,
        logger=logger,
        failure_message=(
            "Supportive response LLM call failed; using deterministic fallback."
        ),
        stream_writer_factory=get_stream_writer,
    )
