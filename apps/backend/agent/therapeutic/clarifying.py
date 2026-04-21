"""Clarifying response mode — ask one focused question."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseKind
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import (
    build_clarifying_system_prompt,
    build_therapeutic_response_prompt,
)

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
    """

    llm_client = runtime.context.response_llm or runtime.context.llm_client

    response_text = _DEFAULT_CLARIFYING_REPLY
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(state, mode="clarifying"),
                system_instruction=build_clarifying_system_prompt(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Clarifying response LLM call failed; using deterministic fallback.",
                exc_info=True,
            )

    return {
        "response": {
            **state.get("response", {}),
            "kind": ResponseKind.THERAPEUTIC,
            "text": response_text,
        },
        "routing": {
            **state.get("routing", {}),
            "response_style": "clarifying",
            "response_style_source": "therapeutic_dispatch",
            "response_style_type": ModeType.THERAPEUTIC,
        },
    }
