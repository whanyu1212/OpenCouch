"""Reflective response mode - pattern naming and gentle probing."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseCategory
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import (
    build_reflective_system_prompt,
    build_therapeutic_response_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_REFLECTIVE_REPLY = (
    "I notice you keep coming back to this — it sounds like there might be a "
    "pattern here that's worth looking at together. "
    "What do you think connects these moments for you?"
)
_REFLECTIVE_FALLBACK_QUESTION = (
    "What do you think keeps connecting these moments for you?"
)


def _ensure_reflective_question(response_text: str) -> str:
    """Ensure reflective replies end with one gentle question.

    Args:
        response_text: The generated reflective reply text.

    Returns:
        The original text when it already includes a question, otherwise the
        text with a short reflective question appended.
    """

    stripped = response_text.strip()
    if not stripped:
        return _DEFAULT_REFLECTIVE_REPLY
    if "?" in stripped:
        return stripped
    suffix = "" if stripped.endswith((".", "!", "?")) else "."
    return f"{stripped}{suffix} {_REFLECTIVE_FALLBACK_QUESTION}"


async def run_reflective_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a pattern-recognizing reflective response.

    Activated when the user describes a recurring pattern, asks "why does
    this keep happening?", or surfaces a theme across multiple turns. The
    agent gently names the pattern and invites reflection.

    Falls back to a deterministic template when no LLM client is available.

    Args:
        state: Current graph state for the turn.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response delta for the parent graph.
    """

    llm_client = runtime.context.response_llm or runtime.context.llm_client

    response_text = _DEFAULT_REFLECTIVE_REPLY
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(state, mode="reflective"),
                system_instruction=build_reflective_system_prompt(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Reflective response LLM call failed; using deterministic fallback.",
                exc_info=True,
            )

    response_text = _ensure_reflective_question(response_text)

    return {
        "response_kind": ResponseCategory.THERAPEUTIC,
        "response_text": response_text,
        "response_style": "reflective",
        "response_style_source": "therapeutic_dispatch",
        "response_style_type": ModeType.THERAPEUTIC,
    }
