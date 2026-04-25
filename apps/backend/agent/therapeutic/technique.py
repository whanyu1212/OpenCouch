"""Technique response style - the therapeutic approach drives the turn.

In technique mode, the approach knowledge (CBT arc, ACT process, MI
rhythm, etc.) is the primary behavioral instruction. The response
style instruction just says "follow the approach's process guidance."
This is the one style where the approach is loud, not background.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseCategory
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import (
    build_technique_system_prompt,
    build_therapeutic_response_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_TECHNIQUE_REPLY = (
    "Let's stay with what you just described. "
    "What's the thought that shows up most in that moment?"
)
_ATTUNED_OPENING_TERMS = (
    "hard",
    "heavy",
    "painful",
    "sounds",
    "makes sense",
    "understandable",
    "a lot",
    "carrying",
    "tough",
)


def _first_sentence(response_text: str) -> str:
    """Return the first sentence-like span from a generated response.

    Args:
        response_text: The generated technique reply text.

    Returns:
        The first sentence-like span, or the stripped text when no sentence
        boundary is present.
    """

    match = re.search(r"(.+?[.!?])(?:\s|$)", response_text.strip())
    if match:
        return match.group(1).strip()
    return response_text.strip()


def _ensure_attuned_opening(response_text: str) -> str:
    """Ensure technique replies open with a brief attuned acknowledgment.

    Args:
        response_text: The generated technique reply text.

    Returns:
        The original reply when its first sentence is already attuned, or a
        prefixed acknowledgment when the opening is too abrupt.
    """

    stripped = response_text.strip()
    if not stripped:
        return _DEFAULT_TECHNIQUE_REPLY

    first_sentence = _first_sentence(stripped).lower()
    if any(term in first_sentence for term in _ATTUNED_OPENING_TERMS):
        return stripped
    return f"That sounds really hard. {stripped}"


async def run_technique_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a response driven by the active therapeutic approach.

    The technique response style delegates behavioral control to the
    approach knowledge loaded in the system prompt. The approach's arc
    template, Socratic rhythm, transition signals, etc. shape the
    response — the style instruction just says "follow the approach."

    Falls back to a deterministic template when no LLM client is
    available.

    Args:
        state: Current graph state for the turn.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response delta for the parent graph.
    """

    llm_client = runtime.context.response_llm or runtime.context.llm_client

    response_text = _DEFAULT_TECHNIQUE_REPLY
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(state, mode="technique"),
                system_instruction=build_technique_system_prompt(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Technique response LLM call failed; using deterministic fallback.",
                exc_info=True,
            )
    response_text = _ensure_attuned_opening(response_text)

    return {
        "response_kind": ResponseCategory.THERAPEUTIC,
        "response_text": response_text,
        "response_style": "technique",
        "response_style_source": "therapeutic_dispatch",
        "response_style_type": ModeType.THERAPEUTIC,
    }
