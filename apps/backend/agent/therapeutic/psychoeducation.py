"""Psychoeducation response mode - short, normalizing framing of a reaction."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import build_psychoeducation_system_prompt
from agent.therapeutic.response_modes.common import run_streamed_mode_response

logger = logging.getLogger(__name__)

_DEFAULT_PSYCHOEDUCATION_REPLY = (
    "Something about this reaction might be worth framing gently. "
    "I have a thought about what could be going on — but first, "
    "does it feel like the right moment to sit with this, "
    "or would something steadier help more?"
)
_PSYCHOEDUCATION_FALLBACK_QUESTION = "Does that fit what you notice in your body?"


def _ensure_psychoeducation_question(response_text: str) -> str:
    """Ensure psychoeducation replies include one check-in question.

    Args:
        response_text: The generated psychoeducation reply text.

    Returns:
        The original text when it already includes a question, otherwise the
        text with a brief fit-check question appended.
    """

    stripped = response_text.strip()
    if not stripped:
        return _DEFAULT_PSYCHOEDUCATION_REPLY
    if "?" in stripped:
        return stripped
    suffix = "" if stripped.endswith((".", "!", "?")) else "."
    return f"{stripped}{suffix} {_PSYCHOEDUCATION_FALLBACK_QUESTION}"


async def run_psychoeducation_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a brief normalizing explanation of the user's reaction.

    Activated when the user is confused about their own reaction
    ("why am I crying over this?", "is it normal to feel both angry
    and relieved?") or asking for a frame on something they're
    experiencing. The agent offers one short, plain-language
    explanation and then pivots back to the user's specific situation.

    Falls back to a deterministic permission-first template when no
    LLM client is available. The fallback is deliberately the minimal
    form of psychoeducation (acknowledge + offer + check-in) rather
    than a topic-specific explanation, because the fallback must work
    across all four topic branches (anxiety/stress/grief/general) and
    all possible user states. Offering a gentle check-in is a safe
    default when the LLM can't produce a context-aware framing.

    Args:
        state: Current graph state for the turn.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response delta for the parent graph.
    """

    return await run_streamed_mode_response(
        state,
        runtime,
        mode="psychoeducation",
        system_prompt_builder=build_psychoeducation_system_prompt,
        fallback_text=_DEFAULT_PSYCHOEDUCATION_REPLY,
        logger=logger,
        failure_message=(
            "Psychoeducation response LLM call failed; using deterministic fallback."
        ),
        postprocess=_ensure_psychoeducation_question,
        stream_writer_factory=get_stream_writer,
    )
