"""Psychoeducation response mode - short, normalizing framing of a reaction."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseCategory
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import (
    build_psychoeducation_system_prompt,
    build_therapeutic_response_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_PSYCHOEDUCATION_REPLY = (
    "Something about this reaction might be worth framing gently. "
    "I have a thought about what could be going on — but first, "
    "does it feel like the right moment to sit with this, "
    "or would something steadier help more?"
)


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

    llm_client = runtime.context.response_llm or runtime.context.llm_client

    response_text = _DEFAULT_PSYCHOEDUCATION_REPLY
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(state, mode="psychoeducation"),
                system_instruction=build_psychoeducation_system_prompt(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(
                "Psychoeducation response LLM call failed; "
                "using deterministic fallback.",
                exc_info=True,
            )

    return {
        "response_kind": ResponseCategory.THERAPEUTIC,
        "response_text": response_text,
        "response_style": "psychoeducation",
        "response_style_source": "therapeutic_dispatch",
        "response_style_type": ModeType.THERAPEUTIC,
    }
