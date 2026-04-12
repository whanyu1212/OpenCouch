"""Supportive response mode — warm validation and gentle reflection."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseKind
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import (
    build_supportive_system_prompt,
    build_therapeutic_response_prompt,
)

logger = logging.getLogger(__name__)

_DEFAULT_SUPPORTIVE_REPLY = (
    "Thank you for sharing that with me. "
    "It sounds like there's a lot on your mind right now, "
    "and I want you to know that what you're feeling makes sense. "
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
    """

    llm_client = runtime.context.get("llm_client")

    response_text = _DEFAULT_SUPPORTIVE_REPLY
    if llm_client is not None:
        try:
            response_text = await llm_client.generate_text(
                prompt=build_therapeutic_response_prompt(state, mode="supportive"),
                system_instruction=build_supportive_system_prompt(state),
                temperature=0.7,
            )
        except Exception:
            logger.warning(
                "Supportive response LLM call failed; using deterministic fallback.",
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
            "mode": "supportive",
            "mode_source": "therapeutic_dispatch",
            "mode_type": ModeType.THERAPEUTIC,
        },
    }
