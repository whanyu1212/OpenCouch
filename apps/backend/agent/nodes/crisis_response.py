"""Crisis response node for the OpenCouch graph."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseKind
from agent.prompts import (
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.tools.web_search import find_local_crisis_resources

logger = logging.getLogger(__name__)


def _default_crisis_reply(state: AgentState) -> str:
    """Return a deterministic fallback crisis reply."""

    crisis = state.get("crisis")
    level = crisis.level if crisis is not None else 0
    urgency = (
        "If you might act on these thoughts soon, please contact your local emergency services right now or go to the nearest emergency department."
        if level >= 3
        else "If you feel at risk of harming yourself, please contact your local emergency services right now or go to the nearest emergency department."
    )
    return (
        "Thank you for telling me this — I'm really glad you reached out. "
        "Your safety matters most right now. "
        f"{urgency} "
        "If possible, move away from anything you could use to hurt yourself and contact a trusted person who can stay with you. "
        "If you're comfortable, you can share your country or region and I can help look up the most relevant local crisis line."
    )


async def run_crisis_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a crisis-mode response and return the resulting state delta.

    When an LLM client is available, attempts to extract the user's location
    and look up verified regional crisis resources before generating the
    reply. Both the location and resources are persisted in state for
    observability. Falls back to a deterministic template on any error.
    """

    llm_client = runtime.context.llm_client

    # ── Step 1: Attempt location-aware resource lookup (silent on failure) ─
    inferred_location = ""
    found_resources: list[dict[str, str]] = []

    if llm_client is not None:
        try:
            inferred_location, found_resources = await find_local_crisis_resources(
                state, llm_client=llm_client
            )
        except Exception:
            logger.warning(
                "Crisis resource lookup failed; continuing without resources.",
                exc_info=True,
            )

    # ── Step 2: Build an enriched view of state for the LLM prompt builder ─
    # The prompt builder reads response.found_resources / response.inferred_location
    # to inject verified hotlines into the system prompt. We construct an in-memory
    # view that includes them without mutating the input state.
    response_with_resources: dict[str, Any] = {
        **state.get("response", {}),
        "inferred_location": inferred_location,
        "found_resources": found_resources,
    }
    enriched_state: AgentState = {**state, "response": response_with_resources}

    # ── Step 3: Generate the empathetic reply, falling back deterministically ─
    response_text = _default_crisis_reply(enriched_state)
    if llm_client is not None:
        try:
            writer = get_stream_writer()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_crisis_response_prompt(enriched_state),
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
            response_text = _default_crisis_reply(enriched_state)

    # ── Step 4: Return only the keys this node updated ────────────────────
    return {
        "response": {
            **response_with_resources,
            "kind": ResponseKind.CRISIS,
            "text": response_text,
        },
        "routing": {
            **state.get("routing", {}),
            "route": "crisis",
            "response_style": "crisis_response",
            "response_style_source": "crisis_gate",
            "response_style_type": ModeType.CRISIS,
        },
    }
