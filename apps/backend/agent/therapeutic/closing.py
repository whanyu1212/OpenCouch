"""Closing response mode - tonal wind-down, not a structural session end.

Closing is a **tonal** mode: it generates a warm farewell response that
acknowledges the arc of the conversation, leaves an open door, and
respects unresolved threads. It does NOT end the session, trigger
summarization, or modify ``session_progress.stage``.

Session termination and summarization are owned by the runtime layer:

- ``/end`` and ``/exit`` CLI commands trigger session wrap-up
- Inactivity timeouts trigger session wrap-up
- The runtime's session lifecycle manager fires the summarizer

This separation is deliberate. The LLM-detected "the user seems to be
winding down" signal is useful for picking a response register, but
it should never auto-terminate the session. If the user keeps
talking after the closing turn, the next dispatcher turn will route
to whatever mode fits. The runtime owns session structure; this
node only owns response tone.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import build_closing_system_prompt
from agent.therapeutic.response_modes.common import run_streamed_mode_response

logger = logging.getLogger(__name__)

_DEFAULT_CLOSING_REPLY = (
    "I'm glad you took the time to talk this through. "
    "Whenever you want to pick this back up, I'm here."
)


async def run_closing_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a tonal closing response for the current turn.

    Activated when the user signals they're winding down (e.g., "I
    should go", "thanks, this helped") or a natural lull follows
    productive work. The agent offers a short, warm farewell that
    acknowledges the arc of the session without triggering any
    structural session-end behavior.

    Falls back to a deterministic two-sentence farewell when no LLM
    client is available. The fallback is deliberately generic — it
    works regardless of what the session contained — because the
    failure mode we care about most is "never say 'it was nice
    talking to you'", and the fallback string avoids that trap.

    Args:
        state: Current graph state for the turn.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response delta for the parent graph.
    """

    return await run_streamed_mode_response(
        state,
        runtime,
        mode="closing",
        system_prompt_builder=build_closing_system_prompt,
        fallback_text=_DEFAULT_CLOSING_REPLY,
        logger=logger,
        failure_message=(
            "Closing response LLM call failed; using deterministic fallback."
        ),
        stream_writer_factory=get_stream_writer,
    )
