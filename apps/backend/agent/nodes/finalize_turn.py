"""Compatibility helper that appends the assistant turn to the transcript.

This node exists because transcript finalization is the one concern that
spans both branches and runs last: regardless of whether the turn took
the crisis path or the therapeutic path, the final assistant response
lives in ``state["response_text"]`` and needs to be appended to
``transcript`` so the next turn's ``get_history`` call in
:class:`agent.persistence.PersistentAgentRuntime` sees it.

The guard against appending an empty response is important: if some
response node short-circuits without setting ``response_text``, we'd
otherwise pollute the transcript with a blank assistant turn.
"""

from __future__ import annotations

import time
from typing import Any

from agent.models import MessageRole
from agent.state import AgentState


async def run_finalize_turn_node(
    state: AgentState,
    runtime: Any,  # noqa: ARG001 - compatibility signature
) -> dict[str, Any]:
    """Append the final assistant response to the transcript.

    The user message was already emitted into ``transcript`` by
    :func:`agent.graph.build_initial_state` at turn start.

    Returns a delta containing ``transcript`` with the new assistant turn.
    If the response text is empty (which
    should only happen if a branch short-circuits without producing
    a reply), returns an empty delta so the transcript stays clean.

    Args:
        state: Current graph state after a response node has run.
        runtime: Runtime object, unused but retained for compatibility.

    Returns:
        State delta appending the assistant turn, or an empty delta.
    """

    response_text = str(state.get("response_text", "") or "").strip()

    # Mark the moment the response is locked in so the runtime can later
    # compute ``post_finalize_ms`` — the wall-clock between this point
    # and graph termination. Used to measure the latency wedge that
    # background extraction (#5) would close.
    finalize_done_at_monotonic = time.monotonic()

    if not response_text:
        # Nothing to append. Better to leave the transcript alone than
        # to write a blank assistant turn that the CLI would render.
        return {
            "diagnostics": {
                "finalize_done_at_monotonic": finalize_done_at_monotonic,
            }
        }

    # Stamp the routing mode onto the assistant turn so it surfaces in history.
    routing_mode = state.get("response_style") or None

    assistant_turn = {
        "role": MessageRole.ASSISTANT.value,
        "content": response_text,
        "response_style": routing_mode,
    }

    return {
        "transcript": [assistant_turn],
        "diagnostics": {
            "finalize_done_at_monotonic": finalize_done_at_monotonic,
        },
    }
