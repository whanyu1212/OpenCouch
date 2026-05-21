"""Streaming and turn presentation helpers for the persistent runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from agent.runtime.turn import state_to_output
from agent.models import AgentOutput, ChunkEvent, Message, MessageRole
from agent.state import AgentState


def messages_from_transcript(
    transcript: list[dict[str, Any]],
) -> list[Message]:
    """Materialize validated messages from a serialized transcript.

    Args:
        transcript (list[dict[str, Any]]): Serialized transcript entries.

    Returns:
        list[Message]: Validated message objects.
    """

    messages: list[Message] = []
    for turn in transcript:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        style = turn.get("response_style") if role == "assistant" else None
        messages.append(
            Message(role=MessageRole(role), content=content, response_style=style)
        )
    return messages


def stamp_turn_total_ms(
    state: AgentState,
    *,
    started_at: float,
) -> None:
    """Record total turn latency and the post-finalize tail in diagnostics.

    Two related metrics land here:

    - ``turn_total_ms``: end-to-end wall-clock from runtime entry to runtime
      termination.
    - ``post_finalize_ms``: wall-clock from when turn finalization
      locked in the response to runtime completion. This is the window
      where the user has *already received* their response (when streaming)
      but the runtime is still finishing terminal bookkeeping. Absent when
      finalization did not run for this turn (early failure paths).

    Args:
        state (AgentState): Final runtime state for the turn.
        started_at (float): Monotonic timestamp captured before runtime execution.

    Returns:
        None: Mutates the state's diagnostics mapping in place.
    """

    end_at = time.monotonic()
    if "diagnostics" not in state or state["diagnostics"] is None:
        state["diagnostics"] = {}
    state["diagnostics"]["turn_total_ms"] = round((end_at - started_at) * 1000, 2)

    finalize_done_at = state["diagnostics"].pop("finalize_done_at_monotonic", None)
    if finalize_done_at is not None:
        state["diagnostics"]["post_finalize_ms"] = round(
            (end_at - float(finalize_done_at)) * 1000,
            2,
        )


def response_ready_output(
    state: Mapping[str, Any] | None,
    *,
    finalize_seen: bool,
    response_ready_emitted: bool,
) -> AgentOutput | None:
    """Return the durable output once finalize has completed.

    Args:
        state (Mapping[str, Any] | None): Latest streamed state snapshot.
        finalize_seen (bool): Whether the finalize node has emitted an update.
        response_ready_emitted (bool): Whether a ready event was already emitted.

    Returns:
        AgentOutput | None: Finalized output when ready to surface.
    """

    if state is None or not finalize_seen or response_ready_emitted:
        return None
    response_text = str(state.get("response_text", "") or "").strip()
    if not response_text:
        return None
    return state_to_output(state)


def chunk_event_from_custom_payload(payload: Any) -> ChunkEvent | None:
    """Build a chunk event from a custom stream payload.

    Args:
        payload (Any): Custom stream payload emitted by the runtime.

    Returns:
        ChunkEvent | None: Token chunk event when the payload is a text chunk.
    """

    if not isinstance(payload, dict) or payload.get("type") != "chunk":
        return None
    return ChunkEvent(text=payload["text"])
