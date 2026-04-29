"""Conversation transcript access helpers.

These helpers make ``transcript`` the preferred source of conversation context
while retaining ``history`` as a compatibility fallback for older checkpoints
and focused tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_transcript(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the transcript-like conversation turns from state.

    Args:
        state: Current graph state or state-like mapping.

    Returns:
        Conversation turns from ``transcript`` when present, otherwise the
        legacy ``history`` fallback. Non-list values return an empty list.
    """

    transcript = state.get("transcript")
    if isinstance(transcript, list):
        return [turn for turn in transcript if isinstance(turn, dict)]

    history = state.get("history")
    if isinstance(history, list):
        return [turn for turn in history if isinstance(turn, dict)]

    return []


def get_recent_history(
    state: Mapping[str, Any], *, limit: int = 6
) -> list[dict[str, Any]]:
    """Return recent transcript turns for prompt/context use.

    Args:
        state: Current graph state or state-like mapping.
        limit: Maximum number of turns to return.

    Returns:
        Up to ``limit`` recent conversation turns.
    """

    if limit <= 0:
        return []
    return get_transcript(state)[-limit:]


def format_recent_history(
    state: Mapping[str, Any],
    *,
    limit: int = 6,
    empty: str = "(no prior history)",
) -> str:
    """Format recent transcript turns for prompt injection.

    Args:
        state: Current graph state or state-like mapping.
        limit: Maximum number of recent turns to include.
        empty: Placeholder returned when no conversation turns are available.

    Returns:
        Formatted conversation history text.
    """

    lines = []
    for turn in get_recent_history(state, limit=limit):
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        role = str(turn.get("role", "unknown") or "unknown")
        lines.append(f"{role}: {content}")
    return "\n".join(lines) or empty


def get_user_turns(state: Mapping[str, Any]) -> list[str]:
    """Return user-authored transcript turn contents.

    Args:
        state: Current graph state or state-like mapping.

    Returns:
        User message contents in transcript order.
    """

    return [
        str(turn.get("content", "") or "")
        for turn in get_transcript(state)
        if turn.get("role") == "user" and turn.get("content")
    ]
