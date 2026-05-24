"""Transcript normalization helpers for voice sessions."""

from __future__ import annotations

from typing import Any


def voice_turn_to_transcript_entries(
    *,
    user_text: str,
    assistant_text: str,
    response_style: str | None,
) -> list[dict[str, Any]]:
    """Convert one voice turn into OpenCouch transcript entries."""

    entries: list[dict[str, Any]] = []
    normalized_user = user_text.strip()
    if normalized_user:
        entries.append({"role": "user", "content": normalized_user})

    normalized_assistant = assistant_text.strip()
    if normalized_assistant:
        entries.append(
            {
                "role": "assistant",
                "content": normalized_assistant,
                "response_style": response_style or "voice",
            }
        )

    return entries
