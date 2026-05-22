"""Shared OpenAI runtime utility helpers."""

from __future__ import annotations

from typing import Any


def final_output_text(output: Any, *, fallback: str = "") -> str:
    text = output if isinstance(output, str) and output else str(output or fallback)
    if not text:
        raise ValueError("OpenAI Agents SDK returned an empty text response.")
    return text


def chunk_from_sdk_event(event: Any) -> str | None:
    if getattr(event, "type", None) != "raw_response_event":
        return None

    data = getattr(event, "data", None)
    event_type = (
        data.get("type") if isinstance(data, dict) else getattr(data, "type", None)
    )
    if event_type != "response.output_text.delta":
        return None

    delta = (
        data.get("delta") if isinstance(data, dict) else getattr(data, "delta", None)
    )
    return str(delta or "") or None
