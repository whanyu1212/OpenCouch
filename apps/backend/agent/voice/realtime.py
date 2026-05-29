"""OpenAI Realtime session helpers for OpenCouch voice."""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from agent.voice.config import (
    DEFAULT_INPUT_TRANSCRIPTION_MODEL,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_VOICE,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TOOL_CHOICE,
    SUPPORTED_REALTIME_VOICES,
)
from agent.voice.policy import build_voice_instructions
from agent.voice.tools import build_voice_realtime_tools


def build_realtime_session_config(
    *,
    thread_id: str,
    user_id: str | None,
    memory_mode: str,
    memory_context: str | None = None,
    assistant_voice: str | None = None,
) -> dict[str, Any]:
    """Build the GA Realtime session configuration for a voice session."""

    normalized_thread_id = thread_id.strip()
    if not normalized_thread_id:
        raise ValueError("thread_id must not be empty.")

    normalized_mode = memory_mode.strip().lower()
    if normalized_mode not in {"incognito", "persistent"}:
        raise ValueError("memory_mode must be 'incognito' or 'persistent'.")

    realtime_voice = _normalize_realtime_voice(assistant_voice)

    return {
        "type": "realtime",
        "model": DEFAULT_REALTIME_MODEL,
        "reasoning": {"effort": DEFAULT_REASONING_EFFORT},
        "audio": {
            "input": {
                "transcription": {"model": DEFAULT_INPUT_TRANSCRIPTION_MODEL},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": True,
                    "interrupt_response": True,
                },
            },
            "output": {"voice": realtime_voice},
        },
        "tool_choice": DEFAULT_TOOL_CHOICE,
        "tools": build_voice_realtime_tools(memory_mode=normalized_mode),
        "instructions": build_voice_instructions(
            thread_id=normalized_thread_id,
            user_id=user_id.strip() if user_id else None,
            memory_mode=normalized_mode,
            memory_context=memory_context,
        ),
    }


def _normalize_realtime_voice(value: str | None) -> str:
    """Return a supported built-in Realtime voice name."""

    normalized = (value or DEFAULT_REALTIME_VOICE).strip().lower()
    if normalized not in SUPPORTED_REALTIME_VOICES:
        raise ValueError(f"Unsupported Realtime voice: {value}.")
    return normalized


async def create_realtime_client_secret(
    *,
    session_config: dict[str, object],
    safety_identifier: str | None,
) -> str:
    """Create an ephemeral OpenAI Realtime client secret."""

    extra_headers = (
        {"OpenAI-Safety-Identifier": safety_identifier} if safety_identifier else None
    )
    response = await AsyncOpenAI().realtime.client_secrets.create(
        session=session_config,
        extra_headers=extra_headers,
    )
    return str(response.value)
