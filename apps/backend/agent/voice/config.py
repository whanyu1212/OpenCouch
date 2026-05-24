"""Configuration defaults for OpenCouch voice sessions."""

from __future__ import annotations

DEFAULT_REALTIME_MODEL = "gpt-realtime-2"
DEFAULT_REALTIME_VOICE = "alloy"
DEFAULT_INPUT_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_TOOL_CHOICE = "auto"
SUPPORTED_REALTIME_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "cedar",
        "coral",
        "echo",
        "marin",
        "sage",
        "shimmer",
        "verse",
    }
)

VoiceMemoryMode = str
