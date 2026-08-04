"""OpenAI Realtime function tool schemas for voice sessions."""

from __future__ import annotations

from typing import Any

from agent.voice.tools.specs import VOICE_TOOL_SPECS


__all__ = ["build_voice_realtime_tools"]


def build_voice_realtime_tools(*, memory_mode: str) -> list[dict[str, Any]]:
    """Return the narrow function-tool surface exposed to Realtime."""

    persistent = memory_mode.strip().lower() == "persistent"
    return [
        spec.as_realtime_function_tool()
        for spec in VOICE_TOOL_SPECS
        if persistent or not spec.persistent_only
    ]
