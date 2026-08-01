"""Compatibility facade for OpenCouch voice tools."""

from __future__ import annotations

from agent.voice.tools.dispatch import (
    execute_voice_tool_call,
    _registered_voice_tool_names,
)
from agent.voice.tools.handlers import (
    _execute_crisis_support_template,
    _execute_recall_saved_memory,
)
from agent.voice.tools.schemas import build_voice_realtime_tools
from agent.voice.tools.specs import (
    VOICE_TOOL_SPECS,
    VoiceToolSpec,
    _SUPPORTED_VOICE_TOOL_NAMES,
)

__all__ = [
    "_SUPPORTED_VOICE_TOOL_NAMES",
    "_execute_crisis_support_template",
    "_execute_recall_saved_memory",
    "_registered_voice_tool_names",
    "build_voice_realtime_tools",
    "execute_voice_tool_call",
    "VOICE_TOOL_SPECS",
    "VoiceToolSpec",
]
