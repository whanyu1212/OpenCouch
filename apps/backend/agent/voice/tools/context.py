"""Shared types for voice tool dispatch."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

VoiceToolHandler = Callable[
    ["VoiceToolDispatchContext", dict[str, object]],
    Awaitable[object],
]


@dataclass(frozen=True)
class VoiceToolDispatchContext:
    runtime: Any
    tool_context: Any | None
    thread_id: str
    user_id: str | None


@dataclass(frozen=True)
class VoiceToolDefinition:
    name: str
    handler: VoiceToolHandler
    requires_context: bool = True
