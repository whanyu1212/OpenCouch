"""Shared session and UI model types for the terminal surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent.models import Message
from agent.state import AgentState
from config import ResponseModelTier
from llm.base import BaseLLMClient

TraceMode = Literal["off", "on", "once"]
UIMode = Literal["full", "compact"]
ObservabilityMode = Literal["compact", "verbose"]
PromptThemeName = Literal["mono", "contrast", "calm"]


@dataclass(slots=True)
class RunnerSession:
    """Mutable local session state shared by terminal frontends."""

    requested_mode: str
    resolved_mode: str
    llm_client: BaseLLMClient | None
    thread_id: str
    sqlite_path: str
    memory_mode: str
    # Optional stable owner for long-term memory. Falls back to thread_id.
    user_id: str | None = None
    history: list[Message] = field(default_factory=list)
    last_context: AgentState | None = None
    response_model_tier: ResponseModelTier = "fast"
    response_llm_client: BaseLLMClient | None = None
    trace_mode: TraceMode = "off"
    ui_mode: UIMode = "full"
    observability_mode: ObservabilityMode = "compact"
    prompt_theme: PromptThemeName = "calm"
    show_onboarding: bool = True

    def owner_id(self) -> str:
        """Return the effective owner identifier for memory operations.

        Returns:
            Explicit ``user_id`` when set, otherwise the active thread id.
        """

        return self.user_id or self.thread_id
