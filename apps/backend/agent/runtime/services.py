"""Explicit service boundary for OpenAI text-runtime flow modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from agent.runtime.context import OpenAITextRunContext
from agent.runtime.types import TextRuntimeConfig
from agent.runtime.workflow_context import WorkflowContext
from agent.specialists.roster import OpenAITextAgentRoster
from agent.state import AgentState


class BuildRunContext(Protocol):
    """Build the local SDK tool context for one text-runtime turn."""

    def __call__(
        self,
        state: AgentState,
        config: TextRuntimeConfig,
        context: WorkflowContext,
    ) -> OpenAITextRunContext: ...


class BuildAgent(Protocol):
    """Select the specialist agent for one runtime state."""

    def __call__(self, state: AgentState) -> Any: ...


class InputTextForState(Protocol):
    """Build the therapeutic agent input text for one runtime state."""

    def __call__(
        self,
        state: AgentState,
        *,
        include_recent_history: bool = True,
        prompt_appendix: str | None = None,
    ) -> str: ...


class CrisisInputTextForState(Protocol):
    """Build crisis-specialist input text for one runtime state."""

    def __call__(
        self,
        state: AgentState,
        *,
        runtime_mode: str,
        include_recent_history: bool = True,
        require_resource_tool: bool = False,
    ) -> str: ...


class RunOpenAIAgentWith(Protocol):
    """Run one SDK agent turn and return normalized text plus duration."""

    async def __call__(
        self,
        state: AgentState,
        *,
        agent: Any,
        input_text: str,
        run_context: OpenAITextRunContext,
        session: Any | None = None,
    ) -> tuple[str, float]: ...


class FinalizeTurn(Protocol):
    """Finalize one OpenAI text-runtime turn."""

    async def __call__(
        self,
        state: AgentState,
        *,
        response_text: str,
        config: TextRuntimeConfig,
        runtime_mode: str,
        response_style: str,
        selected_agent: str | None,
        sdk_duration_ms: float | None,
        streamed: bool,
    ) -> AgentState: ...


class LoadTurnMemory(Protocol):
    """Load prompt-visible memory into one turn state."""

    async def __call__(
        self,
        state: AgentState,
        context: WorkflowContext,
    ) -> AgentState: ...


@dataclass(frozen=True, slots=True)
class TextRuntimeServices:
    """Narrow runtime operations used by specialist flow modules."""

    runner: Any
    roster: OpenAITextAgentRoster
    build_run_context: BuildRunContext
    build_agent: BuildAgent
    input_text_for_state: InputTextForState
    crisis_input_text_for_state: CrisisInputTextForState
    run_openai_agent_with: RunOpenAIAgentWith
    finalize_turn: FinalizeTurn
    load_turn_memory: LoadTurnMemory


TextRuntimeServicesFactory = Callable[[], TextRuntimeServices]


__all__ = ["TextRuntimeServices", "TextRuntimeServicesFactory"]
