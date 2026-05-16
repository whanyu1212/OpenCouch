"""Shared contracts for text-agent runtime adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from langchain_core.runnables import RunnableConfig

from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentState

TextAgentRuntimeName = Literal["langgraph"]


@dataclass(frozen=True)
class TextRuntimeStatusEvent:
    """Provider-neutral status emitted while a text turn runs."""

    stage: str
    turn_finalized: bool = False


@dataclass(frozen=True)
class TextRuntimeChunkEvent:
    """Provider-neutral text chunk emitted while a text turn runs."""

    text: str


@dataclass(frozen=True)
class TextRuntimeStateEvent:
    """Provider-neutral state snapshot emitted by a text runtime."""

    state: AgentState


TextRuntimeStreamEvent = (
    TextRuntimeStatusEvent | TextRuntimeChunkEvent | TextRuntimeStateEvent
)


class TextAgentAdapter(Protocol):
    """Adapter boundary between persistent runtime and agent implementation."""

    async def get_state(self, config: RunnableConfig) -> AgentState | None:
        """Return the latest persisted state snapshot for a thread."""

    async def run_turn(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
    ) -> Mapping[str, Any]:
        """Run one non-streaming text turn."""

    def run_turn_stream(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        """Run one streaming text turn."""
