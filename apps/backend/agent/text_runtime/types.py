"""Shared contracts for text-agent runtime adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from langchain_core.runnables import RunnableConfig

from agent.runtime_context import WorkflowContext
from agent.state import AgentGraphInputState, AgentState

TextAgentRuntimeName = Literal["langgraph", "openai"]
TextRuntimeShadowStatus = Literal["eligible", "fallback", "error"]


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


@dataclass(frozen=True)
class TextRuntimeShadowResult:
    """Non-serving comparison artifact for a candidate text runtime."""

    runtime: TextAgentRuntimeName
    status: TextRuntimeShadowStatus
    eligible: bool
    fallback_reason: str | None = None
    route: str | None = None
    active_flow: str | None = None
    active_flow_action: str | None = None
    memory_reference_mode: str | None = None
    memory_action_type: str | None = None
    grounded_lookup_query: str | None = None
    crisis_level: int | None = None
    needs_crisis_response: bool | None = None
    needs_crisis_clarification: bool | None = None
    selected_agent: str | None = None
    sdk_duration_ms: float | None = None
    shadow_duration_ms: float | None = None
    response_text_length: int | None = None
    response_text_preview: str | None = None
    response_text_sha256: str | None = None
    error_type: str | None = None
    error_message: str | None = None

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-friendly shadow artifact."""

        return asdict(self)


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
        session: Any | None = None,
    ) -> Mapping[str, Any]:
        """Run one non-streaming text turn."""

    def run_turn_stream(
        self,
        initial_state: AgentGraphInputState,
        *,
        config: RunnableConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]:
        """Run one streaming text turn."""

    async def update_state(
        self,
        config: RunnableConfig,
        values: Mapping[str, Any],
        *,
        as_node: str | None = None,
    ) -> None:
        """Persist a state update through the runtime's checkpoint backend."""
