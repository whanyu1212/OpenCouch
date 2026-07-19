"""Shared runtime types for persistent agent sessions."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from agent.runtime.text_turn_graph import TextRoutePlan
    from agent.runtime.workflow_context import WorkflowContext

from agent.models import AgentOutput, Message
from agent.state import AgentState

ExpectedSessionLiveness = Literal["active", "absent"]
TextRuntimeConfig = Mapping[str, Any]


class SessionStatus(StrEnum):
    """Runtime liveness states for active-session coordination."""

    ABSENT = "absent"
    ACTIVE = "active"
    EXPIRED_UNFINALIZED = "expired_unfinalized"
    INTERRUPTED = "interrupted"
    ROTATION_REQUIRED = "rotation_required"


class SessionLeaseExpired(RuntimeError):
    """Raised when a turn was submitted against a non-active session lease."""

    def __init__(self, thread_id: str, status: SessionStatus) -> None:
        """Initialize the liveness mismatch error.

        Args:
            thread_id (str): Thread whose lease check failed.
            status (SessionStatus): Observed session status.
        """

        self.thread_id = thread_id
        self.status = status
        super().__init__(
            f"thread {thread_id!r} is not active; observed status={status.value}"
        )


class ActiveSessionExists(RuntimeError):
    """Raised when a caller expected no active session but one exists."""

    def __init__(self, thread_id: str, status: SessionStatus) -> None:
        """Initialize the active-session conflict.

        Args:
            thread_id (str): Thread whose absence check failed.
            status (SessionStatus): Observed session status.
        """

        self.thread_id = thread_id
        self.status = status
        super().__init__(
            f"thread {thread_id!r} already has session status={status.value}"
        )


class SessionInterrupted(RuntimeError):
    """Raised when a persisted session needs explicit recovery finalization."""

    def __init__(self, thread_id: str) -> None:
        """Initialize the interrupted-session error.

        Args:
            thread_id (str): Thread whose session is interrupted.
        """

        self.thread_id = thread_id
        super().__init__(f"thread {thread_id!r} has an interrupted session")


@dataclass(slots=True)
class PersistentTurnResult:
    """Return value for one persisted conversation turn."""

    output: AgentOutput
    state: AgentState
    history: list[Message]


@dataclass(slots=True)
class ThreadSummary:
    """Compact persisted-thread summary for CLI thread management."""

    thread_id: str
    turn_count: int
    message_count: int
    has_context: bool


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


class ExecuteTextRoute(Protocol):
    """Execute one planned text route to a final state."""

    async def __call__(
        self,
        plan: TextRoutePlan,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AgentState: ...


class StreamTextRoute(Protocol):
    """Stream events for one planned text route."""

    def __call__(
        self,
        plan: TextRoutePlan,
        *,
        config: TextRuntimeConfig,
        context: WorkflowContext,
        session: Any | None = None,
    ) -> AsyncIterator[TextRuntimeStreamEvent]: ...


@dataclass(frozen=True)
class RouteHandler:
    """Paired execute and stream callables for one text route."""

    execute: ExecuteTextRoute
    stream: StreamTextRoute
