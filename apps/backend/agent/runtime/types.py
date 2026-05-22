"""Shared runtime types for persistent agent sessions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

from agent.models import AgentOutput, Message
from agent.state import AgentState

ExpectedSessionLiveness = Literal["active", "absent"]
TextRuntimeConfig = Mapping[str, Any]
TextRuntimeShadowStatus = Literal["eligible", "fallback", "error"]


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


@dataclass(frozen=True)
class TextRuntimeShadowResult:
    """Non-serving comparison artifact for a candidate text runtime."""

    runtime: Literal["openai"]
    status: TextRuntimeShadowStatus
    eligible: bool
    fallback_reason: str | None = None
    route: str | None = None
    active_flow: str | None = None
    active_flow_action: str | None = None
    memory_reference_mode: str | None = None
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
