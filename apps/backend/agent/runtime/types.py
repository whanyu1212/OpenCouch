"""Shared runtime types for persistent agent sessions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from agent.models import AgentOutput, Message
from agent.state import AgentState

ExpectedSessionLiveness = Literal["active", "absent"]


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
