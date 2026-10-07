"""Session store interface and an in-memory implementation."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic_ai.messages import ModelMessage


@dataclass
class Session:
    """A conversation thread as the server knows it.

    Attributes:
        thread_id: The AG-UI thread ID identifying the conversation.
        history: Every persisted message, oldest first.
        state: The server-owned shared state, as a JSON-compatible dict.
    """

    thread_id: str
    history: list[ModelMessage] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


class SessionStore(Protocol):
    """Persistence for conversation sessions."""

    async def load(self, thread_id: str) -> Session:
        """Load a session, returning an empty one if the thread is new."""
        ...

    async def append_messages(
        self, thread_id: str, messages: Sequence[ModelMessage]
    ) -> None:
        """Append messages to a thread's history, creating the thread if needed."""
        ...

    async def save_run(
        self,
        thread_id: str,
        messages: Sequence[ModelMessage],
        state: dict[str, Any],
    ) -> None:
        """Atomically append a run's messages and replace the thread's state."""
        ...


class InMemorySessionStore:
    """Session store backed by a dict. For tests and local runs only."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def load(self, thread_id: str) -> Session:
        """Load a session, returning an empty one if the thread is new."""
        stored = self._sessions.get(thread_id)
        if stored is None:
            return Session(thread_id=thread_id)
        # Copy so callers can't mutate stored history in place.
        return Session(
            thread_id=thread_id,
            history=list(stored.history),
            state=dict(stored.state),
        )

    async def append_messages(
        self, thread_id: str, messages: Sequence[ModelMessage]
    ) -> None:
        """Append messages to a thread's history, creating the thread if needed."""
        session = self._sessions.setdefault(thread_id, Session(thread_id=thread_id))
        session.history.extend(messages)

    async def save_run(
        self,
        thread_id: str,
        messages: Sequence[ModelMessage],
        state: dict[str, Any],
    ) -> None:
        """Atomically append a run's messages and replace the thread's state."""
        session = self._sessions.setdefault(thread_id, Session(thread_id=thread_id))
        session.history.extend(messages)
        session.state = dict(state)
