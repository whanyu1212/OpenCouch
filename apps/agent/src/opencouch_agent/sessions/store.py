"""Session store interface and an in-memory implementation."""

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol

from pydantic_ai.messages import ModelMessage

from opencouch_agent.safety.risk import RiskState


@dataclass
class Session:
    """A conversation thread as the server knows it.

    Attributes:
        thread_id: The AG-UI thread ID identifying the conversation.
        history: Every persisted message, oldest first.
        state: The server-owned shared state, as a JSON-compatible dict.
        risk: The conversation's risk. Server-only: never sent to the client.
    """

    thread_id: str
    history: list[ModelMessage] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    risk: RiskState = field(default_factory=RiskState)


class SessionStore(Protocol):
    """Persistence for conversation sessions."""

    async def load(self, thread_id: str) -> Session:
        """Load a session, returning an empty one if the thread is new."""
        ...

    async def record_user_turn(
        self,
        thread_id: str,
        lease_token: str,
        messages: Sequence[ModelMessage],
        risk: RiskState,
    ) -> bool:
        """Atomically append the user's message and save the conversation's risk.

        Called before the agent runs, so the message and any rise in risk
        survive a failed run. Like `save_run`, it only writes while
        `lease_token` holds the thread.

        Args:
            thread_id: The thread the turn belongs to.
            lease_token: The token the turn claimed the thread with.
            messages: The user's new message.
            risk: The conversation's risk after screening that message.

        Returns:
            True if saved; False, writing nothing, if the lease was lost.
        """
        ...

    async def save_run(
        self,
        thread_id: str,
        lease_token: str,
        messages: Sequence[ModelMessage],
        state: dict[str, Any],
    ) -> bool:
        """Atomically append a run's messages and replace the thread's state.

        The write only happens if `lease_token` still holds the thread, so a
        turn that lost its lease can't overwrite the turn that took over.

        Args:
            thread_id: The thread the run belongs to.
            lease_token: The token the turn claimed the thread with.
            messages: The run's new messages.
            state: The thread's state after the run.

        Returns:
            True if saved; False, writing nothing, if the lease was lost.
        """
        ...

    async def claim_turn(
        self, thread_id: str, lease_token: str, lease: timedelta
    ) -> bool:
        """Claim a thread for one turn.

        Args:
            thread_id: The thread to claim.
            lease_token: A unique token identifying this turn's claim.
            lease: How long the claim lasts if it is never released.

        Returns:
            True if claimed; False if another unexpired turn holds the thread.
        """
        ...

    async def release_turn(self, thread_id: str, lease_token: str) -> None:
        """Release a turn claim, but only if `lease_token` still holds it."""
        ...


class InMemorySessionStore:
    """Session store backed by a dict. For tests and local runs only."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        # thread_id -> (lease_token, monotonic expiry time)
        self._turn_leases: dict[str, tuple[str, float]] = {}

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
            risk=stored.risk,
        )

    async def record_user_turn(
        self,
        thread_id: str,
        lease_token: str,
        messages: Sequence[ModelMessage],
        risk: RiskState,
    ) -> bool:
        """Save the user's turn; see `SessionStore.record_user_turn`."""
        if not self._holds_lease(thread_id, lease_token):
            return False
        session = self._sessions.setdefault(thread_id, Session(thread_id=thread_id))
        session.history.extend(messages)
        session.risk = risk
        return True

    async def save_run(
        self,
        thread_id: str,
        lease_token: str,
        messages: Sequence[ModelMessage],
        state: dict[str, Any],
    ) -> bool:
        """Save a run's messages and state; see `SessionStore.save_run`."""
        if not self._holds_lease(thread_id, lease_token):
            return False
        session = self._sessions.setdefault(thread_id, Session(thread_id=thread_id))
        session.history.extend(messages)
        session.state = dict(state)
        return True

    async def claim_turn(
        self, thread_id: str, lease_token: str, lease: timedelta
    ) -> bool:
        """Claim a thread for one turn; see `SessionStore.claim_turn`."""
        now = time.monotonic()
        current = self._turn_leases.get(thread_id)
        if current is not None and current[1] > now:
            return False
        self._turn_leases[thread_id] = (lease_token, now + lease.total_seconds())
        return True

    async def release_turn(self, thread_id: str, lease_token: str) -> None:
        """Release a turn claim, but only if `lease_token` still holds it."""
        if self._holds_lease(thread_id, lease_token):
            del self._turn_leases[thread_id]

    def _holds_lease(self, thread_id: str, lease_token: str) -> bool:
        current = self._turn_leases.get(thread_id)
        return current is not None and current[0] == lease_token
