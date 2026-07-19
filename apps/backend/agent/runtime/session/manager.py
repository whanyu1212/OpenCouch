"""Active-session storage and coordination helpers for persistence runtime."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4

from agent.runtime.session.store import ActiveSessionStore
from agent.memory.policy.candidates import SessionMemoryBuffer
from agent.memory.modes import MemoryMode


@dataclass(slots=True)
class PersistedActiveSessionState:
    """Durable runtime-owned session state for active-session recovery."""

    thread_id: str
    started_at: str
    last_active_at: str
    transcript_start_index: int
    max_crisis_level: int
    session_buffer: SessionMemoryBuffer

    def to_json(self) -> str:
        """Serialize the session record to JSON.

        Returns:
            The JSON-encoded session payload.
        """

        return json.dumps(
            {
                "thread_id": self.thread_id,
                "started_at": self.started_at,
                "last_active_at": self.last_active_at,
                "transcript_start_index": self.transcript_start_index,
                "max_crisis_level": self.max_crisis_level,
                "session_buffer": self.session_buffer.model_dump(mode="json"),
            }
        )

    @classmethod
    def from_json(cls, payload_json: str) -> PersistedActiveSessionState:
        """Deserialize a persisted session record.

        Args:
            payload_json: The JSON payload to decode.

        Returns:
            The decoded ``PersistedActiveSessionState`` instance.
        """

        payload = json.loads(payload_json)
        return cls(
            thread_id=str(payload["thread_id"]),
            started_at=str(payload["started_at"]),
            last_active_at=str(payload["last_active_at"]),
            transcript_start_index=max(
                0, int(payload.get("transcript_start_index", 0) or 0)
            ),
            max_crisis_level=max(0, int(payload.get("max_crisis_level", 0) or 0)),
            session_buffer=SessionMemoryBuffer.model_validate(
                payload.get("session_buffer")
                or {"session_id": str(payload["thread_id"])}
            ),
        )


@dataclass(slots=True)
class PersistedActiveSessionRow:
    """Raw active-session row including coordination metadata."""

    payload_json: str
    mutation_token: str | None
    mutation_kind: str | None
    rotate_after_this_turn: bool
    finalize_required_reason: str | None


class ActiveSessionManager:
    """Own persistent active-session row storage and mutation coordination."""

    def __init__(
        self,
        *,
        store: ActiveSessionStore,
        memory_mode: MemoryMode,
        session_timeout: timedelta | None = None,
    ) -> None:
        """Initialize the active-session manager.

        Args:
            store: Persistence store for active-session rows.
            memory_mode: Persistence tier for the runtime.
            session_timeout: Deprecated compatibility argument; ignored.
        """

        if session_timeout is not None:
            warnings.warn(
                "ActiveSessionManager session_timeout is deprecated and ignored; "
                "expiration policy belongs to SessionLifecycleService",
                DeprecationWarning,
                stacklevel=2,
            )
        self._store = store
        self._memory_mode = memory_mode
        self._active_mutation_tokens: set[str] = set()
        self._runtime_instance_id = uuid4().hex

    async def ensure_schema(self) -> None:
        """Create or migrate the active-session table.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        await self._store.ensure_schema()

    def new_mutation_token(self) -> str:
        """Return a process-scoped mutation token.

        Returns:
            A mutation token that identifies this runtime instance.
        """

        return f"{uuid4().hex}:{self._runtime_instance_id}:{os.getpid()}"

    def is_mutation_in_flight(self, token: str | None) -> bool:
        """Return whether a mutation token is actively running in this process.

        Args:
            token: Persisted mutation token.

        Returns:
            True when this runtime currently owns an in-flight mutation.
        """

        return token in self._active_mutation_tokens

    @asynccontextmanager
    async def active_session_mutation(
        self,
        thread_id: str,
        *,
        mutation_kind: str,
        finalize_required_reason: str | None = None,
    ) -> AsyncIterator[str]:
        """Track an in-flight active-session mutation for recovery.

        Args:
            thread_id: Thread identifier.
            mutation_kind: Mutation kind for diagnostics.
            finalize_required_reason: Optional durable recovery reason.

        Yields:
            The process-scoped mutation token.
        """

        mutation_token = self.new_mutation_token()
        self._active_mutation_tokens.add(mutation_token)
        await self.set_active_session_mutation(
            thread_id,
            mutation_token=mutation_token,
            mutation_kind=mutation_kind,
            finalize_required_reason=finalize_required_reason,
        )
        try:
            yield mutation_token
        finally:
            self._active_mutation_tokens.discard(mutation_token)

    async def load_persisted_active_session_row(
        self,
        thread_id: str,
    ) -> PersistedActiveSessionRow | None:
        """Load a raw active-session row.

        Args:
            thread_id: Thread identifier.

        Returns:
            The persisted row, or ``None`` when absent.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return None

        row = await self._store.load_row(thread_id)
        if row is None:
            return None
        return PersistedActiveSessionRow(
            payload_json=row[0],
            mutation_token=row[1],
            mutation_kind=row[2],
            rotate_after_this_turn=row[3],
            finalize_required_reason=row[4],
        )

    async def load_persisted_active_session(
        self,
        thread_id: str,
    ) -> PersistedActiveSessionState | None:
        """Load the persisted active-session record for a thread.

        Args:
            thread_id: The thread identifier to read.

        Returns:
            The persisted session record, or ``None`` when absent.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return None

        row = await self.load_persisted_active_session_row(thread_id)
        if row is None:
            return None
        return PersistedActiveSessionState.from_json(row.payload_json)

    async def list_persisted_active_session_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions.

        Returns:
            The unresolved active-session thread ids.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return []

        return await self._store.list_ids()

    async def save_persisted_active_session(
        self,
        session: PersistedActiveSessionState,
    ) -> None:
        """Persist one active-session record.

        Args:
            session: The session record to upsert.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        await self._store.save_payload(session.thread_id, session.to_json())

    async def set_active_session_mutation(
        self,
        thread_id: str,
        *,
        mutation_token: str,
        mutation_kind: str,
        finalize_required_reason: str | None = None,
    ) -> None:
        """Persist a best-effort marker for an in-flight session mutation.

        Args:
            thread_id: Thread identifier.
            mutation_token: Process-scoped mutation token.
            mutation_kind: Mutation kind for diagnostics.
            finalize_required_reason: Optional durable recovery reason.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        await self._store.set_mutation(
            thread_id,
            mutation_token=mutation_token,
            mutation_kind=mutation_kind,
            finalize_required_reason=finalize_required_reason,
        )

    async def clear_active_session_mutation(
        self,
        thread_id: str,
        mutation_token: str,
    ) -> None:
        """Clear a mutation marker when the current process owns it.

        Args:
            thread_id: Thread identifier.
            mutation_token: Token to clear.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        await self._store.clear_mutation(thread_id, mutation_token)

    async def set_active_session_rotation_required(self, thread_id: str) -> None:
        """Mark a persisted active session for channel-level rotation.

        Args:
            thread_id: Thread identifier.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        await self._store.set_rotation_required(thread_id)

    async def delete_persisted_active_session(self, thread_id: str) -> None:
        """Delete the persisted active-session record for a thread.

        Args:
            thread_id: The thread identifier to delete.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        await self._store.delete_session(thread_id)
