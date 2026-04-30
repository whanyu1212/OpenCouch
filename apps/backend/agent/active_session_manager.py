"""Active-session storage and coordination helpers for persistence runtime."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from agent.memory.candidates import SessionMemoryBuffer
from agent.memory.modes import MemoryMode
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

ACTIVE_SESSION_STATE_DDL = """
CREATE TABLE IF NOT EXISTS opencouch_active_sessions (
    thread_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    mutation_token TEXT,
    mutation_kind TEXT,
    rotate_after_this_turn INTEGER NOT NULL DEFAULT 0,
    finalize_required_reason TEXT
);
"""
ACTIVE_SESSION_EXTRA_COLUMNS = {
    "mutation_token": "TEXT",
    "mutation_kind": "TEXT",
    "rotate_after_this_turn": "INTEGER NOT NULL DEFAULT 0",
    "finalize_required_reason": "TEXT",
}


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


def parse_iso_timestamp(value: str | None) -> datetime | None:
    """Parse a stored ISO timestamp.

    Args:
        value: The timestamp string to parse.

    Returns:
        A parsed ``datetime`` or ``None`` when parsing fails.
    """

    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class ActiveSessionManager:
    """Own persistent active-session row storage and mutation coordination."""

    def __init__(
        self,
        *,
        checkpointer_getter: Callable[[], AsyncSqliteSaver],
        memory_mode: MemoryMode,
        session_timeout: timedelta,
    ) -> None:
        """Initialize the active-session manager.

        Args:
            checkpointer_getter: Callback returning the open runtime checkpointer.
            memory_mode: Persistence tier for the runtime.
            session_timeout: Inactivity window before an active session expires.
        """

        self._checkpointer_getter = checkpointer_getter
        self._memory_mode = memory_mode
        self._session_timeout = session_timeout
        self._active_mutation_tokens: set[str] = set()
        self._runtime_instance_id = uuid4().hex

    async def ensure_schema(self) -> None:
        """Create or migrate the active-session table.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            await checkpointer.conn.execute(ACTIVE_SESSION_STATE_DDL)
            async with checkpointer.conn.execute(
                "PRAGMA table_info(opencouch_active_sessions)"
            ) as cursor:
                rows = await cursor.fetchall()
            present_columns = {str(row[1]) for row in rows}
            for column_name, column_ddl in ACTIVE_SESSION_EXTRA_COLUMNS.items():
                if column_name in present_columns:
                    continue
                await checkpointer.conn.execute(
                    "ALTER TABLE opencouch_active_sessions "
                    f"ADD COLUMN {column_name} {column_ddl}"
                )
            await checkpointer.conn.commit()

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

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.execute(
                """
                SELECT
                    payload_json,
                    mutation_token,
                    mutation_kind,
                    rotate_after_this_turn,
                    finalize_required_reason
                FROM opencouch_active_sessions
                WHERE thread_id = ?
                """,
                (thread_id,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return PersistedActiveSessionRow(
            payload_json=str(row[0]),
            mutation_token=str(row[1]) if row[1] is not None else None,
            mutation_kind=str(row[2]) if row[2] is not None else None,
            rotate_after_this_turn=bool(row[3]),
            finalize_required_reason=str(row[4]) if row[4] is not None else None,
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

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.execute(
                """
                SELECT thread_id
                FROM opencouch_active_sessions
                ORDER BY thread_id
                """
            ) as cursor:
                rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

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

        checkpointer = self._checkpointer_getter()
        payload_json = session.to_json()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                INSERT INTO opencouch_active_sessions(thread_id, payload_json)
                VALUES(?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    payload_json = excluded.payload_json
                """,
                (session.thread_id, payload_json),
            )
            await checkpointer.conn.commit()

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

        checkpointer = self._checkpointer_getter()
        if finalize_required_reason is None:
            sql = """
                UPDATE opencouch_active_sessions
                SET mutation_token = ?, mutation_kind = ?
                WHERE thread_id = ?
                """
            params: tuple[Any, ...] = (mutation_token, mutation_kind, thread_id)
        else:
            sql = """
                UPDATE opencouch_active_sessions
                SET
                    mutation_token = ?,
                    mutation_kind = ?,
                    finalize_required_reason = ?
                WHERE thread_id = ?
                """
            params = (
                mutation_token,
                mutation_kind,
                finalize_required_reason,
                thread_id,
            )
        async with checkpointer.lock:
            await checkpointer.conn.execute(sql, params)
            await checkpointer.conn.commit()

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

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                UPDATE opencouch_active_sessions
                SET mutation_token = NULL, mutation_kind = NULL
                WHERE thread_id = ? AND mutation_token = ?
                """,
                (thread_id, mutation_token),
            )
            await checkpointer.conn.commit()

    async def set_active_session_rotation_required(self, thread_id: str) -> None:
        """Mark a persisted active session for channel-level rotation.

        Args:
            thread_id: Thread identifier.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                UPDATE opencouch_active_sessions
                SET rotate_after_this_turn = 1
                WHERE thread_id = ?
                """,
                (thread_id,),
            )
            await checkpointer.conn.commit()

    async def delete_persisted_active_session(self, thread_id: str) -> None:
        """Delete the persisted active-session record for a thread.

        Args:
            thread_id: The thread identifier to delete.

        Returns:
            None.
        """

        if self._memory_mode == MemoryMode.INCOGNITO:
            return

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                DELETE FROM opencouch_active_sessions
                WHERE thread_id = ?
                """,
                (thread_id,),
            )
            await checkpointer.conn.commit()

    def session_has_expired(self, session: PersistedActiveSessionState) -> bool:
        """Return whether an active session crossed the inactivity timeout.

        Args:
            session: The persisted active-session record.

        Returns:
            ``True`` when the session is expired.
        """

        last_active = parse_iso_timestamp(session.last_active_at)
        if last_active is None:
            return True
        return (
            datetime.now(tz=last_active.tzinfo) - last_active >= self._session_timeout
        )
