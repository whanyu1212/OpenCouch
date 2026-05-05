"""Persistence stores for runtime-owned active-session coordination."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

if TYPE_CHECKING:
    from agent.active_session_store_sqlite import SqliteActiveSessionStore

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


class ActiveSessionStore(Protocol):
    """Storage interface for durable active-session coordination state."""

    async def ensure_schema(self) -> None:
        """Create or migrate the backing store schema."""

    async def load_row(
        self,
        thread_id: str,
    ) -> tuple[str, str | None, str | None, bool, str | None] | None:
        """Load one persisted active-session row."""

    async def list_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions."""

    async def save_payload(self, thread_id: str, payload_json: str) -> None:
        """Upsert one serialized active-session payload."""

    async def set_mutation(
        self,
        thread_id: str,
        *,
        mutation_token: str,
        mutation_kind: str,
        finalize_required_reason: str | None = None,
    ) -> None:
        """Persist mutation-coordination metadata for one thread."""

    async def clear_mutation(self, thread_id: str, mutation_token: str) -> None:
        """Clear mutation metadata when the current token still owns it."""

    async def set_rotation_required(self, thread_id: str) -> None:
        """Mark one persisted active session for channel-level rotation."""

    async def delete_session(self, thread_id: str) -> None:
        """Delete one persisted active-session row."""


class PostgresActiveSessionStore:
    """Postgres-backed active-session store using the runtime checkpointer connection."""

    def __init__(
        self,
        *,
        checkpointer_getter: Callable[[], AsyncPostgresSaver],
    ) -> None:
        """Initialize the Postgres-backed active-session store.

        Args:
            checkpointer_getter: Callback returning the open runtime checkpointer.

        Returns:
            None: Stores the checkpointer getter for lazy access.
        """

        self._checkpointer_getter = checkpointer_getter

    async def ensure_schema(self) -> None:
        """Create or migrate the active-session table."""

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.cursor() as cursor:
                await cursor.execute(ACTIVE_SESSION_STATE_DDL)
                await cursor.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'opencouch_active_sessions'
                    """
                )
                rows = await cursor.fetchall()
                present_columns = {str(row["column_name"]) for row in rows}
                for column_name, column_ddl in ACTIVE_SESSION_EXTRA_COLUMNS.items():
                    if column_name in present_columns:
                        continue
                    await cursor.execute(
                        "ALTER TABLE opencouch_active_sessions "
                        f"ADD COLUMN {column_name} {column_ddl}"
                    )

    async def load_row(
        self,
        thread_id: str,
    ) -> tuple[str, str | None, str | None, bool, str | None] | None:
        """Load one persisted active-session row."""

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT
                        payload_json,
                        mutation_token,
                        mutation_kind,
                        rotate_after_this_turn,
                        finalize_required_reason
                    FROM opencouch_active_sessions
                    WHERE thread_id = %s
                    """,
                    (thread_id,),
                )
                row = await cursor.fetchone()
        if row is None:
            return None
        return (
            str(row["payload_json"]),
            str(row["mutation_token"]) if row["mutation_token"] is not None else None,
            str(row["mutation_kind"]) if row["mutation_kind"] is not None else None,
            bool(row["rotate_after_this_turn"]),
            (
                str(row["finalize_required_reason"])
                if row["finalize_required_reason"] is not None
                else None
            ),
        )

    async def list_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions."""

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT thread_id
                    FROM opencouch_active_sessions
                    ORDER BY thread_id
                    """
                )
                rows = await cursor.fetchall()
        return [str(row["thread_id"]) for row in rows]

    async def save_payload(self, thread_id: str, payload_json: str) -> None:
        """Upsert one serialized active-session payload."""

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO opencouch_active_sessions(thread_id, payload_json)
                    VALUES(%s, %s)
                    ON CONFLICT(thread_id) DO UPDATE SET
                        payload_json = excluded.payload_json
                    """,
                    (thread_id, payload_json),
                )

    async def set_mutation(
        self,
        thread_id: str,
        *,
        mutation_token: str,
        mutation_kind: str,
        finalize_required_reason: str | None = None,
    ) -> None:
        """Persist mutation-coordination metadata for one thread."""

        checkpointer = self._checkpointer_getter()
        if finalize_required_reason is None:
            sql = """
                UPDATE opencouch_active_sessions
                SET mutation_token = %s, mutation_kind = %s
                WHERE thread_id = %s
                """
            params: tuple[Any, ...] = (mutation_token, mutation_kind, thread_id)
        else:
            sql = """
                UPDATE opencouch_active_sessions
                SET
                    mutation_token = %s,
                    mutation_kind = %s,
                    finalize_required_reason = %s
                WHERE thread_id = %s
                """
            params = (
                mutation_token,
                mutation_kind,
                finalize_required_reason,
                thread_id,
            )
        async with checkpointer.lock:
            async with checkpointer.conn.cursor() as cursor:
                await cursor.execute(sql, params)

    async def clear_mutation(self, thread_id: str, mutation_token: str) -> None:
        """Clear mutation metadata when the current token still owns it."""

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE opencouch_active_sessions
                    SET mutation_token = NULL, mutation_kind = NULL
                    WHERE thread_id = %s AND mutation_token = %s
                    """,
                    (thread_id, mutation_token),
                )

    async def set_rotation_required(self, thread_id: str) -> None:
        """Mark one persisted active session for channel-level rotation."""

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE opencouch_active_sessions
                    SET rotate_after_this_turn = 1
                    WHERE thread_id = %s
                    """,
                    (thread_id,),
                )

    async def delete_session(self, thread_id: str) -> None:
        """Delete one persisted active-session row."""

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            async with checkpointer.conn.cursor() as cursor:
                await cursor.execute(
                    """
                    DELETE FROM opencouch_active_sessions
                    WHERE thread_id = %s
                    """,
                    (thread_id,),
                )


def __getattr__(name: str) -> Any:
    """Lazily expose compatibility imports for legacy active-session stores."""

    if name == "SqliteActiveSessionStore":
        from agent.active_session_store_sqlite import SqliteActiveSessionStore

        return SqliteActiveSessionStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ACTIVE_SESSION_EXTRA_COLUMNS",
    "ACTIVE_SESSION_STATE_DDL",
    "ActiveSessionStore",
    "PostgresActiveSessionStore",
    "SqliteActiveSessionStore",
]
