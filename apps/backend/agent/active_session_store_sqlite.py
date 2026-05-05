"""SQLite-backed implementation of :class:`ActiveSessionStore`.

The Postgres implementation and shared protocol/DDL constants live in
:mod:`agent.active_session_store`. This module is the SQLite fallback,
selectable via ``OPENCOUCH_PERSISTENCE_BACKEND=sqlite`` for installs
without Docker. It piggybacks on the LangGraph SQLite checkpointer
connection rather than opening its own, since the active-session table
lives alongside thread checkpoints in the same SQLite file.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.active_session_store import (
    ACTIVE_SESSION_EXTRA_COLUMNS,
    ACTIVE_SESSION_STATE_DDL,
    ActiveSessionStore,
)


class SqliteActiveSessionStore:
    """SQLite active-session store using the runtime checkpointer connection."""

    def __init__(
        self,
        *,
        checkpointer_getter: Callable[[], AsyncSqliteSaver],
    ) -> None:
        """Initialize the SQLite-backed active-session store.

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

    async def load_row(
        self,
        thread_id: str,
    ) -> tuple[str, str | None, str | None, bool, str | None] | None:
        """Load one persisted active-session row."""

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
        return (
            str(row[0]),
            str(row[1]) if row[1] is not None else None,
            str(row[2]) if row[2] is not None else None,
            bool(row[3]),
            str(row[4]) if row[4] is not None else None,
        )

    async def list_ids(self) -> list[str]:
        """List thread ids with unresolved active sessions."""

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

    async def save_payload(self, thread_id: str, payload_json: str) -> None:
        """Upsert one serialized active-session payload."""

        checkpointer = self._checkpointer_getter()
        async with checkpointer.lock:
            await checkpointer.conn.execute(
                """
                INSERT INTO opencouch_active_sessions(thread_id, payload_json)
                VALUES(?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    payload_json = excluded.payload_json
                """,
                (thread_id, payload_json),
            )
            await checkpointer.conn.commit()

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

    async def clear_mutation(self, thread_id: str, mutation_token: str) -> None:
        """Clear mutation metadata when the current token still owns it."""

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

    async def set_rotation_required(self, thread_id: str) -> None:
        """Mark one persisted active session for channel-level rotation."""

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

    async def delete_session(self, thread_id: str) -> None:
        """Delete one persisted active-session row."""

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


_: type[ActiveSessionStore] = SqliteActiveSessionStore

__all__ = ["SqliteActiveSessionStore"]
