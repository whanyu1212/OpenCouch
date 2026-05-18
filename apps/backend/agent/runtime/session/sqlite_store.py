"""SQLite-backed implementation of :class:`ActiveSessionStore`."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import aiosqlite

from agent.runtime.session.store import (
    ACTIVE_SESSION_EXTRA_COLUMNS,
    ACTIVE_SESSION_STATE_DDL,
    ActiveSessionStore,
)

logger = logging.getLogger(__name__)


class SqliteActiveSessionStore:
    """SQLite active-session store using a runtime-owned connection."""

    def __init__(
        self,
        *,
        sqlite_path: str | Path,
    ) -> None:
        """Initialize the SQLite-backed active-session store.

        Args:
            sqlite_path: SQLite file path or ``":memory:"``.

        Returns: None.
        """

        self.sqlite_path = (
            Path(sqlite_path) if sqlite_path != ":memory:" else Path(":memory:")
        )
        self._connection: aiosqlite.Connection | None = None
        self._closed = False
        self._connect_lock = asyncio.Lock()

    async def _ensure_connection(self) -> aiosqlite.Connection:
        if self._closed:
            raise RuntimeError("SqliteActiveSessionStore is closed.")
        if self._connection is not None:
            return self._connection

        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("SqliteActiveSessionStore is closed.")
            if self._connection is not None:
                return self._connection

            if str(self.sqlite_path) != ":memory:":
                self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(self.sqlite_path))
            try:
                conn.row_factory = aiosqlite.Row
                await self._ensure_schema(conn)
            except BaseException:
                await conn.close()
                raise
            self._connection = conn
            return self._connection

    @staticmethod
    async def _ensure_schema(conn: aiosqlite.Connection) -> None:
        await conn.execute(ACTIVE_SESSION_STATE_DDL)
        async with conn.execute(
            "PRAGMA table_info(opencouch_active_sessions)"
        ) as cursor:
            rows = await cursor.fetchall()
        present_columns = {str(row[1]) for row in rows}
        for column_name, column_ddl in ACTIVE_SESSION_EXTRA_COLUMNS.items():
            if column_name in present_columns:
                continue
            await conn.execute(
                "ALTER TABLE opencouch_active_sessions "
                f"ADD COLUMN {column_name} {column_ddl}"
            )
        await conn.commit()

    async def ensure_schema(self) -> None:
        """Create or migrate the active-session table."""

        await self._ensure_connection()

    async def load_row(
        self,
        thread_id: str,
    ) -> tuple[str, str | None, str | None, bool, str | None] | None:
        """Load one persisted active-session row."""

        conn = await self._ensure_connection()
        async with conn.execute(
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

        conn = await self._ensure_connection()
        async with conn.execute(
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

        conn = await self._ensure_connection()
        await conn.execute(
            """
            INSERT INTO opencouch_active_sessions(thread_id, payload_json)
            VALUES(?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                payload_json = excluded.payload_json
            """,
            (thread_id, payload_json),
        )
        await conn.commit()

    async def set_mutation(
        self,
        thread_id: str,
        *,
        mutation_token: str,
        mutation_kind: str,
        finalize_required_reason: str | None = None,
    ) -> None:
        """Persist mutation-coordination metadata for one thread."""

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
        conn = await self._ensure_connection()
        await conn.execute(sql, params)
        await conn.commit()

    async def clear_mutation(self, thread_id: str, mutation_token: str) -> None:
        """Clear mutation metadata when the current token still owns it."""

        conn = await self._ensure_connection()
        await conn.execute(
            """
            UPDATE opencouch_active_sessions
            SET mutation_token = NULL, mutation_kind = NULL
            WHERE thread_id = ? AND mutation_token = ?
            """,
            (thread_id, mutation_token),
        )
        await conn.commit()

    async def set_rotation_required(self, thread_id: str) -> None:
        """Mark one persisted active session for channel-level rotation."""

        conn = await self._ensure_connection()
        await conn.execute(
            """
            UPDATE opencouch_active_sessions
            SET rotate_after_this_turn = 1
            WHERE thread_id = ?
            """,
            (thread_id,),
        )
        await conn.commit()

    async def delete_session(self, thread_id: str) -> None:
        """Delete one persisted active-session row."""

        conn = await self._ensure_connection()
        await conn.execute(
            """
            DELETE FROM opencouch_active_sessions
            WHERE thread_id = ?
            """,
            (thread_id,),
        )
        await conn.commit()

    async def aclose(self) -> None:
        """Close the SQLite connection."""

        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                logger.warning(
                    "SqliteActiveSessionStore: connection close raised; ignoring",
                    exc_info=True,
                )
            finally:
                self._connection = None


_: type[ActiveSessionStore] = SqliteActiveSessionStore

__all__ = ["SqliteActiveSessionStore"]
