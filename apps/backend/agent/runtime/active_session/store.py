"""Persistence stores for runtime-owned active-session coordination."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

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

    async def aclose(self) -> None:
        """Close store resources."""


class PostgresActiveSessionStore:
    """Postgres-backed active-session store using a runtime-owned connection."""

    def __init__(
        self,
        *,
        dsn: str,
    ) -> None:
        self.dsn = dsn
        self._connection: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self._closed = False
        self._connect_lock = asyncio.Lock()

    async def _ensure_connection(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("PostgresActiveSessionStore is closed.")
        if self._connection is not None:
            return self._connection

        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("PostgresActiveSessionStore is closed.")
            if self._connection is not None:
                return self._connection

            conn = await psycopg.AsyncConnection.connect(
                self.dsn,
                row_factory=dict_row,
                autocommit=True,
            )
            try:
                await self._ensure_schema(conn)
            except BaseException:
                await conn.close()
                raise
            self._connection = conn
            return self._connection

    @staticmethod
    async def _ensure_schema(
        conn: psycopg.AsyncConnection[dict[str, Any]],
    ) -> None:
        async with conn.transaction():
            async with conn.cursor() as cursor:
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

    async def ensure_schema(self) -> None:
        """Create or migrate the active-session table."""

        await self._ensure_connection()

    async def load_row(
        self,
        thread_id: str,
    ) -> tuple[str, str | None, str | None, bool, str | None] | None:
        """Load one persisted active-session row."""

        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
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

        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
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

        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
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
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(sql, params)

    async def clear_mutation(self, thread_id: str, mutation_token: str) -> None:
        """Clear mutation metadata when the current token still owns it."""

        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
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

        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
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

        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                DELETE FROM opencouch_active_sessions
                WHERE thread_id = %s
                """,
                (thread_id,),
            )

    async def aclose(self) -> None:
        """Close the PostgreSQL connection."""

        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                logger.warning(
                    "PostgresActiveSessionStore: connection close raised; ignoring",
                    exc_info=True,
                )
            finally:
                self._connection = None


__all__ = [
    "ACTIVE_SESSION_EXTRA_COLUMNS",
    "ACTIVE_SESSION_STATE_DDL",
    "ActiveSessionStore",
    "PostgresActiveSessionStore",
]
