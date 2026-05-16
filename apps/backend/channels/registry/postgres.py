"""PostgreSQL Telegram session registry backend."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.rows import dict_row

from channels.telegram import (
    TELEGRAM_RECLAIM_STUCK_AGE,
    TELEGRAM_RECLAIM_STUCK_ATTEMPTS,
    TelegramActiveSession,
    TelegramClosedSession,
    _generate_ulid,
    _legacy_migration_state,
    _LEGACY_MIGRATION_STATES,
    _parse_registry_timestamp,
    _utc_now_iso,
    telegram_session_thread_id,
)


class PostgresTelegramSessionRegistry:
    """PostgreSQL registry for Telegram chat-to-session thread pointers."""

    def __init__(self, dsn: str) -> None:
        """Initialize the registry.

        Args:
            dsn: PostgreSQL database URL.
        """

        self.dsn = dsn
        self._conn: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    async def ensure_started(self) -> None:
        """Open the registry connection and create tables.

        Returns:
            None.
        """

        if self._conn is not None:
            return
        async with self._lock:
            if self._conn is not None:
                return
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
            self._conn = conn

    async def aclose(self) -> None:
        """Close the registry connection.

        Returns:
            None.
        """

        if self._conn is None:
            return
        await self._conn.close()
        self._conn = None

    def _ensure_conn(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        """Return an opened registry connection.

        Returns:
            psycopg.AsyncConnection[dict[str, Any]]: Active PostgreSQL connection.

        Raises:
            RuntimeError: If the registry has not been started.
        """

        if self._conn is None:
            raise RuntimeError("PostgresTelegramSessionRegistry has not been started.")
        return self._conn

    @staticmethod
    async def _ensure_schema(conn: psycopg.AsyncConnection[dict[str, Any]]) -> None:
        """Ensure the PostgreSQL registry schema exists.

        Args:
            conn (psycopg.AsyncConnection[dict[str, Any]]): Open PostgreSQL
                connection.

        Returns:
            None.
        """

        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_chat_active (
                        chat_id TEXT PRIMARY KEY,
                        active_thread_id TEXT,
                        active_started_at TEXT,
                        migrated_legacy TEXT NOT NULL DEFAULT 'pending',
                        migration_last_error TEXT,
                        migration_updated_at TEXT,
                        close_requested_reason TEXT,
                        close_requested_at TEXT
                    )
                    """
                )
                await cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS telegram_chat_sessions (
                        thread_id TEXT PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        closed_at TEXT,
                        closed_reason TEXT,
                        reclaim_started_at TEXT,
                        reclaim_attempts INTEGER NOT NULL DEFAULT 0,
                        reclaim_stuck_at TEXT,
                        last_reclaim_error TEXT,
                        reclaimed_at TEXT
                    )
                    """
                )
                await cursor.execute(
                    """
                    UPDATE telegram_chat_active
                    SET migrated_legacy = CASE
                        WHEN migrated_legacy IN ('1', 'true', 'yes') THEN 'finalized'
                        WHEN migrated_legacy IN ('0', 'false', 'no', '') THEN 'pending'
                        WHEN migrated_legacy IN (
                            'pending',
                            'finalizing',
                            'finalized',
                            'failed'
                        ) THEN migrated_legacy
                        ELSE 'failed'
                    END
                    """
                )

    @staticmethod
    def _active_session_from_row(row: dict[str, Any]) -> TelegramActiveSession:
        """Build an active-session view from a PostgreSQL row.

        Args:
            row (dict[str, Any]): Row returned by ``dict_row``.

        Returns:
            TelegramActiveSession: Registry active pointer view.
        """

        return TelegramActiveSession(
            chat_id=str(row["chat_id"]),
            active_thread_id=(
                str(row["active_thread_id"])
                if row["active_thread_id"] is not None
                else None
            ),
            active_started_at=(
                str(row["active_started_at"])
                if row["active_started_at"] is not None
                else None
            ),
            legacy_migration_state=_legacy_migration_state(row["migrated_legacy"]),
            migration_last_error=(
                str(row["migration_last_error"])
                if row["migration_last_error"] is not None
                else None
            ),
            close_requested_reason=(
                str(row["close_requested_reason"])
                if row["close_requested_reason"] is not None
                else None
            ),
            close_requested_at=(
                str(row["close_requested_at"])
                if row["close_requested_at"] is not None
                else None
            ),
        )

    async def ensure_chat(self, chat_id: int | str) -> TelegramActiveSession:
        """Ensure and return the active pointer row for a chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            Active pointer row.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        chat_key = str(chat_id)
        async with self._lock:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO telegram_chat_active(chat_id)
                    VALUES(%s)
                    ON CONFLICT(chat_id) DO NOTHING
                    """,
                    (chat_key,),
                )
        row = await self.get_active(chat_id)
        if row is None:
            raise RuntimeError("failed to create Telegram active session row")
        return row

    async def get_active(self, chat_id: int | str) -> TelegramActiveSession | None:
        """Return the active pointer row for a chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            Active pointer row, or ``None``.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT
                    chat_id,
                    active_thread_id,
                    active_started_at,
                    migrated_legacy,
                    migration_last_error,
                    close_requested_reason,
                    close_requested_at
                FROM telegram_chat_active
                WHERE chat_id = %s
                """,
                (str(chat_id),),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._active_session_from_row(row)

    async def create_session(self, chat_id: int | str) -> str:
        """Create and activate a new rotated session thread for a chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            Newly active OpenCouch thread id.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        chat_key = str(chat_id)
        started_at = _utc_now_iso()
        for _ in range(3):
            thread_id = telegram_session_thread_id(chat_key, _generate_ulid())
            try:
                async with self._lock:
                    async with conn.transaction():
                        async with conn.cursor() as cursor:
                            await cursor.execute(
                                """
                                INSERT INTO telegram_chat_sessions(
                                    thread_id,
                                    chat_id,
                                    started_at
                                )
                                VALUES(%s, %s, %s)
                                """,
                                (thread_id, chat_key, started_at),
                            )
                            await cursor.execute(
                                """
                                INSERT INTO telegram_chat_active(
                                    chat_id,
                                    active_thread_id,
                                    active_started_at,
                                    migrated_legacy,
                                    migration_last_error,
                                    migration_updated_at,
                                    close_requested_reason,
                                    close_requested_at
                                )
                                VALUES(%s, %s, %s, 'finalized', NULL, %s, NULL, NULL)
                                ON CONFLICT(chat_id) DO UPDATE SET
                                    active_thread_id = excluded.active_thread_id,
                                    active_started_at = excluded.active_started_at,
                                    close_requested_reason = NULL,
                                    close_requested_at = NULL
                                """,
                                (chat_key, thread_id, started_at, started_at),
                            )
                return thread_id
            except psycopg.errors.UniqueViolation:
                continue
        raise RuntimeError(f"failed to allocate Telegram session id for {chat_key}")

    async def set_legacy_migration_state(
        self,
        chat_id: int | str,
        *,
        state: str,
        error: str | None = None,
    ) -> None:
        """Record legacy thread migration state.

        Args:
            chat_id: Telegram chat identifier.
            state: Legacy migration state.
            error: Optional migration error.

        Returns:
            None.
        """

        if state not in _LEGACY_MIGRATION_STATES:
            raise ValueError(f"unsupported legacy migration state: {state}")

        await self.ensure_started()
        conn = self._ensure_conn()
        async with self._lock:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO telegram_chat_active(
                        chat_id,
                        migrated_legacy,
                        migration_last_error,
                        migration_updated_at
                    )
                    VALUES(%s, %s, %s, %s)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        migrated_legacy = excluded.migrated_legacy,
                        migration_last_error = excluded.migration_last_error,
                        migration_updated_at = excluded.migration_updated_at
                    """,
                    (str(chat_id), state, error, _utc_now_iso()),
                )

    async def reset_finalizing_legacy_migrations(self) -> None:
        """Mark interrupted legacy migrations as failed on startup.

        Returns:
            None.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with self._lock:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE telegram_chat_active
                    SET
                        migrated_legacy = 'failed',
                        migration_last_error = COALESCE(
                            migration_last_error,
                            'startup recovered interrupted legacy migration'
                        ),
                        migration_updated_at = %s
                    WHERE migrated_legacy = 'finalizing'
                    """,
                    (_utc_now_iso(),),
                )

    async def set_pending_close(self, chat_id: int | str, reason: str) -> None:
        """Persist a pending close request for the chat's active pointer.

        Args:
            chat_id: Telegram chat identifier.
            reason: Close reason.

        Returns:
            None.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with self._lock:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO telegram_chat_active(
                        chat_id,
                        close_requested_reason,
                        close_requested_at
                    )
                    VALUES(%s, %s, %s)
                    ON CONFLICT(chat_id) DO UPDATE SET
                        close_requested_reason = excluded.close_requested_reason,
                        close_requested_at = excluded.close_requested_at
                    """,
                    (str(chat_id), reason, _utc_now_iso()),
                )

    async def clear_pending_close(self, chat_id: int | str) -> None:
        """Clear a pending close request for a chat.

        Args:
            chat_id: Telegram chat identifier.

        Returns:
            None.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with self._lock:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE telegram_chat_active
                    SET close_requested_reason = NULL, close_requested_at = NULL
                    WHERE chat_id = %s
                    """,
                    (str(chat_id),),
                )

    async def close_thread(
        self,
        chat_id: int | str,
        thread_id: str,
        reason: str,
    ) -> None:
        """Mark a session closed and clear the active pointer if it matches.

        Args:
            chat_id: Telegram chat identifier.
            thread_id: Session thread id.
            reason: Close reason.

        Returns:
            None.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        closed_at = _utc_now_iso()
        async with self._lock:
            async with conn.transaction():
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        """
                        UPDATE telegram_chat_sessions
                        SET
                            closed_at = COALESCE(closed_at, %s),
                            closed_reason = COALESCE(closed_reason, %s)
                        WHERE thread_id = %s
                        """,
                        (closed_at, reason, thread_id),
                    )
                    await cursor.execute(
                        """
                        UPDATE telegram_chat_active
                        SET
                            active_thread_id = NULL,
                            active_started_at = NULL,
                            close_requested_reason = NULL,
                            close_requested_at = NULL
                        WHERE chat_id = %s AND active_thread_id = %s
                        """,
                        (str(chat_id), thread_id),
                    )

    async def list_active(self) -> list[TelegramActiveSession]:
        """List all active pointer rows.

        Returns:
            Active pointer rows.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT
                    chat_id,
                    active_thread_id,
                    active_started_at,
                    migrated_legacy,
                    migration_last_error,
                    close_requested_reason,
                    close_requested_at
                FROM telegram_chat_active
                ORDER BY chat_id
                """
            )
            rows = await cursor.fetchall()
        return [self._active_session_from_row(row) for row in rows]

    async def list_unclosed_sessions(self) -> list[TelegramClosedSession]:
        """List session rows that have not been closed in the registry.

        Returns:
            Open session rows.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT chat_id, thread_id, started_at
                FROM telegram_chat_sessions
                WHERE closed_at IS NULL
                ORDER BY started_at
                """
            )
            rows = await cursor.fetchall()
        return [
            TelegramClosedSession(
                chat_id=str(row["chat_id"]),
                thread_id=str(row["thread_id"]),
                closed_at=str(row["started_at"]),
            )
            for row in rows
        ]

    async def list_reclaimable(self, grace: timedelta) -> list[TelegramClosedSession]:
        """List closed sessions old enough for checkpoint reclaim.

        Args:
            grace: Minimum age since close before reclaim.

        Returns:
            Closed session rows.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        cutoff = datetime.now(UTC) - grace
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT chat_id, thread_id, closed_at
                FROM telegram_chat_sessions
                WHERE
                    closed_at IS NOT NULL
                    AND reclaimed_at IS NULL
                    AND reclaim_stuck_at IS NULL
                ORDER BY closed_at
                """
            )
            rows = await cursor.fetchall()

        reclaimable: list[TelegramClosedSession] = []
        for row in rows:
            closed_at = str(row["closed_at"])
            parsed = _parse_registry_timestamp(closed_at)
            if parsed is None or parsed > cutoff:
                continue
            reclaimable.append(
                TelegramClosedSession(
                    chat_id=str(row["chat_id"]),
                    thread_id=str(row["thread_id"]),
                    closed_at=closed_at,
                )
            )
        return reclaimable

    async def mark_reclaim_result(
        self,
        thread_id: str,
        *,
        error: str | None = None,
    ) -> None:
        """Record checkpoint reclaim outcome.

        Args:
            thread_id: Session thread id.
            error: Optional reclaim error.

        Returns:
            None.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        now = _utc_now_iso()
        async with self._lock:
            if error is None:
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        """
                        UPDATE telegram_chat_sessions
                        SET
                            reclaim_started_at = COALESCE(reclaim_started_at, %s),
                            reclaim_attempts = reclaim_attempts + 1,
                            reclaimed_at = %s,
                            last_reclaim_error = NULL
                        WHERE thread_id = %s
                        """,
                        (now, now, thread_id),
                    )
                return

            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT reclaim_attempts, closed_at
                    FROM telegram_chat_sessions
                    WHERE thread_id = %s
                    """,
                    (thread_id,),
                )
                row = await cursor.fetchone()
                attempts = int(row["reclaim_attempts"] or 0) if row is not None else 0
                closed_at = (
                    _parse_registry_timestamp(str(row["closed_at"]))
                    if row is not None and row["closed_at"] is not None
                    else None
                )
                next_attempts = attempts + 1
                age_stuck = (
                    closed_at is not None
                    and datetime.now(UTC) - closed_at >= TELEGRAM_RECLAIM_STUCK_AGE
                )
                stuck_at = (
                    now
                    if next_attempts >= TELEGRAM_RECLAIM_STUCK_ATTEMPTS or age_stuck
                    else None
                )
                await cursor.execute(
                    """
                    UPDATE telegram_chat_sessions
                    SET
                        reclaim_started_at = COALESCE(reclaim_started_at, %s),
                        reclaim_attempts = reclaim_attempts + 1,
                        reclaim_stuck_at = COALESCE(reclaim_stuck_at, %s),
                        last_reclaim_error = %s
                    WHERE thread_id = %s
                    """,
                    (now, stuck_at, error, thread_id),
                )
