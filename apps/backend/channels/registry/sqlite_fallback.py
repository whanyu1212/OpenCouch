"""Legacy SQLite Telegram session registry backend."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite

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


class SqliteTelegramSessionRegistry:
    """SQLite registry for Telegram chat-to-session thread pointers."""

    def __init__(self, sqlite_path: str | Path) -> None:
        """Initialize the registry.

        Args:
            sqlite_path: SQLite file path, or ``:memory:`` for tests.
        """

        self.sqlite_path = Path(sqlite_path)
        self._conn: aiosqlite.Connection | None = None
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
            if self.sqlite_path != Path(":memory:"):
                self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = await aiosqlite.connect(str(self.sqlite_path))
            await self._conn.execute("PRAGMA journal_mode=WAL")
            await self._conn.execute("PRAGMA busy_timeout=5000")
            await self._conn.execute(
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
            await self._conn.execute(
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
            await self._conn.execute(
                """
                UPDATE telegram_chat_active
                SET migrated_legacy = CASE
                    WHEN migrated_legacy IN (1, '1', 'true', 'yes') THEN 'finalized'
                    WHEN migrated_legacy IN (0, '0', 'false', 'no', '') THEN 'pending'
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
            await self._conn.commit()

    async def aclose(self) -> None:
        """Close the registry connection.

        Returns:
            None.
        """

        if self._conn is None:
            return
        await self._conn.close()
        self._conn = None

    def _ensure_conn(self) -> aiosqlite.Connection:
        """Return an opened registry connection.

        Returns:
            The active SQLite connection.

        Raises:
            RuntimeError: If the registry has not been started.
        """

        if self._conn is None:
            raise RuntimeError("SqliteTelegramSessionRegistry has not been started.")
        return self._conn

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
            await conn.execute(
                """
                INSERT INTO telegram_chat_active(chat_id)
                VALUES(?)
                ON CONFLICT(chat_id) DO NOTHING
                """,
                (chat_key,),
            )
            await conn.commit()
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
        async with conn.execute(
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
            WHERE chat_id = ?
            """,
            (str(chat_id),),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return TelegramActiveSession(
            chat_id=str(row[0]),
            active_thread_id=str(row[1]) if row[1] is not None else None,
            active_started_at=str(row[2]) if row[2] is not None else None,
            legacy_migration_state=_legacy_migration_state(row[3]),
            migration_last_error=str(row[4]) if row[4] is not None else None,
            close_requested_reason=str(row[5]) if row[5] is not None else None,
            close_requested_at=str(row[6]) if row[6] is not None else None,
        )

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
                    await conn.execute(
                        """
                        INSERT INTO telegram_chat_sessions(
                            thread_id,
                            chat_id,
                            started_at
                        )
                        VALUES(?, ?, ?)
                        """,
                        (thread_id, chat_key, started_at),
                    )
                    await conn.execute(
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
                        VALUES(?, ?, ?, 'finalized', NULL, ?, NULL, NULL)
                        ON CONFLICT(chat_id) DO UPDATE SET
                            active_thread_id = excluded.active_thread_id,
                            active_started_at = excluded.active_started_at,
                            close_requested_reason = NULL,
                            close_requested_at = NULL
                        """,
                        (chat_key, thread_id, started_at, started_at),
                    )
                    await conn.commit()
                return thread_id
            except aiosqlite.IntegrityError:
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
            await conn.execute(
                """
                INSERT INTO telegram_chat_active(
                    chat_id,
                    migrated_legacy,
                    migration_last_error,
                    migration_updated_at
                )
                VALUES(?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    migrated_legacy = excluded.migrated_legacy,
                    migration_last_error = excluded.migration_last_error,
                    migration_updated_at = excluded.migration_updated_at
                """,
                (str(chat_id), state, error, _utc_now_iso()),
            )
            await conn.commit()

    async def reset_finalizing_legacy_migrations(self) -> None:
        """Mark interrupted legacy migrations as failed on startup.

        Returns:
            None.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with self._lock:
            await conn.execute(
                """
                UPDATE telegram_chat_active
                SET
                    migrated_legacy = 'failed',
                    migration_last_error = COALESCE(
                        migration_last_error,
                        'startup recovered interrupted legacy migration'
                    ),
                    migration_updated_at = ?
                WHERE migrated_legacy = 'finalizing'
                """,
                (_utc_now_iso(),),
            )
            await conn.commit()

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
            await conn.execute(
                """
                INSERT INTO telegram_chat_active(
                    chat_id,
                    close_requested_reason,
                    close_requested_at
                )
                VALUES(?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    close_requested_reason = excluded.close_requested_reason,
                    close_requested_at = excluded.close_requested_at
                """,
                (str(chat_id), reason, _utc_now_iso()),
            )
            await conn.commit()

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
            await conn.execute(
                """
                UPDATE telegram_chat_active
                SET close_requested_reason = NULL, close_requested_at = NULL
                WHERE chat_id = ?
                """,
                (str(chat_id),),
            )
            await conn.commit()

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
            await conn.execute(
                """
                UPDATE telegram_chat_sessions
                SET
                    closed_at = COALESCE(closed_at, ?),
                    closed_reason = COALESCE(closed_reason, ?)
                WHERE thread_id = ?
                """,
                (closed_at, reason, thread_id),
            )
            await conn.execute(
                """
                UPDATE telegram_chat_active
                SET
                    active_thread_id = NULL,
                    active_started_at = NULL,
                    close_requested_reason = NULL,
                    close_requested_at = NULL
                WHERE chat_id = ? AND active_thread_id = ?
                """,
                (str(chat_id), thread_id),
            )
            await conn.commit()

    async def list_active(self) -> list[TelegramActiveSession]:
        """List all active pointer rows.

        Returns:
            Active pointer rows.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with conn.execute(
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
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            TelegramActiveSession(
                chat_id=str(row[0]),
                active_thread_id=str(row[1]) if row[1] is not None else None,
                active_started_at=str(row[2]) if row[2] is not None else None,
                legacy_migration_state=_legacy_migration_state(row[3]),
                migration_last_error=str(row[4]) if row[4] is not None else None,
                close_requested_reason=str(row[5]) if row[5] is not None else None,
                close_requested_at=str(row[6]) if row[6] is not None else None,
            )
            for row in rows
        ]

    async def list_unclosed_sessions(self) -> list[TelegramClosedSession]:
        """List session rows that have not been closed in the registry.

        Returns:
            Open session rows.
        """

        await self.ensure_started()
        conn = self._ensure_conn()
        async with conn.execute(
            """
            SELECT chat_id, thread_id, started_at
            FROM telegram_chat_sessions
            WHERE closed_at IS NULL
            ORDER BY started_at
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            TelegramClosedSession(
                chat_id=str(row[0]),
                thread_id=str(row[1]),
                closed_at=str(row[2]),
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
        async with conn.execute(
            """
            SELECT chat_id, thread_id, closed_at
            FROM telegram_chat_sessions
            WHERE
                closed_at IS NOT NULL
                AND reclaimed_at IS NULL
                AND reclaim_stuck_at IS NULL
            ORDER BY closed_at
            """
        ) as cursor:
            rows = await cursor.fetchall()

        reclaimable: list[TelegramClosedSession] = []
        for row in rows:
            closed_at = str(row[2])
            parsed = _parse_registry_timestamp(closed_at)
            if parsed is None or parsed > cutoff:
                continue
            reclaimable.append(
                TelegramClosedSession(
                    chat_id=str(row[0]),
                    thread_id=str(row[1]),
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
                await conn.execute(
                    """
                    UPDATE telegram_chat_sessions
                    SET
                        reclaim_started_at = COALESCE(reclaim_started_at, ?),
                        reclaim_attempts = reclaim_attempts + 1,
                        reclaimed_at = ?,
                        last_reclaim_error = NULL
                    WHERE thread_id = ?
                    """,
                    (now, now, thread_id),
                )
            else:
                async with conn.execute(
                    """
                    SELECT reclaim_attempts, closed_at
                    FROM telegram_chat_sessions
                    WHERE thread_id = ?
                    """,
                    (thread_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                attempts = int(row[0] or 0) if row is not None else 0
                closed_at = (
                    _parse_registry_timestamp(str(row[1]))
                    if row is not None and row[1] is not None
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
                await conn.execute(
                    """
                    UPDATE telegram_chat_sessions
                    SET
                        reclaim_started_at = COALESCE(reclaim_started_at, ?),
                        reclaim_attempts = reclaim_attempts + 1,
                        reclaim_stuck_at = COALESCE(reclaim_stuck_at, ?),
                        last_reclaim_error = ?
                    WHERE thread_id = ?
                    """,
                    (now, stuck_at, error, thread_id),
                )
            await conn.commit()
