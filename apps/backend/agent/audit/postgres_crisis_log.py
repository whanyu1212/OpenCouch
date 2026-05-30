"""PostgreSQL-backed implementation of the crisis-log protocol.

This backend is the primary persistent crisis-log implementation and keeps
query semantics compatible with the legacy SQLite backend. Query semantics intentionally
match the SQLite backend: records are bucketed by ``detected_date`` and
returned in insertion order within each date bucket.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent.audit.crisis_log import CrisisLogBackend
from agent.audit.crisis_log_serialization import (
    deserialize_crisis_record,
    serialize_crisis_record,
)
from agent.audit.models import CrisisLogRecord
from agent.memory.hashing import extract_iso_date

logger = logging.getLogger(__name__)

CRISIS_LOG_DDL = """
CREATE TABLE IF NOT EXISTS crisis_log (
    insertion_order BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL UNIQUE,
    session_id_opaque TEXT NOT NULL,
    user_id_or_null TEXT,
    detected_at TEXT NOT NULL,
    detected_date TEXT NOT NULL,
    level INTEGER NOT NULL CHECK (level IN (0, 1, 2, 3)),
    value JSONB NOT NULL
);
"""

CRISIS_LOG_INDEX_DATE_DDL = """
CREATE INDEX IF NOT EXISTS idx_crisis_detected_date
    ON crisis_log(detected_date);
"""

CRISIS_LOG_INDEX_SESSION_DDL = """
CREATE INDEX IF NOT EXISTS idx_crisis_session
    ON crisis_log(session_id_opaque);
"""

CRISIS_LOG_SCHEMA_DDL: tuple[str, ...] = (
    CRISIS_LOG_DDL,
    CRISIS_LOG_INDEX_DATE_DDL,
    CRISIS_LOG_INDEX_SESSION_DDL,
)


class PostgresCrisisLogBackend:
    """PostgreSQL-backed implementation of :class:`CrisisLogBackend`.

    The connection opens lazily on first async use, then stays attached
    to the backend until ``aclose``. Each runtime instance should own its
    own backend instance; the class is not thread-safe.
    """

    def __init__(self, dsn: str) -> None:
        """Initialize the PostgreSQL-backed crisis backend.

        Args:
            dsn (str): PostgreSQL connection string.

        Returns:
            None: Stores connection configuration for lazy initialization.
        """

        self.dsn = dsn
        self._connection: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self._closed = False

    async def _ensure_connection(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        """Open the PostgreSQL connection on first use.

        Returns:
            psycopg.AsyncConnection[dict[str, Any]]: Shared connection for the
                backend instance.
        """

        if self._closed:
            raise RuntimeError("PostgresCrisisLogBackend is closed.")
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
        """Ensure the PostgreSQL crisis-log schema exists.

        Args:
            conn (psycopg.AsyncConnection[dict[str, Any]]): Open PostgreSQL
                connection.

        Returns:
            None: Applies schema DDL.
        """

        async with conn.transaction():
            async with conn.cursor() as cursor:
                for ddl in CRISIS_LOG_SCHEMA_DDL:
                    await cursor.execute(ddl)

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one PostgreSQL-backed crisis record.

        Args:
            record (CrisisLogRecord): Crisis event record to append.

        Returns:
            None: Writes the record to PostgreSQL.
        """

        conn = await self._ensure_connection()
        detected_date = extract_iso_date(record.detected_at)
        serialized = serialize_crisis_record(record)

        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO crisis_log
                    (id, session_id_opaque, user_id_or_null, detected_at,
                     detected_date, level, value)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    record.id,
                    record.session_id_opaque,
                    record.user_id_or_null,
                    record.detected_at,
                    detected_date,
                    record.level,
                    Jsonb(serialized),
                ),
            )

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List PostgreSQL-backed crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Records for the day in insertion order.
        """

        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT value FROM crisis_log
                WHERE detected_date = %s
                ORDER BY insertion_order ASC
                """,
                (day.isoformat(),),
            )
            rows = await cursor.fetchall()
        return [deserialize_crisis_record(row["value"]) for row in rows]

    async def arecord_count(self) -> int:
        """Count PostgreSQL-backed crisis records.

        Returns:
            int: Total crisis-log record count.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT COUNT(*) AS count FROM crisis_log")
            row = await cursor.fetchone()
        return int(row["count"]) if row else 0

    async def apurge_before(self, cutoff: date) -> int:
        """Purge PostgreSQL-backed crisis records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                DELETE FROM crisis_log
                WHERE detected_date < %s
                """,
                (cutoff.isoformat(),),
            )
            return int(cursor.rowcount or 0)

    async def aclose(self) -> None:
        """Close the PostgreSQL crisis backend.

        Returns:
            None: Marks the backend closed and releases the connection.
        """

        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                logger.warning(
                    "PostgresCrisisLogBackend: connection close raised; ignoring",
                    exc_info=True,
                )
            finally:
                self._connection = None


if TYPE_CHECKING:
    _postgres_backend: CrisisLogBackend = PostgresCrisisLogBackend(
        "postgresql://example"
    )


__all__ = [
    "PostgresCrisisLogBackend",
    "CRISIS_LOG_SCHEMA_DDL",
    "CRISIS_LOG_DDL",
    "CRISIS_LOG_INDEX_DATE_DDL",
    "CRISIS_LOG_INDEX_SESSION_DDL",
]
