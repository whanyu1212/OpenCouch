"""Direct PostgreSQL implementation of the crisis-log protocol."""

from __future__ import annotations

import asyncio
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
    """PostgreSQL-backed implementation of :class:`CrisisLogBackend`."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._connection: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self._closed = False
        self._connect_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()

    async def _ensure_connection(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("PostgresCrisisLogBackend is closed.")
        if self._connection is not None:
            return self._connection

        async with self._connect_lock:
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
                try:
                    await conn.close()
                except Exception:
                    logger.warning(
                        "PostgresCrisisLogBackend: connection close during failed "
                        "schema setup raised; ignoring",
                        exc_info=True,
                    )
                raise
            self._connection = conn
            return conn

    @staticmethod
    async def _ensure_schema(
        conn: psycopg.AsyncConnection[dict[str, Any]],
    ) -> None:
        async with conn.transaction():
            async with conn.cursor() as cursor:
                for ddl in CRISIS_LOG_SCHEMA_DDL:
                    await cursor.execute(ddl)

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one crisis record."""

        async with self._operation_lock:
            if self._closed:
                raise RuntimeError("PostgresCrisisLogBackend is closed.")
            detected_date = extract_iso_date(record.detected_at)
            conn = await self._ensure_connection()
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
                        Jsonb(serialize_crisis_record(record)),
                    ),
                )

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List records for one date in insertion order."""

        async with self._operation_lock:
            conn = await self._ensure_connection()
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT value
                    FROM crisis_log
                    WHERE detected_date = %s
                    ORDER BY insertion_order ASC
                    """,
                    (day.isoformat(),),
                )
                rows = await cursor.fetchall()
        return [deserialize_crisis_record(row["value"]) for row in rows]

    async def arecord_count(self) -> int:
        """Count all crisis records, returning zero after close."""

        async with self._operation_lock:
            if self._closed:
                return 0
            conn = await self._ensure_connection()
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) AS count FROM crisis_log")
                row = await cursor.fetchone()
        return int(row["count"]) if row else 0

    async def apurge_before(self, cutoff: date) -> int:
        """Delete records older than the exclusive cutoff date."""

        async with self._operation_lock:
            if self._closed:
                return 0
            conn = await self._ensure_connection()
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM crisis_log WHERE detected_date < %s",
                    (cutoff.isoformat(),),
                )
                return int(cursor.rowcount or 0)

    async def aclose(self) -> None:
        """Close the backend without racing an active operation."""

        async with self._operation_lock:
            if self._closed:
                return
            async with self._connect_lock:
                if self._closed:
                    return
                self._closed = True
                if self._connection is not None:
                    try:
                        await self._connection.close()
                    except Exception:
                        logger.warning(
                            "PostgresCrisisLogBackend: connection close raised; "
                            "ignoring",
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
