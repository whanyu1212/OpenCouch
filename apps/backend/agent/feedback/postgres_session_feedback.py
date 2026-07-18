"""Direct PostgreSQL implementation of the session-feedback protocol."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent.feedback.models import SessionFeedbackRecord
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.memory.hashing import extract_iso_date

logger = logging.getLogger(__name__)

SESSION_FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS session_feedback (
    insertion_order BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL,
    session_id_opaque TEXT NOT NULL,
    user_id_or_null TEXT,
    recorded_at TEXT NOT NULL,
    recorded_date TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('positive', 'negative', 'skip')),
    turn_count_at_end INTEGER NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('cli_end', 'cli_exit', 'api_end')),
    schema_version INTEGER NOT NULL DEFAULT 1,
    value JSONB NOT NULL
);
"""

SESSION_FEEDBACK_INDEX_SESSION_DDL = """
CREATE INDEX IF NOT EXISTS idx_feedback_session
    ON session_feedback(session_id_opaque);
"""

SESSION_FEEDBACK_INDEX_DATE_DDL = """
CREATE INDEX IF NOT EXISTS idx_feedback_recorded_date
    ON session_feedback(recorded_date);
"""

SESSION_FEEDBACK_SCHEMA_DDL: tuple[str, ...] = (
    SESSION_FEEDBACK_DDL,
    SESSION_FEEDBACK_INDEX_SESSION_DDL,
    SESSION_FEEDBACK_INDEX_DATE_DDL,
)


class PostgresSessionFeedbackBackend:
    """PostgreSQL-backed implementation of :class:`SessionFeedbackBackend`."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._connection: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self._closed = False
        self._connect_lock = asyncio.Lock()
        self._operation_lock = asyncio.Lock()

    async def _ensure_connection(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("PostgresSessionFeedbackBackend is closed.")
        if self._connection is not None:
            return self._connection

        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("PostgresSessionFeedbackBackend is closed.")
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
                        "PostgresSessionFeedbackBackend: connection close during "
                        "failed schema setup raised; ignoring",
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
                for ddl in SESSION_FEEDBACK_SCHEMA_DDL:
                    await cursor.execute(ddl)

    async def aappend(self, record: SessionFeedbackRecord) -> None:
        """Append one feedback record."""

        async with self._operation_lock:
            if self._closed:
                raise RuntimeError("PostgresSessionFeedbackBackend is closed.")
            recorded_date = extract_iso_date(record.recorded_at)
            conn = await self._ensure_connection()
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO session_feedback
                        (id, session_id_opaque, user_id_or_null, recorded_at,
                         recorded_date, label, turn_count_at_end, source,
                         schema_version, value)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.id,
                        record.session_id_opaque,
                        record.user_id_or_null,
                        record.recorded_at,
                        recorded_date,
                        record.label,
                        record.turn_count_at_end,
                        record.source,
                        record.schema_version,
                        Jsonb(record.model_dump(mode="json")),
                    ),
                )

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:
        """List feedback for one session in insertion order."""

        async with self._operation_lock:
            conn = await self._ensure_connection()
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT value
                    FROM session_feedback
                    WHERE session_id_opaque = %s
                    ORDER BY insertion_order ASC
                    """,
                    (session_id_opaque,),
                )
                rows = await cursor.fetchall()
        return [SessionFeedbackRecord.model_validate(row["value"]) for row in rows]

    async def arecord_count(self) -> int:
        """Count all feedback records, returning zero after close."""

        async with self._operation_lock:
            if self._closed:
                return 0
            conn = await self._ensure_connection()
            async with conn.cursor() as cursor:
                await cursor.execute("SELECT COUNT(*) AS count FROM session_feedback")
                row = await cursor.fetchone()
        return int(row["count"]) if row else 0

    async def apurge_before(self, cutoff: date) -> int:
        """Delete feedback older than the exclusive cutoff date."""

        async with self._operation_lock:
            if self._closed:
                return 0
            conn = await self._ensure_connection()
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "DELETE FROM session_feedback WHERE recorded_date < %s",
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
                            "PostgresSessionFeedbackBackend: connection close raised; "
                            "ignoring",
                            exc_info=True,
                        )
                    finally:
                        self._connection = None


if TYPE_CHECKING:
    _postgres_backend: SessionFeedbackBackend = PostgresSessionFeedbackBackend(
        "postgresql://example"
    )


__all__ = [
    "PostgresSessionFeedbackBackend",
    "SESSION_FEEDBACK_SCHEMA_DDL",
    "SESSION_FEEDBACK_DDL",
    "SESSION_FEEDBACK_INDEX_SESSION_DDL",
    "SESSION_FEEDBACK_INDEX_DATE_DDL",
]
