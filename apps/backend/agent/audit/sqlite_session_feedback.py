"""SQLite-backed implementation of the session-feedback protocol.

This backend persists ``SessionFeedbackRecord`` rows for local and
synced runtimes. Indexed columns support session lookups and retention
purges; the full serialized record remains in the ``value`` JSON column
for forward compatibility.

Feedback rows are intentionally session-keyed because they are created
at session close, not by the response graph during normal turn handling.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from agent.audit.session_feedback import SessionFeedbackBackend
from agent.audit.models import SessionFeedbackRecord

logger = logging.getLogger(__name__)


SESSION_FEEDBACK_DDL = """
CREATE TABLE IF NOT EXISTS session_feedback (
    insertion_order INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL,
    session_id_opaque TEXT NOT NULL,
    user_id_or_null TEXT,
    recorded_at TEXT NOT NULL,
    recorded_date TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('positive', 'negative', 'skip')),
    turn_count_at_end INTEGER NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('cli_end', 'cli_exit', 'api_end')),
    schema_version INTEGER NOT NULL DEFAULT 1,
    value TEXT NOT NULL
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


class SqliteSessionFeedbackBackend:
    """SQLite-backed implementation of :class:`SessionFeedbackBackend`.

    The connection opens lazily on first async use, then stays attached
    to the backend until ``aclose``. Each runtime instance should own its
    own backend instance; the class is not thread-safe.
    """

    def __init__(self, sqlite_path: str | Path) -> None:
        """Initialize the SQLite-backed feedback backend.

        Args:
            sqlite_path (str | Path): SQLite file path or ``":memory:"``.

        Returns:
            None: Stores connection configuration for lazy initialization.
        """

        self.sqlite_path = (
            Path(sqlite_path) if sqlite_path != ":memory:" else Path(":memory:")
        )
        self._connection: aiosqlite.Connection | None = None
        self._closed = False

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Open the SQLite connection on first use.

        Returns:
            aiosqlite.Connection: Shared connection for the backend instance.
        """

        if self._closed:
            raise RuntimeError("SqliteSessionFeedbackBackend is closed.")
        if self._connection is not None:
            return self._connection

        if str(self.sqlite_path) != ":memory:":
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(str(self.sqlite_path))
        self._connection.row_factory = aiosqlite.Row
        await self._ensure_schema(self._connection)
        return self._connection

    @staticmethod
    async def _ensure_schema(conn: aiosqlite.Connection) -> None:
        """Ensure the SQLite session-feedback schema exists.

        Args:
            conn (aiosqlite.Connection): Open SQLite connection.

        Returns:
            None: Applies schema DDL.
        """

        for ddl in SESSION_FEEDBACK_SCHEMA_DDL:
            await conn.execute(ddl)
        await conn.commit()

    async def aappend(self, record: SessionFeedbackRecord) -> None:
        """Append one SQLite-backed feedback record.

        Args:
            record (SessionFeedbackRecord): Feedback record to append.

        Returns:
            None: Writes the record to SQLite.
        """

        conn = await self._ensure_connection()
        recorded_date = self._extract_date_prefix(record.recorded_at)
        serialized = record.model_dump(mode="json")
        value_json = json.dumps(serialized, default=str)

        await conn.execute(
            """
            INSERT INTO session_feedback
                (id, session_id_opaque, user_id_or_null, recorded_at,
                 recorded_date, label, turn_count_at_end, source,
                 schema_version, value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                value_json,
            ),
        )
        await conn.commit()

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:
        """List SQLite-backed feedback records for one session.

        Args:
            session_id_opaque (str): Opaque session identifier to query.

        Returns:
            list[SessionFeedbackRecord]: Records for the session in insertion order.
        """

        conn = await self._ensure_connection()
        async with conn.execute(
            """
            SELECT value FROM session_feedback
            WHERE session_id_opaque = ?
            ORDER BY insertion_order ASC
            """,
            (session_id_opaque,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            SessionFeedbackRecord.model_validate(json.loads(row["value"]))
            for row in rows
        ]

    async def arecord_count(self) -> int:
        """Count SQLite-backed feedback records.

        Returns:
            int: Total feedback record count.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        async with conn.execute("SELECT COUNT(*) FROM session_feedback") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def apurge_before(self, cutoff: date) -> int:
        """Purge SQLite-backed feedback records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        cutoff_str = cutoff.isoformat()
        cursor = await conn.execute(
            """
            DELETE FROM session_feedback
            WHERE recorded_date < ?
            """,
            (cutoff_str,),
        )
        await conn.commit()
        return int(cursor.rowcount or 0)

    async def aclose(self) -> None:
        """Close the SQLite feedback backend.

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
                    "SqliteSessionFeedbackBackend: connection close raised; ignoring",
                    exc_info=True,
                )
            finally:
                self._connection = None

    @staticmethod
    def _extract_date_prefix(recorded_at: str) -> str:
        """Extract the date prefix from an ISO-8601 timestamp.

        Args:
            recorded_at (str): ISO-8601 feedback timestamp.

        Returns:
            str: ``YYYY-MM-DD`` date prefix.
        """

        date_prefix = recorded_at.split("T", 1)[0]
        date.fromisoformat(date_prefix)
        return date_prefix


if TYPE_CHECKING:
    _sqlite_backend: SessionFeedbackBackend = SqliteSessionFeedbackBackend(":memory:")


__all__ = [
    "SqliteSessionFeedbackBackend",
    "SESSION_FEEDBACK_SCHEMA_DDL",
    "SESSION_FEEDBACK_DDL",
    "SESSION_FEEDBACK_INDEX_SESSION_DDL",
    "SESSION_FEEDBACK_INDEX_DATE_DDL",
]
