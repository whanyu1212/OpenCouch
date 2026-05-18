"""SQLite-backed implementation of the crisis-log protocol.

This backend persists ``CrisisLogRecord`` rows for local and synced
runtimes. Indexed columns support date/session queries and retention
purges; the full serialized record remains in the ``value`` JSON column
for forward compatibility.

Crisis-response side effects append records. ``apurge_before`` exists for
explicit operator or maintenance retention paths.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

from agent.audit.crisis_log import CrisisLogBackend
from agent.audit.models import CrisisLogRecord

logger = logging.getLogger(__name__)


CRISIS_LOG_DDL = """
CREATE TABLE IF NOT EXISTS crisis_log (
    insertion_order INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    session_id_opaque TEXT NOT NULL,
    user_id_or_null TEXT,
    detected_at TEXT NOT NULL,
    detected_date TEXT NOT NULL,
    level INTEGER NOT NULL CHECK (level IN (0, 1, 2, 3)),
    value TEXT NOT NULL
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


class SqliteCrisisLogBackend:
    """SQLite-backed implementation of :class:`CrisisLogBackend`.

    The connection opens lazily on first async use, then stays attached
    to the backend until ``aclose``. Each runtime instance should own its
    own backend instance; the class is not thread-safe.
    """

    def __init__(self, sqlite_path: str | Path) -> None:
        """Initialize the SQLite-backed crisis backend.

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
            raise RuntimeError("SqliteCrisisLogBackend is closed.")
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
        """Ensure the SQLite crisis-log schema exists.

        Args:
            conn (aiosqlite.Connection): Open SQLite connection.

        Returns:
            None: Applies schema DDL.
        """

        for ddl in CRISIS_LOG_SCHEMA_DDL:
            await conn.execute(ddl)
        await conn.commit()

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one SQLite-backed crisis record.

        Args:
            record (CrisisLogRecord): Crisis event record to append.

        Returns:
            None: Writes the record to SQLite.
        """

        conn = await self._ensure_connection()
        detected_date = self._extract_date_prefix(record.detected_at)
        serialized = record.model_dump(mode="json")
        value_json = json.dumps(serialized, default=str)

        await conn.execute(
            """
            INSERT INTO crisis_log
                (id, session_id_opaque, user_id_or_null, detected_at,
                 detected_date, level, value)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.session_id_opaque,
                record.user_id_or_null,
                record.detected_at,
                detected_date,
                record.level,
                value_json,
            ),
        )
        await conn.commit()

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List SQLite-backed crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Records for the day in insertion order.
        """

        conn = await self._ensure_connection()
        date_prefix = day.isoformat()
        async with conn.execute(
            """
            SELECT value FROM crisis_log
            WHERE detected_date = ?
            ORDER BY insertion_order ASC
            """,
            (date_prefix,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            CrisisLogRecord.model_validate(json.loads(row["value"])) for row in rows
        ]

    async def arecord_count(self) -> int:
        """Count SQLite-backed crisis records.

        Returns:
            int: Total crisis-log record count.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        async with conn.execute("SELECT COUNT(*) FROM crisis_log") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def apurge_before(self, cutoff: date) -> int:
        """Purge SQLite-backed crisis records older than a cutoff date.

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
            DELETE FROM crisis_log
            WHERE detected_date < ?
            """,
            (cutoff_str,),
        )
        await conn.commit()
        return int(cursor.rowcount or 0)

    async def aclose(self) -> None:
        """Close the SQLite crisis backend.

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
                    "SqliteCrisisLogBackend: connection close raised; ignoring",
                    exc_info=True,
                )
            finally:
                self._connection = None

    @staticmethod
    def _extract_date_prefix(detected_at: str) -> str:
        """Extract the date prefix from an ISO-8601 timestamp.

        Args:
            detected_at (str): ISO-8601 crisis detection timestamp.

        Returns:
            str: ``YYYY-MM-DD`` date prefix.
        """

        date_prefix = detected_at.split("T", 1)[0]
        date.fromisoformat(date_prefix)
        return date_prefix


if TYPE_CHECKING:
    _sqlite_backend: CrisisLogBackend = SqliteCrisisLogBackend(":memory:")


__all__ = [
    "SqliteCrisisLogBackend",
    "CRISIS_LOG_SCHEMA_DDL",
    "CRISIS_LOG_DDL",
    "CRISIS_LOG_INDEX_DATE_DDL",
    "CRISIS_LOG_INDEX_SESSION_DDL",
]
