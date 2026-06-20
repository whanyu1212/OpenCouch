"""SQLite-backed implementation of the crisis-log protocol.

This backend persists ``CrisisLogRecord`` rows for local and synced
runtimes. Indexed columns support date/session queries and retention
purges; the full serialized record remains in the ``value`` JSON column
for forward compatibility.

Crisis-response side effects append records. ``apurge_before`` exists for
explicit operator or maintenance retention paths.

The store body is shared with the PostgreSQL backend via
:class:`~agent.storage.kv_store.KvStore`; only the SQLite DDL (TEXT value
column, ``INTEGER PRIMARY KEY AUTOINCREMENT``) and the SQLite dialect live here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from agent.audit.crisis_log import CrisisLogBackend
from agent.audit.models import CrisisLogRecord
from agent.storage.kv_store import KvStore
from agent.storage.sqldialect import SQLITE_DIALECT
from agent.audit.crisis_log_store import build_crisis_log_table_config

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
        self._store: KvStore[CrisisLogRecord] = KvStore(
            target=str(self.sqlite_path),
            dialect=SQLITE_DIALECT,
            config=build_crisis_log_table_config(CRISIS_LOG_SCHEMA_DDL),
            backend_label="SqliteCrisisLogBackend",
        )

    @property
    def _connection(self):  # noqa: ANN202 - mirrors the store's lazy handle
        """Expose the lazily-opened connection (None until first use)."""

        return self._store._connection  # noqa: SLF001

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one SQLite-backed crisis record.

        Args:
            record (CrisisLogRecord): Crisis event record to append.

        Returns:
            None: Writes the record to SQLite.
        """

        await self._store.aappend(record)

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List SQLite-backed crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Records for the day in insertion order.
        """

        return await self._store.alist_by_key(day.isoformat())

    async def arecord_count(self) -> int:
        """Count SQLite-backed crisis records.

        Returns:
            int: Total crisis-log record count.
        """

        return await self._store.arecord_count()

    async def apurge_before(self, cutoff: date) -> int:
        """Purge SQLite-backed crisis records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        return await self._store.apurge_before(cutoff)

    async def aclose(self) -> None:
        """Close the SQLite crisis backend.

        Returns:
            None: Marks the backend closed and releases the connection.
        """

        await self._store.aclose()


if TYPE_CHECKING:
    _sqlite_backend: CrisisLogBackend = SqliteCrisisLogBackend(":memory:")


__all__ = [
    "SqliteCrisisLogBackend",
    "CRISIS_LOG_SCHEMA_DDL",
    "CRISIS_LOG_DDL",
    "CRISIS_LOG_INDEX_DATE_DDL",
    "CRISIS_LOG_INDEX_SESSION_DDL",
]
