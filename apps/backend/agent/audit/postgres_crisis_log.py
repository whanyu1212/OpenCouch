"""PostgreSQL-backed implementation of the crisis-log protocol.

This backend is the primary persistent crisis-log implementation and keeps
query semantics compatible with the legacy SQLite backend. Query semantics intentionally
match the SQLite backend: records are bucketed by ``detected_date`` and
returned in insertion order within each date bucket.

The store body is shared with the SQLite backend via
:class:`~agent.storage.kv_store.KvStore`; only the PostgreSQL DDL (JSONB value
column, ``BIGSERIAL`` primary key) and the PostgreSQL dialect live here.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from agent.audit.crisis_log import CrisisLogBackend
from agent.audit.crisis_log_store import build_crisis_log_table_config
from agent.audit.models import CrisisLogRecord
from agent.storage.kv_store import KvStore
from agent.storage.sqldialect import POSTGRES_DIALECT

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
        self._store: KvStore[CrisisLogRecord] = KvStore(
            target=dsn,
            dialect=POSTGRES_DIALECT,
            config=build_crisis_log_table_config(CRISIS_LOG_SCHEMA_DDL),
            backend_label="PostgresCrisisLogBackend",
        )

    @property
    def _connection(self):  # noqa: ANN202 - mirrors the store's lazy handle
        """Expose the lazily-opened connection (None until first use)."""

        return self._store._connection  # noqa: SLF001

    async def aappend(self, record: CrisisLogRecord) -> None:
        """Append one PostgreSQL-backed crisis record.

        Args:
            record (CrisisLogRecord): Crisis event record to append.

        Returns:
            None: Writes the record to PostgreSQL.
        """

        await self._store.aappend(record)

    async def alist_by_date(self, day: date) -> list[CrisisLogRecord]:
        """List PostgreSQL-backed crisis records for one date.

        Args:
            day (date): Calendar day to query.

        Returns:
            list[CrisisLogRecord]: Records for the day in insertion order.
        """

        return await self._store.alist_by_key(day.isoformat())

    async def arecord_count(self) -> int:
        """Count PostgreSQL-backed crisis records.

        Returns:
            int: Total crisis-log record count.
        """

        return await self._store.arecord_count()

    async def apurge_before(self, cutoff: date) -> int:
        """Purge PostgreSQL-backed crisis records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        return await self._store.apurge_before(cutoff)

    async def aclose(self) -> None:
        """Close the PostgreSQL crisis backend.

        Returns:
            None: Marks the backend closed and releases the connection.
        """

        await self._store.aclose()


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
