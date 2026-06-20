"""SQLite-backed implementation of the session-feedback protocol.

This backend persists ``SessionFeedbackRecord`` rows for local and
synced runtimes. Indexed columns support session lookups and retention
purges; the full serialized record remains in the ``value`` JSON column
for forward compatibility.

Feedback rows are intentionally session-keyed because they are created
at session close, not by model execution during normal turn handling.

The store body is shared with the PostgreSQL backend via
:class:`~agent.storage.kv_store.KvStore`; only the SQLite DDL (TEXT value
column, ``INTEGER PRIMARY KEY AUTOINCREMENT``) and the SQLite dialect live here.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from agent.feedback.models import SessionFeedbackRecord
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.feedback.session_feedback_store import (
    build_session_feedback_table_config,
)
from agent.storage.kv_store import KvStore
from agent.storage.sqldialect import SQLITE_DIALECT

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
        self._store: KvStore[SessionFeedbackRecord] = KvStore(
            target=str(self.sqlite_path),
            dialect=SQLITE_DIALECT,
            config=build_session_feedback_table_config(SESSION_FEEDBACK_SCHEMA_DDL),
            backend_label="SqliteSessionFeedbackBackend",
        )

    @property
    def _connection(self):  # noqa: ANN202 - mirrors the store's lazy handle
        """Expose the lazily-opened connection (None until first use)."""

        return self._store._connection  # noqa: SLF001

    async def aappend(self, record: SessionFeedbackRecord) -> None:
        """Append one SQLite-backed feedback record.

        Args:
            record (SessionFeedbackRecord): Feedback record to append.

        Returns:
            None: Writes the record to SQLite.
        """

        await self._store.aappend(record)

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:
        """List SQLite-backed feedback records for one session.

        Args:
            session_id_opaque (str): Opaque session identifier to query.

        Returns:
            list[SessionFeedbackRecord]: Records for the session in insertion order.
        """

        return await self._store.alist_by_key(session_id_opaque)

    async def arecord_count(self) -> int:
        """Count SQLite-backed feedback records.

        Returns:
            int: Total feedback record count.
        """

        return await self._store.arecord_count()

    async def apurge_before(self, cutoff: date) -> int:
        """Purge SQLite-backed feedback records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        return await self._store.apurge_before(cutoff)

    async def aclose(self) -> None:
        """Close the SQLite feedback backend.

        Returns:
            None: Marks the backend closed and releases the connection.
        """

        await self._store.aclose()


if TYPE_CHECKING:
    _sqlite_backend: SessionFeedbackBackend = SqliteSessionFeedbackBackend(":memory:")


__all__ = [
    "SqliteSessionFeedbackBackend",
    "SESSION_FEEDBACK_SCHEMA_DDL",
    "SESSION_FEEDBACK_DDL",
    "SESSION_FEEDBACK_INDEX_SESSION_DDL",
    "SESSION_FEEDBACK_INDEX_DATE_DDL",
]
