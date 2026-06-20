"""PostgreSQL-backed implementation of the session-feedback protocol.

This backend is the primary persistent session-feedback implementation and
keeps query semantics compatible with the legacy SQLite backend. Query semantics intentionally match
the SQLite backend: records are keyed by ``session_id_opaque`` and returned in
insertion order within each session bucket.

The store body is shared with the SQLite backend via
:class:`~agent.storage.kv_store.KvStore`; only the PostgreSQL DDL (JSONB value
column, ``BIGSERIAL`` primary key) and the PostgreSQL dialect live here.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from agent.feedback.models import SessionFeedbackRecord
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.feedback.session_feedback_store import (
    build_session_feedback_table_config,
)
from agent.storage.kv_store import KvStore
from agent.storage.sqldialect import POSTGRES_DIALECT

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
    """PostgreSQL-backed implementation of :class:`SessionFeedbackBackend`.

    The connection opens lazily on first async use, then stays attached
    to the backend until ``aclose``. Each runtime instance should own its
    own backend instance; the class is not thread-safe.
    """

    def __init__(self, dsn: str) -> None:
        """Initialize the PostgreSQL-backed feedback backend.

        Args:
            dsn (str): PostgreSQL connection string.

        Returns:
            None: Stores connection configuration for lazy initialization.
        """

        self.dsn = dsn
        self._store: KvStore[SessionFeedbackRecord] = KvStore(
            target=dsn,
            dialect=POSTGRES_DIALECT,
            config=build_session_feedback_table_config(SESSION_FEEDBACK_SCHEMA_DDL),
            backend_label="PostgresSessionFeedbackBackend",
        )

    @property
    def _connection(self):  # noqa: ANN202 - mirrors the store's lazy handle
        """Expose the lazily-opened connection (None until first use)."""

        return self._store._connection  # noqa: SLF001

    async def aappend(self, record: SessionFeedbackRecord) -> None:
        """Append one PostgreSQL-backed feedback record.

        Args:
            record (SessionFeedbackRecord): Feedback record to append.

        Returns:
            None: Writes the record to PostgreSQL.
        """

        await self._store.aappend(record)

    async def alist_by_session(
        self, session_id_opaque: str
    ) -> list[SessionFeedbackRecord]:
        """List PostgreSQL-backed feedback records for one session.

        Args:
            session_id_opaque (str): Opaque session identifier to query.

        Returns:
            list[SessionFeedbackRecord]: Records for the session in insertion order.
        """

        return await self._store.alist_by_key(session_id_opaque)

    async def arecord_count(self) -> int:
        """Count PostgreSQL-backed feedback records.

        Returns:
            int: Total feedback record count.
        """

        return await self._store.arecord_count()

    async def apurge_before(self, cutoff: date) -> int:
        """Purge PostgreSQL-backed feedback records older than a cutoff date.

        Args:
            cutoff (date): Exclusive cutoff date.

        Returns:
            int: Number of records deleted.
        """

        return await self._store.apurge_before(cutoff)

    async def aclose(self) -> None:
        """Close the PostgreSQL feedback backend.

        Returns:
            None: Marks the backend closed and releases the connection.
        """

        await self._store.aclose()


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
