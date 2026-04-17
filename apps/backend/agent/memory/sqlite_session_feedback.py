"""SQLite-backed implementation of the :class:`SessionFeedbackBackend` protocol.

The v0.10 session-feedback collector needs a durable store so end-of-
session thumbs ratings survive CLI / server restarts. Structurally
parallel to :mod:`agent.memory.sqlite_crisis_log`:

- INCOGNITO mode → :class:`InMemorySessionFeedbackBackend`, nothing
  touches disk.
- LOCAL / SYNCED mode → this backend, paired with
  :class:`agent.memory.sqlite_store.SqliteMemoryStore` and
  :class:`agent.memory.sqlite_crisis_log.SqliteCrisisLogBackend`.

Design decisions mirrored from the crisis_log SQLite backend:

1. **Hybrid schema.** Discriminating columns (``id``,
   ``session_id_opaque``, ``recorded_at``, ``recorded_date``, ``label``,
   ``source``, ``turn_count_at_end``) sit alongside a ``value`` JSON
   column that holds the full serialized :class:`SessionFeedbackRecord`
   as the forward-compatible source of truth.

2. **Pre-computed ``recorded_date`` column.** Matches crisis_log's
   ``detected_date`` trick: store the ``YYYY-MM-DD`` prefix at insert
   time so :meth:`apurge_before` can use a B-tree index instead of
   evaluating ``date(recorded_at)`` per row.

3. **aiosqlite directly, no ORM.** Same rationale as crisis_log: the
   dependency is already in the tree, and the schema is simple.

4. **One connection per runtime lifetime, lazily opened.** Construction
   is cheap; the first async method opens the connection and runs the
   schema DDL. ``aclose`` tears it down.

Differences from crisis_log worth calling out:

- **No UNIQUE constraint on ``id``.** Phase 1 does not claim
  idempotency — two calls to ``record_session_feedback`` produce two
  rows with distinct fresh UUIDs. See ``session_feedback_plan.md`` §3
  (v4.1/v4.2 fix 3) for the reasoning. If a future phase adds
  idempotency, it will come via an explicit idempotency-key column
  rather than repurposing ``id``.

- **Session-keyed read pattern.** The primary read is
  :meth:`alist_by_session` (per opaque session id), matching the
  operational question "what feedback did this session produce?".
  crisis_log's primary read is by date; feedback's is by session.

- **``CHECK`` constraint on ``source``** in addition to ``label``.
  Both are short controlled vocabularies where a schema-level guard
  catches code-DB drift at insert time (e.g., a new Python enum
  member added without a matching migration).

- **Retention default of 180 days**, wider than crisis_log's 90,
  because feedback analytics benefit from a longer lookback window.
  The backend exposes :meth:`apurge_before` but the default is
  enforced by the CLI / scheduled cleanup caller, not by the backend
  itself.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import aiosqlite

from agent.memory.models import SessionFeedbackRecord
from agent.memory.session_feedback import SessionFeedbackBackend

logger = logging.getLogger(__name__)


# ─── Schema DDL ────────────────────────────────────────────────────────


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
# Schema notes:
#
# - ``insertion_order INTEGER PRIMARY KEY AUTOINCREMENT`` is the SQLite
#   primary key. It gives us stable chronological ordering for
#   ``ORDER BY insertion_order ASC`` within a session bucket, matching
#   the in-memory backend's "records in insertion order" contract.
#
# - ``id TEXT NOT NULL`` — intentionally NOT unique. Phase 1 does not
#   provide idempotency; see module docstring for the explicit
#   decision. ``id`` is an opaque UUID useful for external correlation
#   (log lines, future idempotency keys) but it has no database-level
#   uniqueness guarantee.
#
# - ``session_id_opaque TEXT NOT NULL`` — the SHA-256 of the session id,
#   indexed below for per-session lookups. Safe to persist in incognito
#   mode because it can't be reverse-mapped to a user.
#
# - ``recorded_date TEXT NOT NULL`` holds the ``YYYY-MM-DD`` prefix of
#   ``recorded_at``, computed at insert time. Enables B-tree index use
#   for :meth:`apurge_before` instead of per-row ``date(recorded_at)``.
#
# - ``label TEXT ... CHECK (...)`` and ``source TEXT ... CHECK (...)``
#   are controlled-vocabulary guards — defense against silent drift
#   between the Python enum and the DB schema.
#
# - ``value TEXT NOT NULL`` holds the full serialized
#   :class:`SessionFeedbackRecord` as JSON. The denormalized columns
#   enable fast filtering; this JSON is the source of truth.


SESSION_FEEDBACK_INDEX_SESSION_DDL = """
CREATE INDEX IF NOT EXISTS idx_feedback_session
    ON session_feedback(session_id_opaque);
"""

SESSION_FEEDBACK_INDEX_DATE_DDL = """
CREATE INDEX IF NOT EXISTS idx_feedback_recorded_date
    ON session_feedback(recorded_date);
"""


# Every statement uses ``IF NOT EXISTS`` so re-running is a safe no-op.
SESSION_FEEDBACK_SCHEMA_DDL: tuple[str, ...] = (
    SESSION_FEEDBACK_DDL,
    SESSION_FEEDBACK_INDEX_SESSION_DDL,
    SESSION_FEEDBACK_INDEX_DATE_DDL,
)


# ─── SqliteSessionFeedbackBackend ──────────────────────────────────────


class SqliteSessionFeedbackBackend:
    """SQLite-backed implementation of :class:`SessionFeedbackBackend`.

    Behaviorally identical to :class:`InMemorySessionFeedbackBackend`
    for all five method semantics; the only difference is that records
    persist to disk instead of living in a dict.

    Connection lifecycle (same pattern as
    :class:`agent.memory.sqlite_crisis_log.SqliteCrisisLogBackend`):

    - ``__init__`` stores the SQLite path but does NOT open the
      connection. Construction is cheap and doesn't touch disk.
    - The first async method call triggers lazy ``_ensure_connection``,
      which opens an aiosqlite connection and runs the schema DDL.
      Subsequent calls reuse the same connection.
    - ``aclose`` closes the connection and marks the backend as closed.
      Any further calls raise ``RuntimeError``.
    - The backend is **not** thread-safe. Each runtime instance should
      own its own backend; do not share an instance across concurrent
      callers.
    """

    def __init__(self, sqlite_path: str | Path) -> None:
        """Initialize the backend with a SQLite file path.

        Args:
            sqlite_path: Path to the SQLite file. Use ``":memory:"``
                for a pure in-RAM database (tests that want SQL
                semantics without disk writes). Parent directories for
                file paths are created lazily on first connection open.
        """

        self.sqlite_path = (
            Path(sqlite_path) if sqlite_path != ":memory:" else Path(":memory:")
        )
        self._connection: aiosqlite.Connection | None = None
        self._closed = False

    # ── Connection lifecycle ──────────────────────────────────────────

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Open the aiosqlite connection lazily on first use."""

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
        """Run the schema DDL. Idempotent — safe to call repeatedly."""

        for ddl in SESSION_FEEDBACK_SCHEMA_DDL:
            await conn.execute(ddl)
        await conn.commit()

    # ── Public interface (SessionFeedbackBackend protocol) ────────────

    async def aappend(self, record: SessionFeedbackRecord) -> None:
        """Append a feedback record.

        Writes the full record as JSON plus the indexed columns the
        schema needs for fast lookups. Computes ``recorded_date`` from
        ``record.recorded_at`` at insert time so the date index can
        serve :meth:`apurge_before` queries directly.

        No duplicate-id detection: Phase 1 allows two calls with the
        same ``record.id`` to produce two rows. If that ever becomes
        a problem, the fix is an explicit idempotency-key column, not
        a retrofitted UNIQUE constraint on ``id``.
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
        """Return all records for a given opaque session id.

        Uses the ``idx_feedback_session`` B-tree index for an O(log n)
        lookup. Records within a session are ordered by
        ``insertion_order ASC`` to match the in-memory backend's
        chronological contract.
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
        """Return the total number of records across all sessions.

        Returns 0 if the backend is closed — matches the crisis_log
        contract so CLI / API callers don't need defensive try/except.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        async with conn.execute("SELECT COUNT(*) FROM session_feedback") as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def apurge_before(self, cutoff: date) -> int:
        """Delete all feedback records with ``recorded_date < cutoff``.

        Single ``DELETE WHERE recorded_date < ?`` using the
        ``idx_feedback_recorded_date`` B-tree index for O(log n)
        lookups regardless of table size. Returns the rows deleted so
        the CLI / scheduled job can report outcome.

        Boundary is exclusive — records recorded on the cutoff date
        itself are preserved, matching crisis_log's purge contract.

        Closed backends return 0 without opening a connection.
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
        """Close the aiosqlite connection.

        Idempotent — safe to call on an already-closed backend. After
        closing, any subsequent method call raises ``RuntimeError``
        via ``_ensure_connection``.
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

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_date_prefix(recorded_at: str) -> str:
        """Extract the ``YYYY-MM-DD`` prefix from an ISO-8601 timestamp.

        ``SessionFeedbackRecord.recorded_at`` is always a string in
        ``YYYY-MM-DDTHH:MM:SSZ`` form (produced by ``iso_now()``).
        Splitting on ``T`` gives the date prefix without full datetime
        parsing. The prefix is validated via ``date.fromisoformat`` —
        malformed input raises ``ValueError``, letting caller bugs
        fail loudly at insert time rather than silently landing in a
        bad date bucket.
        """

        date_prefix = recorded_at.split("T", 1)[0]
        date.fromisoformat(date_prefix)
        return date_prefix


# ─── Protocol conformance assertion ────────────────────────────────────
#
# Verify at import time that SqliteSessionFeedbackBackend satisfies the
# SessionFeedbackBackend protocol. Matches the sqlite_crisis_log.py
# pattern — a type-level assertion that catches missing or mis-signed
# methods during module load.
_: type[SessionFeedbackBackend] = SqliteSessionFeedbackBackend


__all__ = [
    "SqliteSessionFeedbackBackend",
    "SESSION_FEEDBACK_SCHEMA_DDL",
    "SESSION_FEEDBACK_DDL",
    "SESSION_FEEDBACK_INDEX_SESSION_DDL",
    "SESSION_FEEDBACK_INDEX_DATE_DDL",
]
