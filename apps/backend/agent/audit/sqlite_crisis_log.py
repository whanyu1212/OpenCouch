"""SQLite-backed implementation of the :class:`CrisisLogBackend` protocol.

Shipped in phase 1 v0.8 to close the asymmetric persistence gap for
the crisis log, in parallel with the
:mod:`agent.memory.sqlite_store` work for the memory store. Before
v0.8 the only crisis log backend was
:class:`InMemoryCrisisLogBackend`, which held records in a
per-instance ``defaultdict(list)`` keyed by date. That worked for
single-process tests and incognito-mode CLI sessions but died at CLI
restart — crisis events vanished the moment the Python process exited.

:class:`SqliteCrisisLogBackend` implements the same
:class:`agent.audit.crisis_log.CrisisLogBackend` protocol but backs
its records with an aiosqlite connection so they survive restarts.
The runtime picks between the two implementations based on memory
mode:

- INCOGNITO mode → :class:`InMemoryCrisisLogBackend` is still used,
  paired with :class:`OpenCouchMemoryStore` for the memory store —
  nothing touches disk.
- LOCAL / SYNCED mode → :class:`SqliteCrisisLogBackend`, paired with
  :class:`agent.memory.sqlite_store.SqliteMemoryStore`.

Design decisions locked in the v0.8 scoping discussion and matched
here from :mod:`agent.memory.sqlite_store`:

1. **Hybrid schema** — discriminating columns (``id``, ``detected_at``,
   ``detected_date``, ``level``, ``session_id_opaque``) are
   normalized for indexed queries; the full serialized
   :class:`CrisisLogRecord` lives in a ``value`` JSON column for
   forward compatibility.

2. **Pre-computed ``detected_date`` column.** The primary query is
   ``alist_by_date(day)``, which in a naive schema would be
   ``WHERE date(detected_at) = ?``. SQLite can't use a B-tree index
   on a function call, so we pre-compute the ``YYYY-MM-DD`` prefix
   at insert time and index it. This makes date lookups O(log n)
   instead of O(n).

3. **Append-only from the agent's perspective.** Graph nodes never
   delete or update crisis records. v0.8.1 adds
   :meth:`apurge_before` as a retention operation — it's on the
   protocol so the runtime can call it polymorphically, but it's
   invoked by the CLI (``/memory purge-crisis``) or by a future
   scheduled cleanup job, not by any agent node. The SQLite path
   runs a single ``DELETE WHERE detected_date < ?`` which uses
   the existing ``idx_crisis_detected_date`` B-tree index.

4. **aiosqlite directly, no ORM.** Same rationale as the memory
   store: aiosqlite is already in the dep tree via
   ``langgraph-checkpoint-sqlite``, and the crisis log's schema is
   even simpler than the memory store's (one table, no search
   scoring) so SQLAlchemy would be pure overhead.

5. **One connection per runtime lifetime, lazily opened.** Matches
   the memory store's pattern. The connection opens on first async
   method call and closes via ``aclose``.

6. **Single-column UNIQUE on ``id``.** Unlike the memory store,
   crisis log records have globally unique UUID ids (no namespace
   collisions), so the UNIQUE constraint is single-column. The
   in-memory backend doesn't enforce this (it just appends), but
   the schema-level constraint here is a defensive guard against
   duplicate writes, which would indicate a caller bug per the
   protocol docstring.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import aiosqlite

from agent.audit.crisis_log import CrisisLogBackend
from agent.memory.models import CrisisLogRecord

logger = logging.getLogger(__name__)


# ─── Schema DDL ────────────────────────────────────────────────────────


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
# Schema notes:
#
# - ``insertion_order INTEGER PRIMARY KEY AUTOINCREMENT`` gives us a
#   stable chronological order for ``ORDER BY insertion_order ASC``
#   within a date bucket, matching the in-memory backend's "records
#   in insertion order" contract for ``alist_by_date``.
#
# - ``id TEXT NOT NULL UNIQUE`` — single-column UNIQUE because crisis
#   log record ids are globally unique UUIDs (unlike memory store
#   records which can share an id string across namespaces).
#
# - ``detected_date TEXT NOT NULL`` holds the ``YYYY-MM-DD`` prefix of
#   ``detected_at``, computed at insert time. This lets SQLite use a
#   B-tree index for ``alist_by_date`` queries instead of evaluating
#   ``date(detected_at)`` for every row.
#
# - ``level INTEGER ... CHECK (level IN (0, 1, 2, 3))`` mirrors the
#   memory store's CHECK constraint on ``namespace_kind`` — schema-
#   level validation of controlled vocabularies catches bad writes
#   at the SQL layer before they corrupt the audit trail.
#
# - ``value TEXT NOT NULL`` holds the full serialized
#   :class:`CrisisLogRecord` as JSON. The discriminating columns are
#   denormalized from this value for indexed queries; the JSON is
#   the source of truth for the full record shape.


CRISIS_LOG_INDEX_DATE_DDL = """
CREATE INDEX IF NOT EXISTS idx_crisis_detected_date
    ON crisis_log(detected_date);
"""

CRISIS_LOG_INDEX_SESSION_DDL = """
CREATE INDEX IF NOT EXISTS idx_crisis_session
    ON crisis_log(session_id_opaque);
"""


# All DDL statements run on ``_ensure_schema``, in order. Every
# statement uses ``IF NOT EXISTS`` so re-running is a safe no-op.
CRISIS_LOG_SCHEMA_DDL: tuple[str, ...] = (
    CRISIS_LOG_DDL,
    CRISIS_LOG_INDEX_DATE_DDL,
    CRISIS_LOG_INDEX_SESSION_DDL,
)


# ─── SqliteCrisisLogBackend ────────────────────────────────────────────


class SqliteCrisisLogBackend:
    """SQLite-backed implementation of :class:`CrisisLogBackend`.

    Behaviorally identical to :class:`InMemoryCrisisLogBackend` for
    all three method semantics (aappend / alist_by_date /
    arecord_count / aclose) — the only difference is that records
    persist to disk instead of living in a dict.

    Connection lifecycle:
    - ``__init__`` stores the SQLite path but does NOT open the
      connection. Construction is cheap and doesn't touch disk.
    - The first async method call triggers lazy ``_ensure_connection``,
      which opens an aiosqlite connection and runs the schema DDL.
      Subsequent calls reuse the same connection.
    - ``aclose`` closes the connection and marks the backend as closed.
      Any further calls raise ``RuntimeError``.
    - The backend is **not** thread-safe. Each runtime instance should
      own its own backend; do not share an instance across concurrent
      calls. Matches the in-memory version's contract.

    Why the connection is opened lazily rather than in a context
    manager: ``SqliteCrisisLogBackend`` is constructed eagerly in
    ``PersistentAgentRuntime.__init__`` but can't open its connection
    until ``runtime.__aenter__`` runs. Lazy opening means the backend
    works whether the runtime wraps it in a context manager or not,
    and test fixtures can construct instances synchronously.
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

    # ── Connection lifecycle ──────────────────────────────────────────

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

    # ── Public interface (CrisisLogBackend protocol) ──────────────────

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

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_date_prefix(detected_at: str) -> str:
        """Extract the date prefix from an ISO-8601 timestamp.

        Args:
            detected_at (str): ISO-8601 crisis detection timestamp.

        Returns:
            str: ``YYYY-MM-DD`` date prefix.
        """

        date_prefix = detected_at.split("T", 1)[0]
        # Validate via date.fromisoformat — raises ValueError on
        # malformed input, which we let propagate to the aappend
        # caller as the correct "caller passed bad data" signal.
        date.fromisoformat(date_prefix)
        return date_prefix


# ─── Protocol conformance assertion ────────────────────────────────────
#
# Verify at import time that SqliteCrisisLogBackend satisfies the
# CrisisLogBackend protocol. Same pattern as SqliteMemoryStore — a
# type-level assertion that catches missing or mis-signed methods
# during module load.
_: type[CrisisLogBackend] = SqliteCrisisLogBackend


__all__ = [
    "SqliteCrisisLogBackend",
    "CRISIS_LOG_SCHEMA_DDL",
    "CRISIS_LOG_DDL",
    "CRISIS_LOG_INDEX_DATE_DDL",
    "CRISIS_LOG_INDEX_SESSION_DDL",
]
