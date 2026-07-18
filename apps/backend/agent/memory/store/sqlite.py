"""SQLite-backed implementation of the :class:`MemoryStore` protocol.

:class:`SqliteMemoryStore` provides the same async interface as
:class:`agent.memory.store.OpenCouchMemoryStore`, but persists records
through an aiosqlite connection so memory survives process restarts.
This legacy durable backend remains only for direct compatibility and removal
work. Application runtime selection is Postgres-only; use this class directly
only to inspect compatibility before its planned deletion.

All semantic, episodic, and procedural records live in one
``memory_records`` table and are separated by ``owner_id`` and
``namespace_kind``. The table keeps indexed columns for common filters
and stores the full serialized pydantic payload in a JSON ``value``
column.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from pathlib import Path
from typing import Any

import aiosqlite

from agent.memory.retrieval.ranking import (
    IndexedRecord,
    dense_candidate_limit,
    dense_rank,
    lexical_rank,
    rrf_fuse,
)
from agent.memory.store.base import (
    SEARCH_MATCH_THRESHOLD,
    MemoryRecordFilter,
    MemoryStore,
    Namespace,
    StoreRecord,
    build_store_record,
    memory_record_matches_filter,
    prepare_memory_record_fields,
    unpack_memory_namespace,
)

logger = logging.getLogger(__name__)


def _encode_embedding(embedding: list[float] | None) -> bytes | None:
    """Serialize an embedding vector for SQLite storage.

    Args:
        embedding (list[float] | None): Embedding vector to encode.

    Returns:
        bytes | None: Little-endian ``float32`` blob, or ``None``.
    """

    if embedding is None:
        return None
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    """Deserialize a SQLite embedding blob.

    Args:
        blob (bytes | None): Raw SQLite blob value.

    Returns:
        list[float] | None: Decoded embedding vector, or ``None``.
    """

    if blob is None:
        return None
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


# The single table backing all three namespace kinds. Discriminated
# by ``namespace_kind``, indexed on ``(owner_id, namespace_kind)`` for
# fast per-user namespace scans, and also indexed on
# ``last_referenced_at`` for retention and dormancy queries.
#
# The ``value`` column is a JSON string produced by
# ``pydantic_model.model_dump(mode="json")``. Callers receive a dict
# round-tripped through ``json.loads(row["value"])``, matching the
# ``StoreRecord.value`` shape the in-memory store exposes.
#
# ``CREATE TABLE IF NOT EXISTS`` is idempotent: the runtime calls
# ``_ensure_schema()`` on connection open, which runs this DDL once
# per process. New SQLite files get the schema on first access;
# existing files are left alone.
MEMORY_RECORDS_DDL = """
CREATE TABLE IF NOT EXISTS memory_records (
    insertion_order INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    namespace_kind TEXT NOT NULL
        CHECK (namespace_kind IN ('semantic', 'episodic', 'procedural')),
    category TEXT,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_referenced_at TEXT NOT NULL,
    dormant_at TEXT,
    user_visible INTEGER NOT NULL DEFAULT 1,
    embedding BLOB,
    embedding_dim INTEGER,
    embedding_model TEXT,
    UNIQUE (id, owner_id, namespace_kind)
);
"""
# Embedding columns are nullable so records without embeddings remain
# valid and use lexical retrieval only.
# Schema notes:
#
# - ``insertion_order`` is an explicit INTEGER PRIMARY KEY AUTOINCREMENT
#   column rather than relying on SQLite's implicit ``rowid``. The
#   in-memory store's search path has an insertion-order tiebreaker
#   when two records share the same recall score. An explicit
#   AUTOINCREMENT column makes the tiebreaker contractual at the schema
#   level.
#
# - ``UNIQUE (id, owner_id, namespace_kind)`` is a compound uniqueness
#   constraint, NOT just on ``id``. The same ``id`` string can appear
#   under two different namespaces (e.g., ``shared-key`` under both
#   ``("user-1", "semantic")`` and ``("user-1", "episodic")``). That
#   matches the in-memory store's bucket-isolation semantics. A
#   global UNIQUE on just ``id`` would collapse them, breaking the
#   ``test_memory_store_isolates_namespaces`` contract and any caller
#   that reuses key strings across namespaces.
#
# - The ``INSERT ON CONFLICT`` clause in aput targets this compound
#   uniqueness, so writing the same ``(id, owner_id, namespace_kind)``
#   twice updates in place, but writing the same ``id`` under a
#   different namespace inserts a new row.

MEMORY_RECORDS_INDEX_OWNER_KIND_DDL = """
CREATE INDEX IF NOT EXISTS idx_memory_owner_kind
    ON memory_records(owner_id, namespace_kind);
"""

MEMORY_RECORDS_INDEX_LAST_REF_DDL = """
CREATE INDEX IF NOT EXISTS idx_memory_last_ref
    ON memory_records(last_referenced_at);
"""

MEMORY_RECORDS_UNIQUE_KEY_COLUMNS = ("id", "owner_id", "namespace_kind")
MEMORY_RECORDS_LEGACY_GLOBAL_ID_COLUMNS = ("id",)
MEMORY_RECORDS_REBUILD_BACKUP_TABLE = "memory_records__legacy_unique_migration"
MEMORY_RECORDS_UNIQUE_KEY_INDEX_DDL = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_records_unique_key
    ON memory_records(id, owner_id, namespace_kind);
"""

# All DDL statements to run on ``_ensure_schema``. Executing them in
# order is safe because every statement is ``IF NOT EXISTS``.
MEMORY_SCHEMA_DDL: tuple[str, ...] = (
    MEMORY_RECORDS_DDL,
    MEMORY_RECORDS_INDEX_OWNER_KIND_DDL,
    MEMORY_RECORDS_INDEX_LAST_REF_DDL,
)


class SqliteMemoryStore:
    """SQLite-backed implementation of :class:`MemoryStore`.

    Full implementation of the :class:`MemoryStore` protocol backed by
    an aiosqlite connection. Behaviorally identical to
    :class:`OpenCouchMemoryStore` for all method semantics (put/get/
    search/delete/close/counts/namespaces) — the only difference is
    that records persist to disk instead of living in a dict.

    The search path loads rows with a per-namespace SELECT and then
    applies the shared lexical and dense retrieval helpers in Python.

    Connection lifecycle:
    - ``__init__`` stores the SQLite path but does NOT open the
      connection. Construction is cheap and doesn't touch disk.
    - The first async method call triggers lazy ``_ensure_connection``,
      which opens an aiosqlite connection and runs the schema DDL.
      Subsequent calls reuse the same connection.
    - ``aclose`` closes the connection and marks the store as closed.
      Any further calls raise ``RuntimeError``.
    - The store is **not** thread-safe. Each runtime instance should
      own its own store; do not share an instance across concurrent
      calls. Matches the in-memory version's contract.

    Why the connection is opened lazily rather than in ``__aenter__``:
    ``SqliteMemoryStore`` is not itself an async context manager — the
    ``PersistentAgentRuntime`` owns lifecycle. Construction happens
    eagerly in ``runtime.__init__``, but the connection can't open
    until ``runtime.__aenter__`` runs. Lazy opening means the store
    works whether the runtime wraps it in a context manager or not,
    and the test suite can construct instances in synchronous
    fixtures.

    Thread-safety note: aiosqlite internally uses a single worker
    thread per connection, so sequential awaits from one event loop
    are safe. Concurrent awaits from multiple event loops are not,
    but that's the same constraint the rest of the runtime has.
    """

    def __init__(self, sqlite_path: str | Path) -> None:
        """Initialize the SQLite-backed memory store.

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
        self._connect_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Open the SQLite connection on first use.

        Returns:
            aiosqlite.Connection: Shared connection for the store instance.
        """

        if self._closed:
            raise RuntimeError("SqliteMemoryStore is closed.")
        if self._connection is not None:
            return self._connection

        async with self._connect_lock:
            # Re-check after acquiring the lock — another task may have
            # initialized, or aclose() may have run while we waited.
            if self._closed:
                raise RuntimeError("SqliteMemoryStore is closed.")
            if self._connection is not None:
                return self._connection

            if str(self.sqlite_path) != ":memory:":
                self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize-then-publish: build into a local var and only
            # assign to self._connection after full init succeeds. If
            # PRAGMA setup or schema migration fails, the local is
            # closed and self._connection stays None so the next call
            # retries cleanly instead of reusing a broken handle.
            conn = await aiosqlite.connect(str(self.sqlite_path))
            try:
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA foreign_keys = ON")
                await self._ensure_schema(conn)
            except BaseException:
                await conn.close()
                raise
            self._connection = conn
            return self._connection

    @staticmethod
    def _quote_identifier(identifier: str) -> str:
        """Quote a SQLite identifier for PRAGMA statements."""

        return '"' + identifier.replace('"', '""') + '"'

    @staticmethod
    async def _has_unique_key(
        conn: aiosqlite.Connection,
        columns: tuple[str, ...],
    ) -> bool:
        """Return whether ``memory_records`` has a unique key on ``columns``."""

        async with conn.execute("PRAGMA index_list(memory_records)") as cursor:
            indexes = await cursor.fetchall()
        for row in indexes:
            index_name = str(row[1])
            is_unique = bool(row[2])
            if not is_unique:
                continue
            quoted_index_name = SqliteMemoryStore._quote_identifier(index_name)
            async with conn.execute(
                f"PRAGMA index_info({quoted_index_name})"
            ) as cursor:
                index_columns = tuple(
                    str(index_row[2]) for index_row in await cursor.fetchall()
                )
            if index_columns == columns:
                return True
        return False

    @staticmethod
    async def _has_compound_unique_key(conn: aiosqlite.Connection) -> bool:
        """Return whether ``memory_records`` has the current unique key."""

        return await SqliteMemoryStore._has_unique_key(
            conn,
            MEMORY_RECORDS_UNIQUE_KEY_COLUMNS,
        )

    @staticmethod
    async def _has_legacy_global_id_unique_key(conn: aiosqlite.Connection) -> bool:
        """Return whether an old global ``UNIQUE(id)`` key remains."""

        return await SqliteMemoryStore._has_unique_key(
            conn,
            MEMORY_RECORDS_LEGACY_GLOBAL_ID_COLUMNS,
        )

    @staticmethod
    async def _scalar_int(conn: aiosqlite.Connection, sql: str) -> int:
        """Execute a scalar integer SELECT."""

        async with conn.execute(sql) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    async def _deduplicate_compound_keys(conn: aiosqlite.Connection) -> int:
        """Collapse old duplicate rows before adding the compound unique key.

        Older SQLite memory DBs could contain duplicate
        ``(id, owner_id, namespace_kind)`` rows because the table lacked the
        conflict target used by ``aput``. Migration keeps the highest
        ``insertion_order`` row for each duplicate group, matching the observable
        "latest write wins" semantics that current ``aput`` provides, and
        preserves same-id rows in different owners or namespace kinds.
        """

        cursor = await conn.execute(
            """
            DELETE FROM memory_records
            WHERE insertion_order NOT IN (
                SELECT MAX(insertion_order)
                FROM memory_records
                GROUP BY id, owner_id, namespace_kind
            )
            """
        )
        try:
            return int(cursor.rowcount or 0)
        finally:
            await cursor.close()

    @staticmethod
    async def _rebuild_without_legacy_global_id_unique(
        conn: aiosqlite.Connection,
    ) -> int:
        """Rebuild old global-id-unique tables into the current table shape.

        SQLite cannot drop a table-level ``UNIQUE(id)`` constraint in place.
        Rebuilding removes the old autoindex, preserves rows, keeps the latest
        row per compound key, and recreates non-unique helper indexes for the
        new table.
        """

        backup_table = SqliteMemoryStore._quote_identifier(
            MEMORY_RECORDS_REBUILD_BACKUP_TABLE
        )
        await conn.execute(f"DROP TABLE IF EXISTS {backup_table}")
        total_before = await SqliteMemoryStore._scalar_int(
            conn,
            "SELECT COUNT(*) FROM memory_records",
        )
        await conn.execute(f"ALTER TABLE memory_records RENAME TO {backup_table}")
        await conn.execute(MEMORY_RECORDS_DDL)
        await conn.execute(
            f"""
            INSERT INTO memory_records
                (insertion_order, id, owner_id, namespace_kind, category, value,
                 created_at, last_referenced_at, dormant_at, user_visible,
                 embedding, embedding_dim, embedding_model)
            SELECT
                insertion_order, id, owner_id, namespace_kind, category, value,
                created_at, last_referenced_at, dormant_at, user_visible,
                embedding, embedding_dim, embedding_model
            FROM {backup_table}
            WHERE insertion_order IN (
                SELECT MAX(insertion_order)
                FROM {backup_table}
                GROUP BY id, owner_id, namespace_kind
            )
            ORDER BY insertion_order ASC
            """
        )
        total_after = await SqliteMemoryStore._scalar_int(
            conn,
            "SELECT COUNT(*) FROM memory_records",
        )
        await conn.execute(f"DROP TABLE {backup_table}")
        await conn.execute(MEMORY_RECORDS_INDEX_OWNER_KIND_DDL)
        await conn.execute(MEMORY_RECORDS_INDEX_LAST_REF_DDL)
        return max(0, total_before - total_after)

    @staticmethod
    async def _ensure_compound_unique_key(conn: aiosqlite.Connection) -> None:
        """Ensure old SQLite files have only the current ``aput`` conflict target."""

        if await SqliteMemoryStore._has_legacy_global_id_unique_key(conn):
            deleted = await SqliteMemoryStore._rebuild_without_legacy_global_id_unique(
                conn
            )
            if deleted:
                logger.warning(
                    "SqliteMemoryStore: removed %s duplicate memory_records rows "
                    "while rebuilding legacy global-id-unique table; kept latest "
                    "insertion_order per compound key",
                    deleted,
                )
            logger.info(
                "SqliteMemoryStore: rebuilt memory_records schema "
                "(removed legacy UNIQUE(id) constraint)"
            )
            return

        if await SqliteMemoryStore._has_compound_unique_key(conn):
            return

        deleted = await SqliteMemoryStore._deduplicate_compound_keys(conn)
        if deleted:
            logger.warning(
                "SqliteMemoryStore: removed %s duplicate memory_records rows while "
                "migrating compound unique key; kept latest insertion_order per key",
                deleted,
            )
        await conn.execute(MEMORY_RECORDS_UNIQUE_KEY_INDEX_DDL)
        logger.info(
            "SqliteMemoryStore: migrated memory_records schema "
            "(added compound unique key index)"
        )

    @staticmethod
    async def _ensure_schema(conn: aiosqlite.Connection) -> None:
        """Ensure the SQLite schema is present and migrated.

        Args:
            conn (aiosqlite.Connection): Open SQLite connection.

        Returns:
            None: Applies schema DDL and lightweight migrations.
        """

        await conn.execute("BEGIN")
        try:
            for ddl in MEMORY_SCHEMA_DDL:
                await conn.execute(ddl)

            # Add embedding columns to databases created before hybrid
            # retrieval shipped. PRAGMA table_info returns one row per
            # column; we collect column names and ALTER anything missing.
            async with conn.execute("PRAGMA table_info(memory_records)") as cursor:
                existing_columns = {row[1] for row in await cursor.fetchall()}
            migrations: list[tuple[str, str]] = [
                ("embedding", "ALTER TABLE memory_records ADD COLUMN embedding BLOB"),
                (
                    "embedding_dim",
                    "ALTER TABLE memory_records ADD COLUMN embedding_dim INTEGER",
                ),
                (
                    "embedding_model",
                    "ALTER TABLE memory_records ADD COLUMN embedding_model TEXT",
                ),
            ]
            for column_name, sql in migrations:
                if column_name not in existing_columns:
                    await conn.execute(sql)
                    logger.info(
                        "SqliteMemoryStore: migrated memory_records schema (added %s)",
                        column_name,
                    )

            await SqliteMemoryStore._ensure_compound_unique_key(conn)
        except BaseException:
            await conn.rollback()
            raise
        await conn.commit()

    async def aput(
        self,
        namespace: Namespace,
        key: str,
        value: dict,
        *,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Store or overwrite one SQLite-backed record.

        Args:
            namespace (Namespace): Record namespace tuple.
            key (str): Record key within the namespace.
            value (dict): Serialized record payload.
            embedding (list[float] | None): Optional precomputed embedding vector.
            embedding_model (str | None): Optional embedding model identifier.

        Returns:
            None: Writes the record to SQLite.
        """

        async with self._write_lock:
            conn = await self._ensure_connection()
            owner_id, namespace_kind = unpack_memory_namespace(namespace)
            fields = prepare_memory_record_fields(value, embedding=embedding)
            user_visible = 1 if fields.user_visible else 0
            embedding_blob = _encode_embedding(embedding)

            # INSERT OR REPLACE via a conflict clause on the compound
            # UNIQUE (id, owner_id, namespace_kind) constraint. When a row
            # already exists for this (id, owner_id, namespace_kind) tuple,
            # we update its value in place; otherwise we insert a new row.
            # The same ``id`` under a different namespace is a new row,
            # not a conflict — bucket-isolation semantics match the
            # in-memory store.
            await conn.execute(
                """
                INSERT INTO memory_records
                    (id, owner_id, namespace_kind, category, value,
                     created_at, last_referenced_at, dormant_at, user_visible,
                     embedding, embedding_dim, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id, owner_id, namespace_kind) DO UPDATE SET
                    category = excluded.category,
                    value = excluded.value,
                    created_at = excluded.created_at,
                    last_referenced_at = excluded.last_referenced_at,
                    dormant_at = excluded.dormant_at,
                    user_visible = excluded.user_visible,
                    embedding = excluded.embedding,
                    embedding_dim = excluded.embedding_dim,
                    embedding_model = excluded.embedding_model
                """,
                (
                    key,
                    owner_id,
                    namespace_kind,
                    fields.category,
                    fields.serialized_value,
                    fields.created_at,
                    fields.last_referenced_at,
                    fields.dormant_at,
                    user_visible,
                    embedding_blob,
                    fields.embedding_dim,
                    embedding_model,
                ),
            )
            await conn.commit()

    async def aput_batch(
        self,
        items: list[
            tuple[
                Namespace,
                str,
                dict[str, Any],
                list[float] | None,
                str | None,
            ]
        ],
    ) -> None:
        """Write multiple SQLite-backed records in one transaction.

        Args:
            items (list[tuple[Namespace, str, dict[str, Any], list[float] | None, str | None]]):
                Items shaped as ``(namespace, key, value, embedding, embedding_model)``.

        Returns:
            None: Commits the batch or rolls it back on failure.
        """

        if not items:
            return

        async with self._write_lock:
            conn = await self._ensure_connection()
            try:
                await conn.execute("BEGIN")
                for namespace, key, value, embedding, embedding_model in items:
                    owner_id, namespace_kind = unpack_memory_namespace(namespace)
                    fields = prepare_memory_record_fields(value, embedding=embedding)
                    user_visible = 1 if fields.user_visible else 0
                    embedding_blob = _encode_embedding(embedding)
                    await conn.execute(
                        """
                        INSERT INTO memory_records
                            (id, owner_id, namespace_kind, category, value,
                             created_at, last_referenced_at, dormant_at, user_visible,
                             embedding, embedding_dim, embedding_model)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id, owner_id, namespace_kind) DO UPDATE SET
                            category = excluded.category,
                            value = excluded.value,
                            created_at = excluded.created_at,
                            last_referenced_at = excluded.last_referenced_at,
                            dormant_at = excluded.dormant_at,
                            user_visible = excluded.user_visible,
                            embedding = excluded.embedding,
                            embedding_dim = excluded.embedding_dim,
                            embedding_model = excluded.embedding_model
                        """,
                        (
                            key,
                            owner_id,
                            namespace_kind,
                            fields.category,
                            fields.serialized_value,
                            fields.created_at,
                            fields.last_referenced_at,
                            fields.dormant_at,
                            user_visible,
                            embedding_blob,
                            fields.embedding_dim,
                            embedding_model,
                        ),
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    @staticmethod
    def _row_to_store_record(
        row: aiosqlite.Row,
        namespace: Namespace,
    ) -> StoreRecord:
        """Convert a SQLite row into a store record.

        Args:
            row (aiosqlite.Row): Row from ``memory_records``.
            namespace (Namespace): Namespace tuple for the row.

        Returns:
            StoreRecord: Converted store record.
        """

        # Row access via sqlite3.Row supports both name and index
        # lookups, but name lookups on missing columns raise
        # IndexError. Guard with a try/except so the helper works
        # on partial-column SELECTs.
        try:
            embedding_blob = row["embedding"]
        except IndexError:
            embedding_blob = None
        try:
            embedding_model = row["embedding_model"]
        except IndexError:
            embedding_model = None

        return build_store_record(
            namespace=namespace,
            key=row["id"],
            value=row["value"],
            embedding=_decode_embedding(embedding_blob),
            embedding_model=embedding_model,
        )

    async def aget(
        self,
        namespace: Namespace,
        key: str,
    ) -> StoreRecord | None:
        """Fetch one SQLite-backed record.

        Args:
            namespace (Namespace): Namespace tuple to search.
            key (str): Record key within the namespace.

        Returns:
            StoreRecord | None: Matching record, or ``None``.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = unpack_memory_namespace(namespace)
        async with conn.execute(
            """
            SELECT id, value, embedding, embedding_model FROM memory_records
            WHERE id = ? AND owner_id = ? AND namespace_kind = ?
            LIMIT 1
            """,
            (key, owner_id, namespace_kind),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_store_record(row, namespace)

    async def asearch(
        self,
        namespace: Namespace,
        *,
        query: str | None = None,
        limit: int = 10,
    ) -> list[StoreRecord]:
        """Search SQLite-backed records with lexical scoring.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query (str | None): Optional lexical query. ``None`` enumerates records.
            limit (int): Maximum number of records to return.

        Returns:
            list[StoreRecord]: Matching records in recall-score order.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = unpack_memory_namespace(namespace)

        if query is None:
            # Return all records in insertion order, limited.
            async with conn.execute(
                """
                SELECT id, value, embedding, embedding_model FROM memory_records
                WHERE owner_id = ? AND namespace_kind = ?
                ORDER BY insertion_order ASC
                LIMIT ?
                """,
                (owner_id, namespace_kind, limit),
            ) as cursor:
                rows = await cursor.fetchall()
            return [self._row_to_store_record(row, namespace) for row in rows]

        # Pull all rows for this namespace (ordered by insertion_order)
        # and run the recall scorer in Python. This matches the
        # in-memory store's loop exactly — see store.py asearch for
        # the scoring rationale.
        async with conn.execute(
            """
            SELECT id, value, embedding, embedding_model FROM memory_records
            WHERE owner_id = ? AND namespace_kind = ?
            ORDER BY insertion_order ASC
            """,
            (owner_id, namespace_kind),
        ) as cursor:
            rows = await cursor.fetchall()

        candidates = [
            IndexedRecord(
                record=self._row_to_store_record(row, namespace),
                insertion_index=insertion_index,
            )
            for insertion_index, row in enumerate(rows)
        ]
        scored = lexical_rank(
            candidates,
            query_text=query,
            match_threshold=SEARCH_MATCH_THRESHOLD,
        )
        return [scored_record.record for scored_record in scored[:limit]]

    async def asearch_similar(
        self,
        namespace: Namespace,
        *,
        query_text: str,
        query_embedding: list[float] | None,
        embedding_model: str | None = None,
        limit: int = 10,
        max_age_days: int | None = None,
        record_filter: MemoryRecordFilter | None = None,
    ) -> list[StoreRecord]:
        """Run hybrid retrieval over SQLite-backed records.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query_text (str): Query text for lexical scoring.
            query_embedding (list[float] | None): Optional dense query embedding.
            embedding_model (str | None): Optional query embedding model identifier.
            limit (int): Maximum number of records to return.
            max_age_days (int | None): Optional age filter in days.
            record_filter (MemoryRecordFilter | None): Optional declarative filter
                applied before ranking and truncation.

        Returns:
            list[StoreRecord]: Top fused retrieval results.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = unpack_memory_namespace(namespace)

        # Build the age filter clause. Use datetime() to normalize both
        # sides — stored timestamps may use "Z" suffix while datetime('now')
        # produces "+00:00" format.
        age_clause = ""
        age_params: tuple = ()
        if max_age_days is not None:
            age_clause = " AND datetime(created_at) >= datetime('now', ?)"
            age_params = (f"-{max_age_days} days",)

        async with conn.execute(
            f"""
            SELECT id, value, embedding, embedding_model FROM memory_records
            WHERE owner_id = ? AND namespace_kind = ?{age_clause}
            ORDER BY insertion_order ASC
            """,
            (owner_id, namespace_kind, *age_params),
        ) as cursor:
            lexical_rows = await cursor.fetchall()

        lexical_candidates = [
            IndexedRecord(
                record=self._row_to_store_record(row, namespace),
                insertion_index=insertion_index,
            )
            for insertion_index, row in enumerate(lexical_rows)
        ]
        if record_filter is not None:
            lexical_candidates = [
                candidate
                for candidate in lexical_candidates
                if memory_record_matches_filter(candidate.record, record_filter)
            ]
        lexical_scored = lexical_rank(
            lexical_candidates,
            query_text=query_text,
            match_threshold=SEARCH_MATCH_THRESHOLD,
        )

        dense_scored = []
        if query_embedding is not None:
            # Keep the dense-side row selection unchanged so this refactor
            # preserves the SQLite store's current insertion-index semantics.
            async with conn.execute(
                f"""
                SELECT id, value, embedding, embedding_model, insertion_order
                FROM memory_records
                WHERE owner_id = ? AND namespace_kind = ?
                    AND embedding IS NOT NULL{age_clause}
                ORDER BY insertion_order ASC
                """,
                (owner_id, namespace_kind, *age_params),
            ) as cursor:
                dense_rows = await cursor.fetchall()

            dense_candidates = [
                IndexedRecord(
                    record=self._row_to_store_record(row, namespace),
                    insertion_index=insertion_index,
                )
                for insertion_index, row in enumerate(dense_rows)
            ]
            if record_filter is not None:
                dense_candidates = [
                    candidate
                    for candidate in dense_candidates
                    if memory_record_matches_filter(candidate.record, record_filter)
                ]
            dense_scored = dense_rank(
                dense_candidates,
                query_embedding=query_embedding,
                embedding_model=embedding_model,
            )[: dense_candidate_limit(limit)]

        if not lexical_scored and not dense_scored:
            return []

        return rrf_fuse(
            lexical_ranked=lexical_scored,
            dense_ranked=dense_scored,
            limit=limit,
        )

    async def adelete(
        self,
        namespace: Namespace,
        key: str,
    ) -> bool:
        """Delete one SQLite-backed record.

        Args:
            namespace (Namespace): Namespace tuple containing the record.
            key (str): Record key within the namespace.

        Returns:
            bool: ``True`` when a record was deleted.
        """

        async with self._write_lock:
            conn = await self._ensure_connection()
            owner_id, namespace_kind = unpack_memory_namespace(namespace)
            cursor = await conn.execute(
                """
                DELETE FROM memory_records
                WHERE id = ? AND owner_id = ? AND namespace_kind = ?
                """,
                (key, owner_id, namespace_kind),
            )
            await conn.commit()
            return (cursor.rowcount or 0) > 0

    async def aclose(self) -> None:
        """Close the SQLite store connection.

        Returns:
            None: Marks the store closed and releases the connection.
        """

        if self._closed:
            return
        # Serialize with _write_lock first so close cannot race with a
        # commit/rollback on the shared connection. Then take _connect_lock in
        # the same order write methods use (_write_lock -> _ensure_connection ->
        # _connect_lock) to avoid deadlocks with lazy connection opening.
        async with self._write_lock:
            async with self._connect_lock:
                self._closed = True
                if self._connection is not None:
                    try:
                        await self._connection.close()
                    except Exception:
                        logger.warning(
                            "SqliteMemoryStore: connection close raised; ignoring",
                            exc_info=True,
                        )
                    finally:
                        self._connection = None

    # These share the same aiosqlite connection as the other async
    # methods, which is why they're async. An earlier attempt made
    # them sync by opening a short-lived sqlite3 connection per call,
    # but that breaks ``:memory:`` databases because each sqlite3
    # connection handle opens its own private in-memory DB —
    # connections don't share data. Making them async means they use
    # the same connection that holds the live data, which works for
    # both ``:memory:`` and file-backed paths.

    async def arecord_count(self, namespace: Namespace | None = None) -> int:
        """Count SQLite-backed records.

        Args:
            namespace (Namespace | None): Optional namespace filter.

        Returns:
            int: Total record count for the store or namespace.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        if namespace is None:
            async with conn.execute("SELECT COUNT(*) FROM memory_records") as cursor:
                row = await cursor.fetchone()
        else:
            owner_id, namespace_kind = unpack_memory_namespace(namespace)
            async with conn.execute(
                """
                SELECT COUNT(*) FROM memory_records
                WHERE owner_id = ? AND namespace_kind = ?
                """,
                (owner_id, namespace_kind),
            ) as cursor:
                row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def anamespaces(self) -> list[Namespace]:
        """List non-empty SQLite namespaces.

        Returns:
            list[Namespace]: Namespaces that currently contain records.
        """

        if self._closed:
            return []
        conn = await self._ensure_connection()
        async with conn.execute(
            """
            SELECT DISTINCT owner_id, namespace_kind
            FROM memory_records
            """
        ) as cursor:
            rows = await cursor.fetchall()
        return [(row["owner_id"], row["namespace_kind"]) for row in rows]

    async def alatest(self, namespace: Namespace) -> StoreRecord | None:
        """Fetch the latest SQLite-backed record in a namespace.

        Args:
            namespace (Namespace): Namespace tuple to inspect.

        Returns:
            StoreRecord | None: Most recent record, or ``None``.
        """

        if self._closed:
            return None
        conn = await self._ensure_connection()
        owner_id, namespace_kind = unpack_memory_namespace(namespace)
        async with conn.execute(
            """
            SELECT id, value, embedding, embedding_model FROM memory_records
            WHERE owner_id = ? AND namespace_kind = ?
            ORDER BY insertion_order DESC
            LIMIT 1
            """,
            (owner_id, namespace_kind),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_store_record(row, namespace)


# Static conformance check for the ``MemoryStore`` protocol.
_: type[MemoryStore] = SqliteMemoryStore


__all__ = [
    "SqliteMemoryStore",
    "MEMORY_SCHEMA_DDL",
    "MEMORY_RECORDS_DDL",
    "MEMORY_RECORDS_INDEX_OWNER_KIND_DDL",
    "MEMORY_RECORDS_INDEX_LAST_REF_DDL",
]
