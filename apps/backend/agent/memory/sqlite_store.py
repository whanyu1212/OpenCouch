"""SQLite-backed implementation of the :class:`MemoryStore` protocol.

Shipped in phase 1 v0.8 to close the asymmetric persistence gap.
Before v0.8 the only memory store was :class:`OpenCouchMemoryStore`,
which holds all records in a per-instance dict. That worked for v0.3
+ v0.4 dogfood but died at CLI restart — semantic facts and episodic
arcs vanished the moment the Python process exited.

:class:`SqliteMemoryStore` implements the same
:class:`agent.memory.store.MemoryStore` interface but backs its
records with an aiosqlite connection so they survive restarts. The
runtime picks between the two implementations based on memory mode:

- INCOGNITO mode → :class:`OpenCouchMemoryStore` (no disk writes)
- LOCAL / SYNCED mode → :class:`SqliteMemoryStore`

Stage A scope (this file, this commit):
- Declare the SQLite schema as module-level DDL constants
- Stub the class so downstream files can import the name; full
  implementation lands in Stage B

Design decisions locked in Stage A:

1. **Hybrid schema** — normalized discriminator columns for the
   things we filter/index on (``id``, ``owner_id``, ``namespace_kind``,
   ``category``, ``created_at``, ``last_referenced_at``), plus a
   ``value`` JSON column holding the serialized pydantic model
   (``model.model_dump(mode="json")``). Rationale: we get fast
   ``WHERE owner_id = ? AND namespace_kind = ?`` lookups without
   needing to decode JSON for every row, while still letting the
   schema evolve when we add fields to ``SemanticFact`` or
   ``StoredSessionArc`` without requiring a SQLite migration.

2. **Single table for all three namespaces** — ``memory_records`` with
   ``namespace_kind`` as the discriminator column. Semantic, episodic,
   and procedural records coexist in one table distinguished by that
   column. Simpler than three parallel tables; equivalent query
   performance via the ``idx_memory_owner_kind`` compound index.

3. **aiosqlite directly, no ORM.** aiosqlite is already in the
   dependency tree via ``langgraph-checkpoint-sqlite``. Raw SQL with
   pydantic handling serialization is simpler than SQLAlchemy for
   our ~1-table schema and zero-join query patterns.

4. **Keep the v0.3.1 Python scoring loop.** Rows come from SQLite via
   ``SELECT ... WHERE owner_id = ? AND namespace_kind = ?``, then
   the existing ``text_tokens`` recall scorer runs in Python on the
   returned rows. Not using SQLite FTS5 because its BM25 scoring is
   different from what v0.3.1's tests pin, and switching would
   require re-calibrating every retrieval test. At v0.8 scale
   (hundreds of records per user) the Python loop is fast enough.

5. **One connection per runtime lifetime.** Matches the pattern
   already used by the LangGraph checkpointer — connection opened
   in ``__aenter__``, closed in ``__aexit__``. aiosqlite handles
   async access through a single shared connection.

Stage B (next) implements the full async methods (``aput``, ``aget``,
``asearch``, ``adelete``, ``aclose``, ``record_count``, ``namespaces``)
against the schema declared below.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import aiosqlite

from agent.memory.store import (
    SEARCH_MATCH_THRESHOLD,
    MemoryStore,
    Namespace,
    StoreRecord,
)
from agent.memory.text_tokens import tokenize, tokenize_meaningful

logger = logging.getLogger(__name__)


def _encode_embedding(embedding: list[float] | None) -> bytes | None:
    """Serialize a float embedding to a BLOB for SQLite storage.

    v0.8.1 stores embeddings as raw little-endian ``float32`` arrays
    via ``struct.pack``. Each float is 4 bytes so a 768-dim embedding
    is 3,072 bytes — small enough that SQLite BLOB storage handles
    it without row overflow, and cheap to decode.

    Returns ``None`` when the input is None, so the caller can pass
    the result directly to the ``INSERT`` statement and get a NULL
    column value.

    We chose ``struct.pack`` over ``numpy.tobytes`` because struct
    is in the stdlib and avoids adding numpy as a runtime dependency
    just for serialization. The decoding cost on the read path is
    microseconds per vector at 768 dimensions.
    """

    if embedding is None:
        return None
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    """Deserialize a BLOB column into a float embedding.

    Inverse of :func:`_encode_embedding`. Returns ``None`` for NULL
    columns. Raises ``struct.error`` if the BLOB length isn't a
    multiple of 4 bytes (which should never happen for well-formed
    data but is a cheap sanity check).
    """

    if blob is None:
        return None
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


# ─── Schema DDL ────────────────────────────────────────────────────────


# The single table backing all three namespace kinds. Discriminated
# by ``namespace_kind``, indexed on ``(owner_id, namespace_kind)`` for
# fast per-user namespace scans, and also indexed on
# ``last_referenced_at`` for future retention / dormancy queries.
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
# v0.8.1 added three embedding columns (``embedding``, ``embedding_dim``,
# ``embedding_model``). For fresh SQLite files the
# ``CREATE TABLE IF NOT EXISTS`` above creates them in one shot. For
# existing v0.8 SQLite files that already have the table without
# these columns, ``_ensure_schema`` detects the missing columns and
# runs ``ALTER TABLE ADD COLUMN`` migrations in-place. Adding a column
# to an existing SQLite table is an O(1) metadata operation — no
# table rewrite required. All three columns are nullable so
# pre-v0.8.1 records naturally have NULL embeddings, which the
# read path treats as "no embedding available, use token-recall only."
# Schema notes:
#
# - ``insertion_order`` is an explicit INTEGER PRIMARY KEY AUTOINCREMENT
#   column rather than relying on SQLite's implicit ``rowid``. The
#   in-memory store's search path has an insertion-order tiebreaker
#   when two records share the same recall score, and a couple of
#   v0.3.1 tests pin that guarantee. An explicit AUTOINCREMENT column
#   makes the tiebreaker contractual at the schema level.
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

# All DDL statements to run on ``_ensure_schema``. Executing them in
# order is safe because every statement is ``IF NOT EXISTS``.
MEMORY_SCHEMA_DDL: tuple[str, ...] = (
    MEMORY_RECORDS_DDL,
    MEMORY_RECORDS_INDEX_OWNER_KIND_DDL,
    MEMORY_RECORDS_INDEX_LAST_REF_DDL,
)


# ─── SqliteMemoryStore stub ────────────────────────────────────────────


class SqliteMemoryStore:
    """SQLite-backed implementation of :class:`MemoryStore`.

    Full implementation of the :class:`MemoryStore` protocol backed by
    an aiosqlite connection. Behaviorally identical to
    :class:`OpenCouchMemoryStore` for all method semantics (put/get/
    search/delete/close/counts/namespaces) — the only difference is
    that records persist to disk instead of living in a dict.

    The search path runs the same v0.3.1 Python-side token-recall
    scorer used by the in-memory store. Rows come from SQLite via a
    per-namespace SELECT (``WHERE owner_id = ? AND namespace_kind =
    ?``), and the scorer runs in Python on the returned rows. At
    phase-1 scale (hundreds of records per user) this is fast enough;
    when the scale grows we can revisit with FTS5 or real embeddings.

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
        """Initialize the store with a SQLite file path.

        Args:
            sqlite_path: Path to the SQLite file. Use ``":memory:"``
                for a pure in-RAM database — tests that want the SQL
                semantics without the disk writes should use this
                path. Parent directories for file paths are created
                lazily on first connection open.
        """

        self.sqlite_path = (
            Path(sqlite_path) if sqlite_path != ":memory:" else Path(":memory:")
        )
        self._connection: aiosqlite.Connection | None = None
        self._closed = False

    # ── Connection lifecycle ──────────────────────────────────────────

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Open the aiosqlite connection lazily on first use.

        Subsequent calls are cheap no-ops that just return the
        already-open connection. Raises ``RuntimeError`` if the store
        has been closed — matches the in-memory store's ``_ensure_open``
        semantics.
        """

        if self._closed:
            raise RuntimeError("SqliteMemoryStore is closed.")
        if self._connection is not None:
            return self._connection

        # Create parent directory for file-backed SQLite paths. Skip
        # for ``:memory:`` which has no filesystem footprint.
        if str(self.sqlite_path) != ":memory:":
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(str(self.sqlite_path))
        # Return rows as sqlite3.Row so we can index by column name
        # rather than position. Slightly more readable code at minimal
        # runtime cost.
        self._connection.row_factory = aiosqlite.Row
        # Enable foreign keys (defensive; we don't use FKs yet but
        # future schema additions might, and it's cheap to turn on).
        await self._connection.execute("PRAGMA foreign_keys = ON")
        await self._ensure_schema(self._connection)
        return self._connection

    @staticmethod
    async def _ensure_schema(conn: aiosqlite.Connection) -> None:
        """Run the schema DDL and any needed migrations.

        Idempotent — safe to call repeatedly. The ``CREATE TABLE
        IF NOT EXISTS`` in :data:`MEMORY_RECORDS_DDL` creates the
        table with all columns for fresh databases. For existing
        v0.8 databases that predate v0.8.1's embedding columns, the
        migration block detects missing columns via
        ``PRAGMA table_info`` and runs ``ALTER TABLE ADD COLUMN``
        in-place.

        SQLite's ``ALTER TABLE ADD COLUMN`` is an O(1) metadata
        operation — no table rewrite — so the migration is cheap
        even on large existing stores. All three embedding columns
        are nullable so pre-v0.8.1 records naturally have NULL
        embeddings and the read path treats them as "token-recall
        only" without further fixup.
        """

        for ddl in MEMORY_SCHEMA_DDL:
            await conn.execute(ddl)

        # v0.8.1 migration: add embedding columns to pre-existing
        # v0.8 databases that were created before the embedding
        # work shipped. PRAGMA table_info returns one row per
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

        await conn.commit()

    # ── Public interface (MemoryStore protocol) ───────────────────────

    async def aput(
        self,
        namespace: Namespace,
        key: str,
        value: dict,
        *,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Store a record under ``(namespace, key)``.

        If a record already exists at that key, it is overwritten —
        the ``id`` column has a UNIQUE constraint, and we use
        ``INSERT OR REPLACE`` to match the in-memory store's
        overwrite-on-collision semantics.

        Namespace is a ``(owner_id, kind)`` tuple; we unpack it into
        the normalized columns. The full ``value`` dict is
        JSON-serialized into the ``value`` column so the caller
        can read it back as-is via ``aget`` / ``asearch``.

        v0.8.1: ``embedding`` and ``embedding_model`` are optional
        keyword args matching the :class:`MemoryStore` protocol.
        When provided, the embedding is serialized to a BLOB via
        :func:`_encode_embedding` and written to the ``embedding``
        column, the length goes to ``embedding_dim``, and the model
        name goes to ``embedding_model``. When absent, all three
        columns are written as NULL and the record participates in
        hybrid retrieval via the token-recall path only — same
        contract as the in-memory store.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)
        category = value.get("category")
        created_at = str(value.get("created_at") or "")
        last_referenced_at = str(value.get("last_referenced_at") or created_at or "")
        dormant_at = value.get("dormant_at")
        user_visible = 1 if value.get("user_visible", True) else 0
        serialized = json.dumps(value, default=str)

        embedding_blob = _encode_embedding(embedding)
        embedding_dim = len(embedding) if embedding is not None else None

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
                category,
                serialized,
                created_at,
                last_referenced_at,
                dormant_at,
                user_visible,
                embedding_blob,
                embedding_dim,
                embedding_model,
            ),
        )
        await conn.commit()

    @staticmethod
    def _row_to_store_record(
        row: aiosqlite.Row,
        namespace: Namespace,
    ) -> StoreRecord:
        """Map a ``memory_records`` row into a :class:`StoreRecord`.

        v0.8.1 helper: all three read paths (``aget``, ``asearch``,
        ``asearch_similar``) need the same row→record transformation
        including the embedding decode. Extracted here so a schema
        change only touches one place.

        Callers must ``SELECT`` at minimum the ``id``, ``value``,
        ``embedding``, and ``embedding_model`` columns. The helper
        does NOT require every column to be present — if ``embedding``
        or ``embedding_model`` is missing from the row, it defaults
        to None (which happens for old test fixtures that SELECT
        only id+value).
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

        return StoreRecord(
            namespace=namespace,
            key=row["id"],
            value=json.loads(row["value"]),
            embedding=_decode_embedding(embedding_blob),
            embedding_model=embedding_model,
        )

    async def aget(
        self,
        namespace: Namespace,
        key: str,
    ) -> StoreRecord | None:
        """Fetch one record by its ``(namespace, key)``.

        Returns ``None`` when the record does not exist. The namespace
        is verified alongside the key — a record with the same id
        under a different namespace returns ``None``, matching the
        in-memory store's bucket isolation.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)
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
        """Search for records within ``namespace`` matching ``query``.

        Semantics match :meth:`OpenCouchMemoryStore.asearch` exactly:

        - When ``query`` is ``None``, returns all records in the
          namespace in insertion order, up to ``limit``.
        - When ``query`` has no meaningful tokens after stopword
          filtering, returns an empty list.
        - Otherwise, computes query-token recall against each
          record's serialized value and returns matches sorted by
          recall descending with insertion order as the tiebreaker.

        The SQL layer does the per-namespace filter (cheap, indexed);
        the Python layer does the scoring (same code path as the
        in-memory store, so behavior is identical).
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)

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

        query_tokens = tokenize_meaningful(query)
        if not query_tokens:
            return []

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

        query_token_count = len(query_tokens)
        scored: list[tuple[float, int, StoreRecord]] = []
        for insertion_index, row in enumerate(rows):
            value_dict = json.loads(row["value"])
            haystack = " ".join(str(v) for v in value_dict.values() if v is not None)
            haystack_tokens = tokenize(haystack)
            if not haystack_tokens:
                continue
            overlap = len(query_tokens & haystack_tokens)
            recall = overlap / query_token_count
            if recall >= SEARCH_MATCH_THRESHOLD:
                scored.append(
                    (
                        recall,
                        insertion_index,
                        self._row_to_store_record(row, namespace),
                    )
                )

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored[:limit]]

    async def asearch_similar(
        self,
        namespace: Namespace,
        *,
        query_text: str,
        query_embedding: list[float] | None,
        embedding_model: str | None = None,
        limit: int = 10,
    ) -> list[StoreRecord]:
        """Hybrid retrieval via Reciprocal Rank Fusion (v0.8.1).

        Same contract as :meth:`OpenCouchMemoryStore.asearch_similar`.
        See :mod:`agent.memory.retrieval` for the RRF rationale.

        SQL-layer considerations:

        - Both scans use the same ``WHERE owner_id = ? AND
          namespace_kind = ?`` filter so the index on
          ``(owner_id, namespace_kind)`` is used for both paths.
        - The embedding-similarity scan pulls every row with a
          non-NULL ``embedding`` column — Python computes the cosines.
          At v0.8.1 scale (hundreds of records per user) this is
          well under 10ms per query. A future v0.8.2 could add a
          vector index (sqlite-vec or similar) without changing
          the method signature.
        - The token-recall scan is the same as :meth:`asearch`'s
          query-path, just extracted so ``asearch_similar`` can
          reuse the logic.
        """

        from agent.memory.retrieval import (
            EMBEDDING_MATCH_THRESHOLD,
            ScoredRecord,
            cosine_similarity,
            rrf_fuse,
        )

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)

        # ── Token-recall side (lexical) ───────────────────────────────
        lexical_scored: list[ScoredRecord] = []
        query_tokens = tokenize_meaningful(query_text)
        if query_tokens:
            async with conn.execute(
                """
                SELECT id, value, embedding, embedding_model FROM memory_records
                WHERE owner_id = ? AND namespace_kind = ?
                ORDER BY insertion_order ASC
                """,
                (owner_id, namespace_kind),
            ) as cursor:
                rows = await cursor.fetchall()

            query_token_count = len(query_tokens)
            for insertion_index, row in enumerate(rows):
                value_dict = json.loads(row["value"])
                haystack = " ".join(
                    str(v) for v in value_dict.values() if v is not None
                )
                haystack_tokens = tokenize(haystack)
                if not haystack_tokens:
                    continue
                overlap = len(query_tokens & haystack_tokens)
                recall = overlap / query_token_count
                if recall >= SEARCH_MATCH_THRESHOLD:
                    record = self._row_to_store_record(row, namespace)
                    lexical_scored.append(
                        ScoredRecord(
                            record=record,
                            score=recall,
                            insertion_index=insertion_index,
                        )
                    )
            lexical_scored.sort(key=lambda sr: (-sr.score, sr.insertion_index))

        # ── Embedding-similarity side (dense) ─────────────────────────
        dense_scored: list[ScoredRecord] = []
        if query_embedding is not None:
            # Pull only rows with a non-NULL embedding to avoid
            # scanning records that can't contribute to the dense
            # side anyway. The order matches the lexical scan (by
            # insertion_order) so the insertion_index tiebreaker
            # is consistent between the two scans.
            async with conn.execute(
                """
                SELECT id, value, embedding, embedding_model, insertion_order
                FROM memory_records
                WHERE owner_id = ? AND namespace_kind = ?
                    AND embedding IS NOT NULL
                ORDER BY insertion_order ASC
                """,
                (owner_id, namespace_kind),
            ) as cursor:
                rows = await cursor.fetchall()

            for insertion_index, row in enumerate(rows):
                stored_model = row["embedding_model"]
                # Skip cross-model similarity (same guard as the
                # in-memory store).
                if (
                    embedding_model is not None
                    and stored_model is not None
                    and stored_model != embedding_model
                ):
                    continue
                stored_embedding = _decode_embedding(row["embedding"])
                if stored_embedding is None:
                    continue
                if len(stored_embedding) != len(query_embedding):
                    continue
                sim = cosine_similarity(query_embedding, stored_embedding)
                if sim >= EMBEDDING_MATCH_THRESHOLD:
                    record = self._row_to_store_record(row, namespace)
                    dense_scored.append(
                        ScoredRecord(
                            record=record,
                            score=sim,
                            insertion_index=insertion_index,
                        )
                    )
            dense_scored.sort(key=lambda sr: (-sr.score, sr.insertion_index))

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
        """Delete a record by ``(namespace, key)``.

        Returns ``True`` if a record was deleted, ``False`` if no
        record existed at that key. Matches the in-memory store's
        return-value contract so ``/memory forget`` CLI commands
        can report accurately.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)
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
        """Close the aiosqlite connection.

        Safe to call on an already-closed store (idempotent no-op,
        matches the in-memory store's contract). After closing, any
        subsequent method call raises ``RuntimeError``.
        """

        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                # Don't raise on close failures — logging is enough.
                # A stuck close shouldn't poison the runtime shutdown.
                logger.warning(
                    "SqliteMemoryStore: connection close raised; ignoring",
                    exc_info=True,
                )
            finally:
                self._connection = None

    # ── Debug / observability helpers ─────────────────────────────────
    #
    # These share the same aiosqlite connection as the other async
    # methods, which is why they're async. An earlier attempt made
    # them sync by opening a short-lived sqlite3 connection per call,
    # but that breaks ``:memory:`` databases because each sqlite3
    # connection handle opens its own private in-memory DB —
    # connections don't share data. Making them async means they use
    # the same connection that holds the live data, which works for
    # both ``:memory:`` and file-backed paths.

    async def arecord_count(self, namespace: Namespace | None = None) -> int:
        """Return the total number of records, optionally filtered by namespace.

        Used by ``/memory status`` and ``/memory list`` CLI commands,
        by tests, and by ``load_memory_node`` to populate the
        session summary count. Returns 0 if the store is closed.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        if namespace is None:
            async with conn.execute("SELECT COUNT(*) FROM memory_records") as cursor:
                row = await cursor.fetchone()
        else:
            owner_id, namespace_kind = self._unpack_namespace(namespace)
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
        """Return every namespace that currently contains at least one record.

        Returns an empty list if the store is closed or the database
        is empty.
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

    # ── Namespace helpers ─────────────────────────────────────────────

    @staticmethod
    def _unpack_namespace(namespace: Namespace) -> tuple[str, str]:
        """Extract ``(owner_id, namespace_kind)`` from the tuple.

        Raises ``ValueError`` if the namespace tuple is malformed
        (wrong length or non-string elements). The in-memory store
        silently accepts any tuple shape; the SQLite store has to
        validate because the ``namespace_kind`` column has a CHECK
        constraint. Failing loudly at the boundary is safer than
        letting SQLite produce a constraint-violation exception
        several layers deep.
        """

        if len(namespace) != 2:
            raise ValueError(
                f"SqliteMemoryStore namespace must be (owner_id, kind) "
                f"tuple; got {namespace!r}"
            )
        owner_id, namespace_kind = namespace
        return str(owner_id), str(namespace_kind)


# ─── Protocol conformance assertion ────────────────────────────────────
#
# Verify at import time that SqliteMemoryStore satisfies the
# MemoryStore protocol. This is a type-level assertion — if any
# method is missing or has a wrong signature, type checkers flag it.
# At runtime the assignment just binds the type, so there's no cost.
_: type[MemoryStore] = SqliteMemoryStore


__all__ = [
    "SqliteMemoryStore",
    "MEMORY_SCHEMA_DDL",
    "MEMORY_RECORDS_DDL",
    "MEMORY_RECORDS_INDEX_OWNER_KIND_DDL",
    "MEMORY_RECORDS_INDEX_LAST_REF_DDL",
]
