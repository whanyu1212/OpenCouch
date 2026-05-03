"""PostgreSQL-backed implementation of the :class:`MemoryStore` protocol.

:class:`PostgresMemoryStore` mirrors the behavioral contract of
:class:`agent.memory.legacy.sqlite_store.SqliteMemoryStore`, but persists
records through a lazily opened psycopg async connection.

This first-cut implementation intentionally keeps retrieval semantics in
Python by reusing the shared lexical/dense ranking helpers. That keeps
result ordering aligned with the in-memory and SQLite stores while the
project migrates persistence off SQLite.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, cast

import psycopg
from psycopg.rows import dict_row

from agent.memory.embeddings import DEFAULT_OPENAI_EMBEDDING_DIMENSION
from agent.memory.retrieval import (
    EMBEDDING_MATCH_THRESHOLD,
    IndexedRecord,
    ScoredRecord,
    lexical_rank,
    rrf_fuse,
)
from agent.memory.store import (
    SEARCH_MATCH_THRESHOLD,
    MemoryStore,
    Namespace,
    StoreRecord,
)

logger = logging.getLogger(__name__)

DENSE_CANDIDATE_MULTIPLIER = 5
MIN_DENSE_CANDIDATES = 50
PGVECTOR_FIXED_DIMENSION = DEFAULT_OPENAI_EMBEDDING_DIMENSION

MEMORY_RECORDS_DDL = """
CREATE TABLE IF NOT EXISTS memory_records (
    insertion_order BIGSERIAL PRIMARY KEY,
    id TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    namespace_kind TEXT NOT NULL
        CHECK (namespace_kind IN ('semantic', 'episodic', 'procedural')),
    category TEXT,
    value JSONB NOT NULL,
    created_at TEXT NOT NULL,
    last_referenced_at TEXT NOT NULL,
    dormant_at TEXT,
    user_visible BOOLEAN NOT NULL DEFAULT TRUE,
    embedding DOUBLE PRECISION[],
    embedding_vector_3072 vector(3072),
    embedding_dim INTEGER,
    embedding_model TEXT,
    UNIQUE (id, owner_id, namespace_kind)
);
"""

MEMORY_RECORDS_INDEX_OWNER_KIND_DDL = """
CREATE INDEX IF NOT EXISTS idx_memory_owner_kind
    ON memory_records(owner_id, namespace_kind);
"""

MEMORY_RECORDS_INDEX_LAST_REF_DDL = """
CREATE INDEX IF NOT EXISTS idx_memory_last_ref
    ON memory_records(last_referenced_at);
"""

MEMORY_RECORDS_INDEX_EMBEDDING_VECTOR_3072_HNSW_DDL = """
CREATE INDEX IF NOT EXISTS idx_memory_embedding_vector_3072_hnsw
    ON memory_records
    USING hnsw (embedding_vector_3072 vector_cosine_ops)
    WHERE embedding_vector_3072 IS NOT NULL;
"""

MEMORY_SCHEMA_DDL: tuple[str, ...] = (
    MEMORY_RECORDS_DDL,
    MEMORY_RECORDS_INDEX_OWNER_KIND_DDL,
    MEMORY_RECORDS_INDEX_LAST_REF_DDL,
    MEMORY_RECORDS_INDEX_EMBEDDING_VECTOR_3072_HNSW_DDL,
)


class PostgresMemoryStore:
    """PostgreSQL-backed implementation of :class:`MemoryStore`.

    Construction is cheap; the database connection opens on first use.
    Retrieval semantics intentionally match the SQLite store by loading
    namespace rows and applying the shared ranking helpers in Python.
    """

    def __init__(self, dsn: str) -> None:
        """Initialize the PostgreSQL-backed memory store.

        Args:
            dsn (str): PostgreSQL connection string.

        Returns:
            None: Stores connection configuration for lazy initialization.
        """

        self.dsn = dsn
        self._connection: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self._closed = False
        self._connect_lock = asyncio.Lock()

    async def _ensure_connection(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        """Open the PostgreSQL connection on first use.

        Returns:
            psycopg.AsyncConnection[dict[str, Any]]: Shared connection for the
                store instance.
        """

        if self._closed:
            raise RuntimeError("PostgresMemoryStore is closed.")
        if self._connection is not None:
            return self._connection

        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("PostgresMemoryStore is closed.")
            if self._connection is not None:
                return self._connection

            conn = await psycopg.AsyncConnection.connect(
                self.dsn,
                row_factory=dict_row,
                autocommit=True,
            )
            try:
                await self._ensure_schema(conn)
            except BaseException:
                await conn.close()
                raise
            self._connection = conn
            return self._connection

    @staticmethod
    async def _ensure_schema(
        conn: psycopg.AsyncConnection[dict[str, Any]],
    ) -> None:
        """Ensure the PostgreSQL schema is present.

        Args:
            conn (psycopg.AsyncConnection[dict[str, Any]]): Open PostgreSQL
                connection.

        Returns:
            None: Applies schema DDL and lightweight migrations.
        """

        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                for ddl in MEMORY_SCHEMA_DDL:
                    await cursor.execute(ddl)
                await cursor.execute(
                    """
                    ALTER TABLE memory_records
                    ADD COLUMN IF NOT EXISTS embedding_vector_3072 vector(3072)
                    """
                )
            await PostgresMemoryStore._backfill_embedding_vector_3072(conn)

    @staticmethod
    def _embedding_to_vector_literal(embedding: list[float] | None) -> str | None:
        """Serialize an embedding for pgvector SQL casts.

        Args:
            embedding (list[float] | None): Embedding vector to serialize.

        Returns:
            str | None: pgvector literal like ``[1.0,2.0]``, or ``None``.
        """

        if embedding is None:
            return None
        return "[" + ",".join(str(value) for value in embedding) + "]"

    @staticmethod
    def _embedding_vector_3072_literal(
        embedding: list[float] | None,
        embedding_dim: int | None,
    ) -> str | None:
        """Return a fixed-dimension pgvector literal for supported embeddings.

        Args:
            embedding (list[float] | None): Embedding vector to serialize.
            embedding_dim (int | None): Recorded embedding dimension.

        Returns:
            str | None: Vector literal when the embedding matches the canonical
                3072-dimension cohort, otherwise ``None``.
        """

        if embedding is None or embedding_dim != PGVECTOR_FIXED_DIMENSION:
            return None
        return PostgresMemoryStore._embedding_to_vector_literal(embedding)

    @staticmethod
    async def _backfill_embedding_vector_3072(
        conn: psycopg.AsyncConnection[dict[str, Any]],
    ) -> None:
        """Populate the fixed-dimension pgvector column for existing rows.

        Args:
            conn (psycopg.AsyncConnection[dict[str, Any]]): Open PostgreSQL
                connection.

        Returns:
            None: Backfills eligible rows in-place.
        """

        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT insertion_order, embedding
                FROM memory_records
                WHERE embedding IS NOT NULL
                    AND embedding_dim = %s
                    AND embedding_vector_3072 IS NULL
                """,
                (PGVECTOR_FIXED_DIMENSION,),
            )
            rows = await cursor.fetchall()

        if not rows:
            return

        async with conn.transaction():
            async with conn.cursor() as cursor:
                for row in rows:
                    embedding = row.get("embedding")
                    if embedding is None:
                        continue
                    embedding_vector_3072 = (
                        PostgresMemoryStore._embedding_vector_3072_literal(
                            list(embedding),
                            PGVECTOR_FIXED_DIMENSION,
                        )
                    )
                    if embedding_vector_3072 is None:
                        continue
                    await cursor.execute(
                        """
                        UPDATE memory_records
                        SET embedding_vector_3072 = %s::vector(3072)
                        WHERE insertion_order = %s
                        """,
                        (
                            embedding_vector_3072,
                            row["insertion_order"],
                        ),
                    )

    @staticmethod
    def _dense_candidate_limit(limit: int) -> int:
        """Return the bounded dense candidate count for SQL retrieval.

        Args:
            limit (int): Final requested result count.

        Returns:
            int: Candidate pool size for dense SQL pruning.
        """

        return max(limit * DENSE_CANDIDATE_MULTIPLIER, MIN_DENSE_CANDIDATES)

    @staticmethod
    def _sql_dense_row_to_scored_record(
        row: dict[str, Any],
        *,
        namespace: Namespace,
        insertion_index: int,
    ) -> ScoredRecord | None:
        """Convert a SQL dense-search row into a scored record.

        Args:
            row (dict[str, Any]): Row returned from the dense pgvector query.
            namespace (Namespace): Namespace tuple for the row.
            insertion_index (int): Caller-defined insertion index for deterministic
                tiebreaking within the SQL dense candidate list.

        Returns:
            ScoredRecord | None: Dense scored record when the similarity clears the
                shared threshold, otherwise ``None``.
        """

        cosine_distance = row.get("cosine_distance")
        if cosine_distance is None:
            return None
        similarity = 1.0 - float(cosine_distance)
        if similarity < EMBEDDING_MATCH_THRESHOLD:
            return None

        return ScoredRecord(
            record=PostgresMemoryStore._row_to_store_record(row, namespace),
            score=similarity,
            insertion_index=insertion_index,
        )

    @staticmethod
    def _row_to_store_record(
        row: dict[str, Any],
        namespace: Namespace,
    ) -> StoreRecord:
        """Convert a PostgreSQL row into a store record.

        Args:
            row (dict[str, Any]): Row from ``memory_records``.
            namespace (Namespace): Namespace tuple for the row.

        Returns:
            StoreRecord: Converted store record.
        """

        value = row["value"]
        if isinstance(value, str):
            parsed_value = cast(dict[str, Any], json.loads(value))
        else:
            parsed_value = cast(dict[str, Any], value)

        embedding = row.get("embedding")
        if embedding is not None:
            embedding = list(embedding)

        return StoreRecord(
            namespace=namespace,
            key=str(row["id"]),
            value=parsed_value,
            embedding=cast(list[float] | None, embedding),
            embedding_model=cast(str | None, row.get("embedding_model")),
        )

    async def aput(
        self,
        namespace: Namespace,
        key: str,
        value: dict[str, Any],
        *,
        embedding: list[float] | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """Store or overwrite one PostgreSQL-backed record.

        Args:
            namespace (Namespace): Record namespace tuple.
            key (str): Record key within the namespace.
            value (dict[str, Any]): Serialized record payload.
            embedding (list[float] | None): Optional precomputed embedding vector.
            embedding_model (str | None): Optional embedding model identifier.

        Returns:
            None: Writes the record to PostgreSQL.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)
        category = value.get("category")
        created_at = str(value.get("created_at") or "")
        last_referenced_at = str(value.get("last_referenced_at") or created_at or "")
        dormant_at = value.get("dormant_at")
        user_visible = bool(value.get("user_visible", True))
        embedding_dim = len(embedding) if embedding is not None else None
        embedding_vector_3072 = self._embedding_vector_3072_literal(
            embedding,
            embedding_dim,
        )

        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO memory_records
                        (id, owner_id, namespace_kind, category, value,
                         created_at, last_referenced_at, dormant_at, user_visible,
                         embedding, embedding_vector_3072, embedding_dim, embedding_model)
                    VALUES (
                        %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::vector(3072), %s, %s
                    )
                    ON CONFLICT (id, owner_id, namespace_kind) DO UPDATE SET
                        category = EXCLUDED.category,
                        value = EXCLUDED.value,
                        created_at = EXCLUDED.created_at,
                        last_referenced_at = EXCLUDED.last_referenced_at,
                        dormant_at = EXCLUDED.dormant_at,
                        user_visible = EXCLUDED.user_visible,
                        embedding = EXCLUDED.embedding,
                        embedding_vector_3072 = EXCLUDED.embedding_vector_3072,
                        embedding_dim = EXCLUDED.embedding_dim,
                        embedding_model = EXCLUDED.embedding_model
                    """,
                    (
                        key,
                        owner_id,
                        namespace_kind,
                        category,
                        json.dumps(value, default=str),
                        created_at,
                        last_referenced_at,
                        dormant_at,
                        user_visible,
                        embedding,
                        embedding_vector_3072,
                        embedding_dim,
                        embedding_model,
                    ),
                )

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
        """Write multiple PostgreSQL-backed records in one transaction.

        Args:
            items (list[tuple[Namespace, str, dict[str, Any], list[float] | None, str | None]]):
                Items shaped as ``(namespace, key, value, embedding, embedding_model)``.

        Returns:
            None: Commits the batch or rolls it back on failure.
        """

        if not items:
            return

        conn = await self._ensure_connection()
        async with conn.transaction():
            async with conn.cursor() as cursor:
                for namespace, key, value, embedding, embedding_model in items:
                    owner_id, namespace_kind = self._unpack_namespace(namespace)
                    category = value.get("category")
                    created_at = str(value.get("created_at") or "")
                    last_referenced_at = str(
                        value.get("last_referenced_at") or created_at or ""
                    )
                    dormant_at = value.get("dormant_at")
                    user_visible = bool(value.get("user_visible", True))
                    embedding_dim = len(embedding) if embedding is not None else None
                    embedding_vector_3072 = self._embedding_vector_3072_literal(
                        embedding,
                        embedding_dim,
                    )
                    await cursor.execute(
                        """
                        INSERT INTO memory_records
                            (id, owner_id, namespace_kind, category, value,
                             created_at, last_referenced_at, dormant_at, user_visible,
                             embedding, embedding_vector_3072, embedding_dim, embedding_model)
                        VALUES (
                            %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::vector(3072), %s, %s
                        )
                        ON CONFLICT (id, owner_id, namespace_kind) DO UPDATE SET
                            category = EXCLUDED.category,
                            value = EXCLUDED.value,
                            created_at = EXCLUDED.created_at,
                            last_referenced_at = EXCLUDED.last_referenced_at,
                            dormant_at = EXCLUDED.dormant_at,
                            user_visible = EXCLUDED.user_visible,
                            embedding = EXCLUDED.embedding,
                            embedding_vector_3072 = EXCLUDED.embedding_vector_3072,
                            embedding_dim = EXCLUDED.embedding_dim,
                            embedding_model = EXCLUDED.embedding_model
                        """,
                        (
                            key,
                            owner_id,
                            namespace_kind,
                            category,
                            json.dumps(value, default=str),
                            created_at,
                            last_referenced_at,
                            dormant_at,
                            user_visible,
                            embedding,
                            embedding_vector_3072,
                            embedding_dim,
                            embedding_model,
                        ),
                    )

    async def aget(
        self,
        namespace: Namespace,
        key: str,
    ) -> StoreRecord | None:
        """Fetch one PostgreSQL-backed record.

        Args:
            namespace (Namespace): Namespace tuple to search.
            key (str): Record key within the namespace.

        Returns:
            StoreRecord | None: Matching record, or ``None``.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT id, value, embedding, embedding_model
                FROM memory_records
                WHERE id = %s AND owner_id = %s AND namespace_kind = %s
                LIMIT 1
                """,
                (key, owner_id, namespace_kind),
            )
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
        """Search PostgreSQL-backed records with lexical scoring.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query (str | None): Optional lexical query. ``None`` enumerates records.
            limit (int): Maximum number of records to return.

        Returns:
            list[StoreRecord]: Matching records in recall-score order.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)

        async with conn.cursor() as cursor:
            if query is None:
                await cursor.execute(
                    """
                    SELECT id, value, embedding, embedding_model
                    FROM memory_records
                    WHERE owner_id = %s AND namespace_kind = %s
                    ORDER BY insertion_order ASC
                    LIMIT %s
                    """,
                    (owner_id, namespace_kind, limit),
                )
                rows = await cursor.fetchall()
                return [self._row_to_store_record(row, namespace) for row in rows]

            await cursor.execute(
                """
                SELECT id, value, embedding, embedding_model
                FROM memory_records
                WHERE owner_id = %s AND namespace_kind = %s
                ORDER BY insertion_order ASC
                """,
                (owner_id, namespace_kind),
            )
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
        record_filter: Callable[[StoreRecord], bool] | None = None,
    ) -> list[StoreRecord]:
        """Run hybrid retrieval over PostgreSQL-backed records.

        Args:
            namespace (Namespace): Namespace tuple to search.
            query_text (str): Query text for lexical scoring.
            query_embedding (list[float] | None): Optional dense query embedding.
            embedding_model (str | None): Optional query embedding model identifier.
            limit (int): Maximum number of records to return.
            max_age_days (int | None): Optional age filter in days.
            record_filter (Callable[[StoreRecord], bool] | None): Optional predicate
                applied to candidate records before ranking and truncation.

        Returns:
            list[StoreRecord]: Top fused retrieval results.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)

        age_clause = ""
        age_params: list[Any] = []
        if max_age_days is not None:
            age_clause = (
                " AND NULLIF(created_at, '')::timestamptz >= "
                "(NOW() AT TIME ZONE 'UTC') - (%s * INTERVAL '1 day')"
            )
            age_params.append(max_age_days)

        async with conn.cursor() as cursor:
            await cursor.execute(
                f"""
                SELECT id, value, embedding, embedding_model
                FROM memory_records
                WHERE owner_id = %s AND namespace_kind = %s{age_clause}
                ORDER BY insertion_order ASC
                """,
                [owner_id, namespace_kind, *age_params],
            )
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
                if record_filter(candidate.record)
            ]
        lexical_scored = lexical_rank(
            lexical_candidates,
            query_text=query_text,
            match_threshold=SEARCH_MATCH_THRESHOLD,
        )

        dense_scored = []
        if query_embedding is not None:
            dense_query_embedding = self._embedding_to_vector_literal(query_embedding)
            dense_candidate_limit = self._dense_candidate_limit(limit)
            embedding_model_clause = ""
            dense_params: list[Any] = [dense_query_embedding, owner_id, namespace_kind]
            dense_params.extend(age_params)
            if embedding_model is not None:
                embedding_model_clause = " AND embedding_model = %s"
                dense_params.append(embedding_model)
            dense_params.append(dense_candidate_limit)

            async with conn.cursor() as cursor:
                await cursor.execute(
                    f"""
                    SELECT
                        id,
                        value,
                        embedding,
                        embedding_model,
                        insertion_order,
                        embedding_vector_3072 <=> %s::vector(3072) AS cosine_distance
                    FROM memory_records
                    WHERE owner_id = %s AND namespace_kind = %s
                        AND embedding_vector_3072 IS NOT NULL{age_clause}{embedding_model_clause}
                    ORDER BY cosine_distance ASC, insertion_order ASC
                    LIMIT %s
                    """,
                    dense_params,
                )
                dense_rows = await cursor.fetchall()

            dense_scored = []
            for insertion_index, row in enumerate(dense_rows):
                record = self._row_to_store_record(row, namespace)
                if record_filter is not None and not record_filter(record):
                    continue
                scored = self._sql_dense_row_to_scored_record(
                    row,
                    namespace=namespace,
                    insertion_index=insertion_index,
                )
                if scored is not None:
                    dense_scored.append(scored)

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
        """Delete one PostgreSQL-backed record.

        Args:
            namespace (Namespace): Namespace tuple containing the record.
            key (str): Record key within the namespace.

        Returns:
            bool: ``True`` when a record was deleted.
        """

        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)
        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(
                    """
                    DELETE FROM memory_records
                    WHERE id = %s AND owner_id = %s AND namespace_kind = %s
                    """,
                    (key, owner_id, namespace_kind),
                )
                deleted = cursor.rowcount or 0
        return deleted > 0

    async def aclose(self) -> None:
        """Close the PostgreSQL store connection.

        Returns:
            None: Marks the store closed and releases the connection.
        """

        if self._closed:
            return
        async with self._connect_lock:
            self._closed = True
            if self._connection is not None:
                try:
                    await self._connection.close()
                except Exception:
                    logger.warning(
                        "PostgresMemoryStore: connection close raised; ignoring",
                        exc_info=True,
                    )
                finally:
                    self._connection = None

    async def arecord_count(self, namespace: Namespace | None = None) -> int:
        """Count PostgreSQL-backed records.

        Args:
            namespace (Namespace | None): Optional namespace filter.

        Returns:
            int: Total record count for the store or namespace.
        """

        if self._closed:
            return 0
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            if namespace is None:
                await cursor.execute("SELECT COUNT(*) AS count FROM memory_records")
            else:
                owner_id, namespace_kind = self._unpack_namespace(namespace)
                await cursor.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM memory_records
                    WHERE owner_id = %s AND namespace_kind = %s
                    """,
                    (owner_id, namespace_kind),
                )
            row = await cursor.fetchone()
        return int(row["count"]) if row else 0

    async def anamespaces(self) -> list[Namespace]:
        """List non-empty PostgreSQL namespaces.

        Returns:
            list[Namespace]: Namespaces that currently contain records.
        """

        if self._closed:
            return []
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT DISTINCT owner_id, namespace_kind
                FROM memory_records
                """
            )
            rows = await cursor.fetchall()
        return [(str(row["owner_id"]), str(row["namespace_kind"])) for row in rows]

    async def alatest(self, namespace: Namespace) -> StoreRecord | None:
        """Fetch the latest PostgreSQL-backed record in a namespace.

        Args:
            namespace (Namespace): Namespace tuple to inspect.

        Returns:
            StoreRecord | None: Most recent record, or ``None``.
        """

        if self._closed:
            return None
        conn = await self._ensure_connection()
        owner_id, namespace_kind = self._unpack_namespace(namespace)
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT id, value, embedding, embedding_model
                FROM memory_records
                WHERE owner_id = %s AND namespace_kind = %s
                ORDER BY insertion_order DESC
                LIMIT 1
                """,
                (owner_id, namespace_kind),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_store_record(row, namespace)

    @staticmethod
    def _unpack_namespace(namespace: Namespace) -> tuple[str, str]:
        """Extract normalized namespace fields from the tuple.

        Args:
            namespace (Namespace): Namespace tuple to validate and unpack.

        Returns:
            tuple[str, str]: ``(owner_id, namespace_kind)`` pair.
        """

        if len(namespace) != 2:
            raise ValueError(
                f"PostgresMemoryStore namespace must be (owner_id, kind) "
                f"tuple; got {namespace!r}"
            )
        owner_id, namespace_kind = namespace
        return str(owner_id), str(namespace_kind)


_: type[MemoryStore] = PostgresMemoryStore


__all__ = [
    "PostgresMemoryStore",
    "MEMORY_SCHEMA_DDL",
    "MEMORY_RECORDS_DDL",
    "MEMORY_RECORDS_INDEX_OWNER_KIND_DDL",
    "MEMORY_RECORDS_INDEX_LAST_REF_DDL",
]
