"""PostgreSQL-backed implementation of the :class:`MemoryStore` protocol.

:class:`PostgresMemoryStore` is the supported durable implementation of the
memory-store contract and persists records through a lazily opened psycopg
async connection.

Retrieval reuses the shared lexical and fusion helpers while pushing bounded
canonical-vector ranking and declarative filters into SQL. This keeps result
semantics aligned with the in-memory store without loading every
embedding into Python.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import psycopg
from psycopg.rows import dict_row

from agent.memory.providers.embeddings import DEFAULT_OPENAI_EMBEDDING_DIMENSION
from agent.memory.retrieval.ranking import (
    EMBEDDING_MATCH_THRESHOLD,
    IndexedRecord,
    ScoredRecord,
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
    prepare_memory_record_fields,
    unpack_memory_namespace,
)

logger = logging.getLogger(__name__)

PGVECTOR_FIXED_DIMENSION = DEFAULT_OPENAI_EMBEDDING_DIMENSION
MEMORY_SCHEMA_ADVISORY_LOCK_ID = 0x4F50434D454D
MEMORY_BACKFILL_ADVISORY_LOCK_ID = MEMORY_SCHEMA_ADVISORY_LOCK_ID + 1

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

MEMORY_SCHEMA_DDL: tuple[str, ...] = (
    MEMORY_RECORDS_DDL,
    MEMORY_RECORDS_INDEX_OWNER_KIND_DDL,
    MEMORY_RECORDS_INDEX_LAST_REF_DDL,
)


class PostgresMemoryStore:
    """PostgreSQL-backed implementation of :class:`MemoryStore`.

    Construction is cheap; the database connection opens on first use.
    Retrieval semantics intentionally match the SQLite store while canonical
    vector ranking and declarative filters execute in PostgreSQL.
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
        self._write_lock = asyncio.Lock()

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
                await cursor.execute("SET LOCAL lock_timeout = '10s'")
                await cursor.execute("SET LOCAL statement_timeout = '30s'")
                await cursor.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (MEMORY_SCHEMA_ADVISORY_LOCK_ID,),
                )
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

        async with conn.transaction():
            async with conn.cursor() as cursor:
                # A legacy table may take longer than startup-oriented DDL timeouts.
                await cursor.execute("SET LOCAL lock_timeout = '0'")
                await cursor.execute("SET LOCAL statement_timeout = '0'")
                await cursor.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (MEMORY_BACKFILL_ADVISORY_LOCK_ID,),
                )
                await cursor.execute(
                    """
                    UPDATE memory_records
                    SET embedding_vector_3072 = embedding::vector(3072)
                    WHERE embedding IS NOT NULL
                        AND embedding_dim = %s
                        AND embedding_vector_3072 IS NULL
                    """,
                    (PGVECTOR_FIXED_DIMENSION,),
                )

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

        embedding = row.get("embedding")
        if embedding is not None:
            embedding = list(embedding)

        return build_store_record(
            namespace=namespace,
            key=row["id"],
            value=row["value"],
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

        async with self._write_lock:
            conn = await self._ensure_connection()
            owner_id, namespace_kind = unpack_memory_namespace(namespace)
            fields = prepare_memory_record_fields(value, embedding=embedding)
            embedding_vector_3072 = self._embedding_vector_3072_literal(
                embedding,
                fields.embedding_dim,
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
                            fields.category,
                            fields.serialized_value,
                            fields.created_at,
                            fields.last_referenced_at,
                            fields.dormant_at,
                            fields.user_visible,
                            embedding,
                            embedding_vector_3072,
                            fields.embedding_dim,
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

        async with self._write_lock:
            conn = await self._ensure_connection()
            async with conn.transaction():
                async with conn.cursor() as cursor:
                    for namespace, key, value, embedding, embedding_model in items:
                        owner_id, namespace_kind = unpack_memory_namespace(namespace)
                        fields = prepare_memory_record_fields(
                            value, embedding=embedding
                        )
                        embedding_vector_3072 = self._embedding_vector_3072_literal(
                            embedding,
                            fields.embedding_dim,
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
                                fields.category,
                                fields.serialized_value,
                                fields.created_at,
                                fields.last_referenced_at,
                                fields.dormant_at,
                                fields.user_visible,
                                embedding,
                                embedding_vector_3072,
                                fields.embedding_dim,
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

        async with self._write_lock:
            conn = await self._ensure_connection()
            owner_id, namespace_kind = unpack_memory_namespace(namespace)
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

        async with self._write_lock:
            conn = await self._ensure_connection()
            owner_id, namespace_kind = unpack_memory_namespace(namespace)

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
        record_filter: MemoryRecordFilter | None = None,
    ) -> list[StoreRecord]:
        """Run hybrid retrieval in one coherent PostgreSQL snapshot."""

        async with self._write_lock:
            conn = await self._ensure_connection()
            async with conn.transaction():
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"
                    )
                return await self._asearch_similar_locked(
                    conn,
                    namespace,
                    query_text=query_text,
                    query_embedding=query_embedding,
                    embedding_model=embedding_model,
                    limit=limit,
                    max_age_days=max_age_days,
                    record_filter=record_filter,
                )

    async def _asearch_similar_locked(
        self,
        conn: psycopg.AsyncConnection[dict[str, Any]],
        namespace: Namespace,
        *,
        query_text: str,
        query_embedding: list[float] | None,
        embedding_model: str | None = None,
        limit: int = 10,
        max_age_days: int | None = None,
        record_filter: MemoryRecordFilter | None = None,
    ) -> list[StoreRecord]:
        """Run hybrid retrieval over PostgreSQL-backed records.

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

        owner_id, namespace_kind = unpack_memory_namespace(namespace)

        age_clause = ""
        age_params: list[Any] = []
        if max_age_days is not None:
            age_clause = (
                " AND NULLIF(created_at, '')::timestamptz >= "
                "NOW() - (%s * INTERVAL '1 day')"
            )
            age_params.append(max_age_days)

        record_filter_clause = ""
        if record_filter == "active_semantic":
            record_filter_clause = (
                " AND user_visible = TRUE"
                " AND NULLIF(dormant_at, '') IS NULL"
                " AND NULLIF(value->>'superseded_by', '') IS NULL"
            )
        elif record_filter is not None:
            raise ValueError(f"Unsupported memory record filter: {record_filter}")

        needs_python_dense = (
            query_embedding is not None
            and len(query_embedding) != PGVECTOR_FIXED_DIMENSION
        )
        embedding_select = (
            "embedding"
            if needs_python_dense
            else "NULL::double precision[] AS embedding"
        )
        async with conn.cursor() as cursor:
            await cursor.execute(
                f"""
                SELECT id, value, {embedding_select}, embedding_model
                FROM memory_records
                WHERE owner_id = %s AND namespace_kind = %s
                    {age_clause}{record_filter_clause}
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
        lexical_scored = lexical_rank(
            lexical_candidates,
            query_text=query_text,
            match_threshold=SEARCH_MATCH_THRESHOLD,
        )

        dense_scored = []
        if query_embedding is not None:
            if needs_python_dense:
                # Non-canonical configured providers remain supported through
                # the stored array column; only the canonical cohort can use the
                # fixed-dimension pgvector column.
                dense_scored = dense_rank(
                    lexical_candidates,
                    query_embedding=query_embedding,
                    embedding_model=embedding_model,
                )[: dense_candidate_limit(limit)]
            else:
                dense_query_embedding = self._embedding_to_vector_literal(
                    query_embedding
                )
                candidate_limit = dense_candidate_limit(limit)
                embedding_model_clause = ""
                dense_params: list[Any] = [owner_id, namespace_kind]
                dense_params.extend(age_params)
                dense_params.append(dense_query_embedding)
                if embedding_model is not None:
                    embedding_model_clause = " AND embedding_model = %s"
                    dense_params.append(embedding_model)
                dense_query = f"""
                    WITH candidates AS (
                        SELECT
                            id,
                            value,
                            embedding_model,
                            embedding_vector_3072,
                            insertion_order,
                            ROW_NUMBER() OVER (ORDER BY insertion_order ASC) - 1
                                AS candidate_index
                        FROM memory_records
                        WHERE owner_id = %s AND namespace_kind = %s
                            {age_clause}{record_filter_clause}
                    )
                    SELECT
                        id,
                        value,
                        NULL::double precision[] AS embedding,
                        embedding_model,
                        insertion_order,
                        candidate_index,
                        embedding_vector_3072 <=> %s::vector(3072) AS cosine_distance
                    FROM candidates
                    WHERE embedding_vector_3072 IS NOT NULL{embedding_model_clause}
                    ORDER BY cosine_distance ASC, insertion_order ASC
                    LIMIT %s
                """
                async with conn.cursor() as cursor:
                    await cursor.execute(
                        dense_query,
                        [*dense_params, candidate_limit],
                    )
                    dense_rows = await cursor.fetchall()

                for row in dense_rows:
                    scored = self._sql_dense_row_to_scored_record(
                        row,
                        namespace=namespace,
                        insertion_index=int(row["candidate_index"]),
                    )
                    if scored is not None:
                        dense_scored.append(scored)

        if not lexical_scored and not dense_scored:
            return []

        fused = rrf_fuse(
            lexical_ranked=lexical_scored,
            dense_ranked=dense_scored,
            limit=limit,
        )
        if not fused:
            return []
        keys = [record.key for record in fused]
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT id, value, embedding, embedding_model
                FROM memory_records
                WHERE owner_id = %s AND namespace_kind = %s AND id = ANY(%s)
                """,
                (owner_id, namespace_kind, keys),
            )
            hydrated_rows = await cursor.fetchall()
        hydrated = {
            row["id"]: self._row_to_store_record(row, namespace)
            for row in hydrated_rows
        }
        return [hydrated[key] for key in keys if key in hydrated]

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

        async with self._write_lock:
            conn = await self._ensure_connection()
            owner_id, namespace_kind = unpack_memory_namespace(namespace)
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
        # Serialize close with write transactions on the shared connection and
        # keep lock ordering consistent with write methods
        # (_write_lock -> _ensure_connection -> _connect_lock).
        async with self._write_lock:
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

        async with self._write_lock:
            if self._closed:
                return 0
            conn = await self._ensure_connection()
            async with conn.cursor() as cursor:
                if namespace is None:
                    await cursor.execute("SELECT COUNT(*) AS count FROM memory_records")
                else:
                    owner_id, namespace_kind = unpack_memory_namespace(namespace)
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

        async with self._write_lock:
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

        async with self._write_lock:
            if self._closed:
                return None
            conn = await self._ensure_connection()
            owner_id, namespace_kind = unpack_memory_namespace(namespace)
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


_: type[MemoryStore] = PostgresMemoryStore


__all__ = [
    "PostgresMemoryStore",
    "MEMORY_BACKFILL_ADVISORY_LOCK_ID",
    "MEMORY_SCHEMA_DDL",
    "MEMORY_RECORDS_DDL",
    "MEMORY_RECORDS_INDEX_OWNER_KIND_DDL",
    "MEMORY_RECORDS_INDEX_LAST_REF_DDL",
]
