"""Tests for PostgreSQL memory-store schema definitions."""

from __future__ import annotations

from typing import Any

import pytest

from agent.memory.store import postgres


def test_schema_does_not_create_invalid_3072_hnsw_index() -> None:
    """The default 3072-d embedding column must not get an HNSW index.

    Returns:
        None: Asserts the generated schema avoids pgvector's vector HNSW
            dimension limit while keeping the fixed vector column available.
    """

    schema = "\n".join(postgres.MEMORY_SCHEMA_DDL)

    assert "embedding_vector_3072 vector(3072)" in schema
    assert "USING hnsw" not in schema
    assert "idx_memory_embedding_vector_3072_hnsw" not in schema


@pytest.mark.asyncio
async def test_backfill_disables_startup_timeouts() -> None:
    """Large legacy backfills may wait and run longer than schema DDL."""

    statements: list[str] = []

    class _Context:
        def __init__(self, value: Any = None) -> None:
            self.value = value

        async def __aenter__(self) -> Any:
            return self.value

        async def __aexit__(self, *_args: object) -> None:
            return None

    class _Cursor:
        async def execute(self, statement: str, _params: object = None) -> None:
            statements.append(statement)

    class _Connection:
        def transaction(self) -> _Context:
            return _Context()

        def cursor(self) -> _Context:
            return _Context(_Cursor())

    await postgres.PostgresMemoryStore._backfill_embedding_vector_3072(  # noqa: SLF001
        _Connection()  # type: ignore[arg-type]
    )

    assert statements[:2] == [
        "SET LOCAL lock_timeout = '0'",
        "SET LOCAL statement_timeout = '0'",
    ]
