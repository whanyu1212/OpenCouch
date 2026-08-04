"""Cold-startup preparation keeps migrations off the request path.

Crisis appends run inside a bounded safety-capture timeout, and the memory
store's preparation includes a vector-column backfill. Both must happen at
startup rather than inside whichever user request arrives first.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from sqlalchemy import event

from agent.audit.capture import (
    DEFAULT_SAFETY_EVENT_CAPTURE_TIMEOUT_SECONDS,
)
from agent.audit.models import CrisisLogRecord
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.memory.store.postgres import PostgresMemoryStore
from agent.runtime.session_store import TextSessionStore, TextSessionStoreConfig
from tests.support.persistence_contracts import require_postgres_database_url

pytestmark = pytest.mark.asyncio


def _crisis_record() -> CrisisLogRecord:
    return CrisisLogRecord(
        id=f"crisis-{uuid4()}",
        event_type="crisis_response",
        session_id_opaque="a" * 64,
        user_id_or_null="user-startup-prep",
        detected_at="2026-07-30T00:00:00Z",
        level=2,
        override_kind="none",
        classifier_path="llm_primary",
        reason="startup preparation test record",
        response_node_completed=True,
        llm_failure_occurred=False,
        response_path="sdk_tool_fallback",
        response_style="crisis_response",
        resource_lookup_status="no_verified_results",
    )


class _StatementRecordingConnection:
    """Wrapper that records every SQL statement executed through a cursor."""

    def __init__(self, conn: Any, statements: list[str]) -> None:
        self._conn = conn
        self._statements = statements

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return _StatementRecordingCursor(
            self._conn.cursor(*args, **kwargs), self._statements
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class _StatementRecordingCursor:
    def __init__(self, cursor: Any, statements: list[str]) -> None:
        self._cursor = cursor
        self._statements = statements

    async def __aenter__(self) -> "_StatementRecordingCursor":
        await self._cursor.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        return await self._cursor.__aexit__(*args)

    async def execute(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        self._statements.append(str(query))
        return await self._cursor.execute(query, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


async def test_prepared_crisis_append_runs_no_ddl_within_capture_budget() -> None:
    """After startup preparation, the first append is data work only.

    The dominant cold-start cost is connection establishment rather than the
    DDL itself, and connection latency is what degrades against a remote
    database. Asserting on statements rather than elapsed time keeps this
    meaningful without making it a timing-sensitive flake.
    """

    backend = PostgresCrisisLogBackend(require_postgres_database_url())
    try:
        await backend.ensure_schema()
        assert backend._connection is not None, (  # noqa: SLF001
            "preparation should establish the connection before serving traffic"
        )

        statements: list[str] = []
        backend._connection = _StatementRecordingConnection(  # noqa: SLF001
            backend._connection,  # noqa: SLF001
            statements,
        )

        # The real capture path bounds appends by this timeout.
        await asyncio.wait_for(
            backend.aappend(_crisis_record()),
            timeout=DEFAULT_SAFETY_EVENT_CAPTURE_TIMEOUT_SECONDS,
        )

        assert statements, "expected the append to issue at least one statement"
        migration_statements = [
            statement
            for statement in statements
            if any(
                keyword in statement.upper()
                for keyword in ("CREATE TABLE", "CREATE INDEX", "ALTER TABLE")
            )
        ]
        assert migration_statements == []
    finally:
        await backend.aclose()


async def test_memory_preparation_runs_backfill_before_serving_traffic() -> None:
    """Vector-column migration happens at preparation, not on first retrieval."""

    store = PostgresMemoryStore(require_postgres_database_url())
    try:
        await store.ensure_schema()

        statements: list[str] = []
        store._connection = _StatementRecordingConnection(  # noqa: SLF001
            store._connection,  # noqa: SLF001
            statements,
        )

        owner_id = f"startup-prep-{uuid4()}"
        await store.asearch((owner_id, "semantic"), query=None, limit=5)

        migration_statements = [
            statement
            for statement in statements
            if any(
                keyword in statement.upper()
                for keyword in ("CREATE TABLE", "CREATE EXTENSION", "ALTER TABLE")
            )
        ]
        assert migration_statements == []
    finally:
        await store.aclose()


async def test_prepared_text_session_runs_no_ddl_on_first_thread_read() -> None:
    """Startup preparation removes SDK table creation from the request path."""

    store = TextSessionStore(
        TextSessionStoreConfig(
            backend="sqlalchemy",
            database_url=require_postgres_database_url(),
            create_tables=True,
        )
    )
    try:
        await store.ensure_schema()
        assert store._engine is not None  # noqa: SLF001

        statements: list[str] = []

        def _record_statement(
            _conn: Any,
            _cursor: Any,
            statement: str,
            _parameters: Any,
            _context: Any,
            _executemany: bool,
        ) -> None:
            statements.append(statement)

        event.listen(
            store._engine.sync_engine,  # noqa: SLF001
            "before_cursor_execute",
            _record_statement,
        )
        try:
            await store.session_for_thread("startup-prepared-thread").get_items()
        finally:
            event.remove(
                store._engine.sync_engine,  # noqa: SLF001
                "before_cursor_execute",
                _record_statement,
            )

        assert statements, "expected the history read to issue a SELECT"
        migration_statements = [
            statement
            for statement in statements
            if any(
                keyword in statement.upper()
                for keyword in ("CREATE TABLE", "CREATE INDEX", "ALTER TABLE")
            )
        ]
        assert migration_statements == []
    finally:
        await store.aclose()


async def test_preparation_creates_schema_for_every_durable_backend() -> None:
    """Each backend's tables exist after preparation, before any operation."""

    dsn = require_postgres_database_url()
    memory_store = PostgresMemoryStore(dsn)
    crisis_backend = PostgresCrisisLogBackend(dsn)
    feedback_backend = PostgresSessionFeedbackBackend(dsn)
    text_session_store = TextSessionStore(
        TextSessionStoreConfig(
            backend="sqlalchemy",
            database_url=dsn,
            create_tables=True,
        )
    )
    try:
        await memory_store.ensure_schema()
        await crisis_backend.ensure_schema()
        await feedback_backend.ensure_schema()
        await text_session_store.ensure_schema()

        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            async with conn.cursor() as cursor:
                for table in (
                    "memory_records",
                    "crisis_log",
                    "session_feedback",
                    "agent_sessions",
                    "agent_messages",
                ):
                    await cursor.execute("SELECT to_regclass(%s)", (table,))
                    row = await cursor.fetchone()
                    assert row is not None and row[0] is not None, (
                        f"{table} should exist after preparation"
                    )
    finally:
        await memory_store.aclose()
        await crisis_backend.aclose()
        await feedback_backend.aclose()
        await text_session_store.aclose()
