"""Shared helpers for Postgres persistence contract tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import psycopg
import pytest

from agent.audit.crisis_log import CrisisLogBackend
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from agent.feedback.postgres_session_feedback import PostgresSessionFeedbackBackend
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.memory.store import MemoryStore
from agent.memory.store.postgres import PostgresMemoryStore
from agent.runtime.session.active_session import (
    ActiveSessionStore,
    PostgresActiveSessionStore,
)
from agent.runtime.state_store import PostgresRuntimeStateStore, RuntimeStateStore
from tests.support.persistence import postgres_database_url


def require_postgres_database_url() -> str:
    """Return the enabled Postgres DSN or skip the test."""

    dsn = postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration tests are disabled; set "
            "OPENCOUCH_ENABLE_POSTGRES_INTEGRATION_TESTS=1 and "
            "OPENCOUCH_TEST_POSTGRES_URL"
        )
    return dsn


async def delete_postgres_memory_records_for_owners(
    dsn: str,
    owner_ids: Sequence[str],
) -> None:
    """Delete memory rows owned by one contract-test cohort."""

    if not owner_ids:
        return
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("SELECT to_regclass('public.memory_records')")
            if await cursor.fetchone() == (None,):
                return
            await cursor.execute(
                "DELETE FROM memory_records WHERE owner_id = ANY(%s)",
                (list(owner_ids),),
            )


@asynccontextmanager
async def open_postgres_memory_store(
    *,
    owner_ids: Sequence[str] = (),
) -> AsyncIterator[MemoryStore]:
    """Open the supported durable memory store for a contract test."""

    dsn = require_postgres_database_url()
    store = PostgresMemoryStore(dsn)
    try:
        yield store
    finally:
        await store.aclose()
        await delete_postgres_memory_records_for_owners(dsn, owner_ids)


@asynccontextmanager
async def open_postgres_runtime_state_store() -> AsyncIterator[RuntimeStateStore]:
    """Open the supported durable runtime-state store for a contract test."""

    store = PostgresRuntimeStateStore(require_postgres_database_url())

    await store.ensure_schema()
    try:
        yield store
    finally:
        await store.aclose()


@asynccontextmanager
async def open_postgres_active_session_store() -> AsyncIterator[ActiveSessionStore]:
    """Open the supported durable active-session store for a contract test."""

    store = PostgresActiveSessionStore(dsn=require_postgres_database_url())

    await store.ensure_schema()
    try:
        yield store
    finally:
        await store.aclose()


@asynccontextmanager
async def open_postgres_crisis_log_backend() -> AsyncIterator[CrisisLogBackend]:
    """Open the supported durable crisis-log backend for a contract test."""

    backend = PostgresCrisisLogBackend(require_postgres_database_url())
    try:
        yield backend
    finally:
        await backend.aclose()


@asynccontextmanager
async def open_postgres_session_feedback_backend() -> AsyncIterator[
    SessionFeedbackBackend
]:
    """Open the supported durable feedback backend for a contract test."""

    backend = PostgresSessionFeedbackBackend(require_postgres_database_url())
    try:
        yield backend
    finally:
        await backend.aclose()
