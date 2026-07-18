"""Shared helpers for Postgres persistence contract tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

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
