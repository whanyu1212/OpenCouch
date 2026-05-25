"""Shared helpers for persistence backend contract tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, TypeAlias

import pytest

from agent.runtime.session.active_session import (
    ActiveSessionStore,
    PostgresActiveSessionStore,
    SqliteActiveSessionStore,
)
from agent.runtime.state_store import (
    PostgresRuntimeStateStore,
    RuntimeStateStore,
    SqliteRuntimeStateStore,
)
from tests.support.persistence import postgres_database_url

PersistenceBackend: TypeAlias = Literal["sqlite", "postgres"]


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
async def open_runtime_state_store(
    backend: PersistenceBackend,
    *,
    tmp_path: Path,
) -> AsyncIterator[RuntimeStateStore]:
    """Open one runtime-state store implementation for a contract test."""

    if backend == "sqlite":
        store: RuntimeStateStore = SqliteRuntimeStateStore(
            tmp_path / "runtime-state-contract.sqlite3"
        )
    else:
        store = PostgresRuntimeStateStore(require_postgres_database_url())

    await store.ensure_schema()
    try:
        yield store
    finally:
        await store.aclose()


@asynccontextmanager
async def open_active_session_store(
    backend: PersistenceBackend,
    *,
    tmp_path: Path,
) -> AsyncIterator[ActiveSessionStore]:
    """Open one active-session store implementation for a contract test."""

    if backend == "sqlite":
        store: ActiveSessionStore = SqliteActiveSessionStore(
            sqlite_path=tmp_path / "active-session-contract.sqlite3"
        )
    else:
        store = PostgresActiveSessionStore(dsn=require_postgres_database_url())

    await store.ensure_schema()
    try:
        yield store
    finally:
        await store.aclose()
