"""Shared fixtures for persistence integration tests.

The Postgres tests in this package run against a single shared database.
Without isolation, rows written by one test — including finalize-on-close writes
under owner ids a later test cannot enumerate — leak into the next and break
tests that assert exact result sets (e.g. ``list_thread_ids`` ordering).

The autouse fixture below truncates the shared tables before and after each
test when Postgres is enabled, giving every test a clean slate. When Postgres
is disabled it is a no-op, so local runs without a configured database are
unaffected.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from tests.support.persistence import (
    postgres_database_url,
    truncate_postgres_tables,
)

# Shared Postgres tables that accumulate rows across persistence tests.
_SHARED_POSTGRES_TABLES = (
    "opencouch_thread_state",
    "opencouch_active_sessions",
    "memory_records",
    "crisis_log",
    "session_feedback",
)


@pytest.fixture(autouse=True)
async def _isolate_postgres_persistence() -> AsyncIterator[None]:
    """Truncate shared Postgres tables before and after each test.

    No-op when Postgres integration tests are disabled, so local runs without
    an explicitly configured database remain unaffected.
    """

    dsn = postgres_database_url()
    if not dsn:
        # ``postgres_database_url`` returns ``None`` when the suite is disabled
        # and ``""`` when enabled with an empty URL; both mean "not configured".
        # Matching the existing Postgres tests' ``if not dsn`` guard avoids
        # psycopg.connect("") falling back to the local default database.
        yield
        return

    await truncate_postgres_tables(dsn, *_SHARED_POSTGRES_TABLES)
    try:
        yield
    finally:
        await truncate_postgres_tables(dsn, *_SHARED_POSTGRES_TABLES)
