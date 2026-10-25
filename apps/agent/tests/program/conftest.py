"""Fixtures for the program records tests: every test runs against both stores.

The Postgres runs need `OPENCOUCH_TEST_POSTGRES_URL`; they're skipped otherwise.
"""

import os
import uuid
from collections.abc import AsyncIterator

import pytest
from psycopg_pool import AsyncConnectionPool

from opencouch_agent.program.postgres import PostgresRecordStore
from opencouch_agent.program.records import InMemoryRecordStore, RecordStore

POSTGRES_URL = os.environ.get("OPENCOUCH_TEST_POSTGRES_URL")


@pytest.fixture(params=["memory", "postgres"])
async def records(request: pytest.FixtureRequest) -> AsyncIterator[RecordStore]:
    if request.param == "memory":
        yield InMemoryRecordStore()
        return
    if not POSTGRES_URL:
        pytest.skip("OPENCOUCH_TEST_POSTGRES_URL is not set")
    async with AsyncConnectionPool(POSTGRES_URL) as pool:
        postgres = PostgresRecordStore(pool)
        await postgres.create_schema()
        yield postgres


@pytest.fixture
async def alice(records: RecordStore) -> AsyncIterator[str]:
    # Unique per test, so Postgres runs don't see each other's rows, and
    # deleted afterwards so they don't pile up.
    user = f"alice-{uuid.uuid4()}"
    yield user
    await records.delete_user(user)


@pytest.fixture
async def bob(records: RecordStore) -> AsyncIterator[str]:
    user = f"bob-{uuid.uuid4()}"
    yield user
    await records.delete_user(user)
