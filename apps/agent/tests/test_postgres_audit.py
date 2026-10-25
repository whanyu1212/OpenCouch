"""Integration tests for the Postgres crisis audit log.

Set `OPENCOUCH_TEST_POSTGRES_URL` to run them; they are skipped otherwise.
"""

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from psycopg_pool import AsyncConnectionPool

from opencouch_agent.safety import FailureReason, RiskLevel
from opencouch_agent.safety.audit import AuditEntry
from opencouch_agent.safety.postgres_audit import PostgresCrisisAuditLog

POSTGRES_URL = os.environ.get("OPENCOUCH_TEST_POSTGRES_URL")
AT = datetime(2026, 10, 25, 9, 0, tzinfo=UTC)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="OPENCOUCH_TEST_POSTGRES_URL is not set"
)


@pytest.fixture
async def pool() -> AsyncIterator[AsyncConnectionPool]:
    assert POSTGRES_URL
    async with AsyncConnectionPool(POSTGRES_URL) as connection_pool:
        await PostgresCrisisAuditLog(connection_pool).create_schema()
        yield connection_pool


async def _rows(pool: AsyncConnectionPool, thread_id: str) -> list[tuple[object, ...]]:
    async with pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT thread_id, turn_at, level_before, level_after, reason,"
            " screening_failure FROM crisis_audit_log"
            " WHERE thread_id = %s ORDER BY id",
            (thread_id,),
        )
        return await cursor.fetchall()


async def test_entries_are_stored_in_order_with_enum_values(
    pool: AsyncConnectionPool,
) -> None:
    thread_id = f"test-{uuid.uuid4()}"
    log = PostgresCrisisAuditLog(pool)

    await log.append(
        AuditEntry(thread_id, AT, RiskLevel.NONE, RiskLevel.CRISIS, "CRISIS: intent")
    )
    await log.append(
        AuditEntry(
            thread_id,
            AT,
            RiskLevel.CRISIS,
            RiskLevel.CRISIS,
            "screening failed: kept the stored level",
            FailureReason.TIMEOUT,
        )
    )

    assert await _rows(pool, thread_id) == [
        (thread_id, AT, "none", "crisis", "CRISIS: intent", None),
        (
            thread_id,
            AT,
            "crisis",
            "crisis",
            "screening failed: kept the stored level",
            "timeout",
        ),
    ]


async def test_an_entry_needs_no_conversation_row(pool: AsyncConnectionPool) -> None:
    # Incognito threads (HY-205) never get a row in `conversations`.
    thread_id = f"incognito-{uuid.uuid4()}"

    await PostgresCrisisAuditLog(pool).append(
        AuditEntry(thread_id, AT, RiskLevel.NONE, RiskLevel.WATCH, "WATCH")
    )

    assert len(await _rows(pool, thread_id)) == 1
