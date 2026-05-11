"""Integration tests for the PostgreSQL crisis-log backend."""

from __future__ import annotations

import os
from datetime import date
from uuid import uuid4

import psycopg
import pytest

from agent.audit.models import CrisisLogRecord
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend

_POSTGRES_TEST_URL_ENV = "OPENCOUCH_TEST_POSTGRES_URL"


def _postgres_database_url() -> str | None:
    """Return the opt-in Postgres DSN for backend integration tests.

    Returns:
        str | None: Configured Postgres DSN, or ``None`` when unavailable.
    """

    return os.getenv(_POSTGRES_TEST_URL_ENV) or os.getenv(
        "OPENCOUCH_MEMORY_DATABASE_URL"
    )


def _crisis_record(
    *,
    record_id: str,
    detected_at: str,
    level: int = 2,
    user_id: str | None = None,
    session_id_opaque: str | None = None,
) -> CrisisLogRecord:
    """Build a valid ``CrisisLogRecord`` for tests.

    Args:
        record_id (str): Unique record identifier.
        detected_at (str): ISO-8601 timestamp for the record.
        level (int): Crisis level for the record.
        user_id (str | None): Optional user identifier.
        session_id_opaque (str | None): Optional opaque session id.

    Returns:
        CrisisLogRecord: Valid crisis-log record for testing.
    """

    return CrisisLogRecord(
        id=record_id,
        session_id_opaque=session_id_opaque or ("a" * 64),
        user_id_or_null=user_id,
        detected_at=detected_at,
        level=level,  # type: ignore[arg-type]
        override_kind="none",
        classifier_path="llm_primary",
        reason="test",
        response_node_completed=True,
        llm_failure_occurred=False,
    )


async def _delete_records(dsn: str, record_ids: list[str]) -> None:
    """Delete test-owned rows from the shared Postgres crisis-log table.

    Args:
        dsn (str): PostgreSQL connection string.
        record_ids (list[str]): Record ids to delete.

    Returns:
        None: Removes matching rows in place.
    """

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM crisis_log WHERE id = ANY(%s)",
                (record_ids,),
            )


@pytest.mark.asyncio
async def test_postgres_crisis_log_round_trip_preserves_order() -> None:
    """Records appended on the same day should come back in insertion order."""

    dsn = _postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration DSN not configured; set "
            "OPENCOUCH_TEST_POSTGRES_URL or OPENCOUCH_MEMORY_DATABASE_URL"
        )

    record_ids = [f"crisis-log-{uuid4()}" for _ in range(3)]
    backend = PostgresCrisisLogBackend(dsn)

    try:
        await backend.aappend(
            _crisis_record(
                record_id=record_ids[0],
                detected_at="2099-04-10T09:00:00Z",
            )
        )
        await backend.aappend(
            _crisis_record(
                record_id=record_ids[1],
                detected_at="2099-04-10T12:00:00Z",
            )
        )
        await backend.aappend(
            _crisis_record(
                record_id=record_ids[2],
                detected_at="2099-04-10T23:00:00Z",
            )
        )

        results = await backend.alist_by_date(date(2099, 4, 10))
        filtered_ids = [record.id for record in results if record.id in set(record_ids)]
        assert filtered_ids == record_ids
    finally:
        await backend.aclose()
        await _delete_records(dsn, record_ids)


@pytest.mark.asyncio
async def test_postgres_crisis_log_persists_across_close_and_reopen() -> None:
    """File-backed SQLite semantics should hold across backend reopen in Postgres too."""

    dsn = _postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration DSN not configured; set "
            "OPENCOUCH_TEST_POSTGRES_URL or OPENCOUCH_MEMORY_DATABASE_URL"
        )

    record_id = f"crisis-log-{uuid4()}"
    backend_a = PostgresCrisisLogBackend(dsn)

    try:
        await backend_a.aappend(
            _crisis_record(
                record_id=record_id,
                detected_at="2099-04-11T08:00:00Z",
                level=3,
            )
        )
        await backend_a.aclose()

        backend_b = PostgresCrisisLogBackend(dsn)
        try:
            results = await backend_b.alist_by_date(date(2099, 4, 11))
            filtered = [record for record in results if record.id == record_id]
            assert len(filtered) == 1
            assert filtered[0].level == 3
        finally:
            await backend_b.aclose()
    finally:
        await _delete_records(dsn, [record_id])
