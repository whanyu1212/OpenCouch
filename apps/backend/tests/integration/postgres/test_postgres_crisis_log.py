"""Integration tests for the PostgreSQL crisis-log backend."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import psycopg
import pytest

from agent.audit.models import CrisisLogRecord
from agent.audit.postgres_crisis_log import PostgresCrisisLogBackend
from tests.support.persistence import postgres_database_url


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
        response_path="sdk_tool_fallback",
        response_style="crisis_response",
        resource_lookup_status="no_verified_results",
        resource_count=2,
        tool_calls=["lookup_crisis_resources"],
        fallback_reason="crisis_resource_tool_not_called",
        trace_id="trace-1",
        trace_session_id="trace-session-1",
        trace_turn_id="turn-1",
        trace_runtime_mode="voice",
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

    dsn = postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration tests are disabled; set "
            "OPENCOUCH_ENABLE_POSTGRES_INTEGRATION_TESTS=1 and "
            "OPENCOUCH_TEST_POSTGRES_URL"
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

    dsn = postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration tests are disabled; set "
            "OPENCOUCH_ENABLE_POSTGRES_INTEGRATION_TESTS=1 and "
            "OPENCOUCH_TEST_POSTGRES_URL"
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
            assert filtered[0].event_type == "crisis_response"
            assert filtered[0].response_path == "sdk_tool_fallback"
            assert filtered[0].resource_lookup_status == "no_verified_results"
            assert filtered[0].resource_count == 2
            assert filtered[0].tool_calls == ["lookup_crisis_resources"]
            assert filtered[0].fallback_reason == "crisis_resource_tool_not_called"
            assert filtered[0].trace_id == "trace-1"
            assert filtered[0].trace_session_id == "trace-session-1"
            assert filtered[0].trace_turn_id == "turn-1"
            assert filtered[0].trace_runtime_mode == "voice"
        finally:
            await backend_b.aclose()
    finally:
        await _delete_records(dsn, [record_id])
