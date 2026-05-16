"""Integration tests for the PostgreSQL session-feedback backend."""

from __future__ import annotations

import os
from datetime import date
from uuid import uuid4

import psycopg
import pytest

from agent.audit.models import SessionFeedbackRecord
from agent.audit.postgres_session_feedback import PostgresSessionFeedbackBackend

_POSTGRES_TEST_URL_ENV = "OPENCOUCH_TEST_POSTGRES_URL"


def _postgres_database_url() -> str | None:
    """Return the opt-in Postgres DSN for backend integration tests.

    Returns:
        str | None: Configured Postgres DSN, or ``None`` when unavailable.
    """

    return os.getenv(_POSTGRES_TEST_URL_ENV) or os.getenv(
        "OPENCOUCH_MEMORY_DATABASE_URL"
    )


def _record(
    *,
    record_id: str,
    session: str,
    label: str = "positive",
    source: str = "cli_end",
    recorded_at: str = "2099-04-16T10:00:00Z",
    user_id: str | None = None,
    turn_count: int = 3,
) -> SessionFeedbackRecord:
    """Produce a valid ``SessionFeedbackRecord`` for testing.

    Args:
        record_id (str): Unique record identifier.
        session (str): Opaque session identifier.
        label (str): Feedback label.
        source (str): Feedback source.
        recorded_at (str): ISO-8601 feedback timestamp.
        user_id (str | None): Optional user identifier.
        turn_count (int): Turn count at session end.

    Returns:
        SessionFeedbackRecord: Valid feedback record for tests.
    """

    return SessionFeedbackRecord(
        id=record_id,
        session_id_opaque=session,
        user_id_or_null=user_id,
        recorded_at=recorded_at,
        label=label,  # type: ignore[arg-type]
        turn_count_at_end=turn_count,
        source=source,  # type: ignore[arg-type]
    )


async def _delete_records(dsn: str, record_ids: list[str]) -> None:
    """Delete test-owned rows from the shared Postgres feedback table.

    Args:
        dsn (str): PostgreSQL connection string.
        record_ids (list[str]): Record ids to delete.

    Returns:
        None: Removes matching rows in place.
    """

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM session_feedback WHERE id = ANY(%s)",
                (record_ids,),
            )


@pytest.mark.asyncio
async def test_postgres_feedback_round_trip_preserves_order() -> None:
    """Records appended for the same session should come back in insertion order."""

    dsn = _postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration DSN not configured; set "
            "OPENCOUCH_TEST_POSTGRES_URL or OPENCOUCH_MEMORY_DATABASE_URL"
        )

    session_id = f"session-{uuid4()}"
    record_ids = [f"feedback-{uuid4()}" for _ in range(3)]
    backend = PostgresSessionFeedbackBackend(dsn)

    try:
        await backend.aappend(_record(record_id=record_ids[0], session=session_id))
        await backend.aappend(
            _record(
                record_id=record_ids[1],
                session=session_id,
                label="negative",
                source="api_end",
                recorded_at="2099-04-16T11:00:00Z",
            )
        )
        await backend.aappend(
            _record(
                record_id=record_ids[2],
                session=session_id,
                label="skip",
                source="cli_exit",
                recorded_at="2099-04-16T12:00:00Z",
            )
        )

        results = await backend.alist_by_session(session_id)
        assert [record.id for record in results] == record_ids
    finally:
        await backend.aclose()
        await _delete_records(dsn, record_ids)


@pytest.mark.asyncio
async def test_postgres_feedback_persists_across_close_and_reopen() -> None:
    """Session feedback should survive backend close and reopen in Postgres."""

    dsn = _postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration DSN not configured; set "
            "OPENCOUCH_TEST_POSTGRES_URL or OPENCOUCH_MEMORY_DATABASE_URL"
        )

    session_id = f"session-{uuid4()}"
    record_id = f"feedback-{uuid4()}"
    backend_a = PostgresSessionFeedbackBackend(dsn)

    try:
        await backend_a.aappend(
            _record(
                record_id=record_id,
                session=session_id,
                recorded_at="2099-04-17T08:00:00Z",
                turn_count=7,
            )
        )
        await backend_a.aclose()

        backend_b = PostgresSessionFeedbackBackend(dsn)
        try:
            results = await backend_b.alist_by_session(session_id)
            assert len(results) == 1
            assert results[0].id == record_id
            assert results[0].turn_count_at_end == 7
        finally:
            await backend_b.aclose()
    finally:
        await _delete_records(dsn, [record_id])


@pytest.mark.asyncio
async def test_postgres_feedback_purge_before_uses_exclusive_boundary() -> None:
    """Records on the cutoff date survive; only older rows are purged."""

    dsn = _postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration DSN not configured; set "
            "OPENCOUCH_TEST_POSTGRES_URL or OPENCOUCH_MEMORY_DATABASE_URL"
        )

    session_id = f"session-{uuid4()}"
    record_ids = [f"feedback-{uuid4()}" for _ in range(3)]
    backend = PostgresSessionFeedbackBackend(dsn)

    try:
        await backend.aappend(
            _record(
                record_id=record_ids[0],
                session=session_id,
                recorded_at="2099-04-14T10:00:00Z",
            )
        )
        await backend.aappend(
            _record(
                record_id=record_ids[1],
                session=session_id,
                recorded_at="2099-04-15T10:00:00Z",
            )
        )
        await backend.aappend(
            _record(
                record_id=record_ids[2],
                session=session_id,
                recorded_at="2099-04-16T10:00:00Z",
            )
        )

        deleted = await backend.apurge_before(date(2099, 4, 15))
        assert deleted >= 1
        results = await backend.alist_by_session(session_id)
        assert [record.id for record in results] == record_ids[1:]
    finally:
        await backend.aclose()
        await _delete_records(dsn, record_ids)
