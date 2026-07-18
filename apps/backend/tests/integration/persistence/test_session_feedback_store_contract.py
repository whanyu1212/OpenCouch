"""Persistence contracts for the supported durable feedback backend."""

from __future__ import annotations

import asyncio
from datetime import date
from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from agent.feedback.models import SessionFeedbackRecord
from tests.support.persistence_contracts import (
    open_postgres_session_feedback_backend,
    require_postgres_database_url,
)

pytestmark = pytest.mark.asyncio


def _record(
    *,
    record_id: str | None = None,
    session_id: str = "session-contract",
    recorded_at: str = "2099-05-10T10:00:00Z",
    label: str = "positive",
    source: str = "cli_end",
    modality: str = "text",
) -> SessionFeedbackRecord:
    """Build a complete session-feedback record for durable contracts."""

    return SessionFeedbackRecord(
        id=record_id or f"feedback-{uuid4()}",
        session_id_opaque=session_id,
        user_id_or_null="user-contract",
        recorded_at=recorded_at,
        label=label,  # type: ignore[arg-type]
        turn_count_at_end=7,
        source=source,  # type: ignore[arg-type]
        modality=modality,  # type: ignore[arg-type]
    )


async def test_feedback_round_trip_preserves_complete_record_and_order() -> None:
    """JSONB storage should preserve complete records in append order."""

    session_id = f"session-{uuid4()}"
    records = [
        _record(session_id=session_id),
        _record(
            session_id=session_id,
            recorded_at="2099-05-10T11:00:00Z",
            label="negative",
            source="api_end",
            modality="voice",
        ),
        _record(
            session_id=session_id,
            recorded_at="2099-05-10T12:00:00Z",
            label="skip",
            source="cli_exit",
        ),
    ]

    async with open_postgres_session_feedback_backend() as backend:
        for record in records:
            await backend.aappend(record)

        assert await backend.alist_by_session(session_id) == records


async def test_feedback_isolates_sessions_and_counts_all_records() -> None:
    """Session reads should be isolated while count spans all sessions."""

    first = _record(session_id=f"session-a-{uuid4()}")
    second = _record(session_id=f"session-b-{uuid4()}")

    async with open_postgres_session_feedback_backend() as backend:
        assert await backend.alist_by_session("missing") == []
        assert await backend.arecord_count() == 0
        await backend.aappend(first)
        await backend.aappend(second)

        assert await backend.alist_by_session(first.session_id_opaque) == [first]
        assert await backend.alist_by_session(second.session_id_opaque) == [second]
        assert await backend.arecord_count() == 2


async def test_feedback_persists_order_across_close_and_reopen() -> None:
    """Feedback and append ordering should survive a new connection."""

    session_id = f"session-{uuid4()}"
    records = [
        _record(session_id=session_id, recorded_at=f"2099-05-11T10:0{i}:00Z")
        for i in range(3)
    ]

    async with open_postgres_session_feedback_backend() as backend:
        for record in records:
            await backend.aappend(record)

    async with open_postgres_session_feedback_backend() as backend:
        assert await backend.alist_by_session(session_id) == records
        assert await backend.arecord_count() == 3


async def test_feedback_writes_are_visible_across_connections() -> None:
    """Committed feedback should be visible through another live backend."""

    record = _record(session_id=f"session-{uuid4()}")

    async with open_postgres_session_feedback_backend() as writer:
        async with open_postgres_session_feedback_backend() as reader:
            await writer.aappend(record)
            assert await reader.alist_by_session(record.session_id_opaque) == [record]


async def test_feedback_purge_is_exclusive_idempotent_and_durable() -> None:
    """Purge should preserve cutoff-day feedback and survive reconnect."""

    session_id = f"session-{uuid4()}"
    old = _record(session_id=session_id, recorded_at="2099-05-12T10:00:00Z")
    cutoff = _record(session_id=session_id, recorded_at="2099-05-13T10:00:00Z")

    async with open_postgres_session_feedback_backend() as backend:
        assert await backend.apurge_before(date(2099, 5, 13)) == 0
        await backend.aappend(old)
        await backend.aappend(cutoff)
        assert await backend.apurge_before(date(2099, 5, 13)) == 1
        assert await backend.apurge_before(date(2099, 5, 13)) == 0

    async with open_postgres_session_feedback_backend() as backend:
        assert await backend.alist_by_session(session_id) == [cutoff]
        assert await backend.arecord_count() == 1


async def test_feedback_close_contract() -> None:
    """Close should be idempotent and block stateful operations."""

    async with open_postgres_session_feedback_backend() as backend:
        await backend.aappend(_record())
        await backend.aclose()
        await backend.aclose()

        assert await backend.arecord_count() == 0
        assert await backend.apurge_before(date(2099, 5, 11)) == 0
        with pytest.raises(RuntimeError, match="closed"):
            await backend.aappend(_record())
        with pytest.raises(RuntimeError, match="closed"):
            await backend.alist_by_session("session-contract")


async def test_feedback_allows_duplicate_ids() -> None:
    """Feedback appends are events, so duplicate opaque IDs remain allowed."""

    record = _record(record_id=f"duplicate-{uuid4()}")

    async with open_postgres_session_feedback_backend() as backend:
        await backend.aappend(record)
        await backend.aappend(record)

        assert await backend.alist_by_session(record.session_id_opaque) == [
            record,
            record,
        ]
        assert await backend.arecord_count() == 2


async def test_feedback_rejects_malformed_timestamp() -> None:
    """Malformed timestamps must fail instead of entering invalid date buckets."""

    async with open_postgres_session_feedback_backend() as backend:
        with pytest.raises(ValueError):
            await backend.aappend(_record(recorded_at="not-a-timestamp"))


async def test_feedback_concurrent_appends_all_persist() -> None:
    """Concurrent callers should not lose feedback on the shared connection."""

    session_id = f"session-{uuid4()}"
    records = [_record(session_id=session_id) for _ in range(10)]

    async with open_postgres_session_feedback_backend() as backend:
        await asyncio.gather(*(backend.aappend(record) for record in records))
        stored = await backend.alist_by_session(session_id)

        assert {record.id for record in stored} == {record.id for record in records}
        assert await backend.arecord_count() == len(records)


@pytest.mark.parametrize(
    ("label", "source"),
    [("invalid-label", "cli_end"), ("positive", "invalid-source")],
)
async def test_feedback_schema_rejects_invalid_enums(label: str, source: str) -> None:
    """Postgres constraints should reject values that bypass model validation."""

    async with open_postgres_session_feedback_backend() as backend:
        await backend.arecord_count()

    dsn = require_postgres_database_url()
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        async with conn.cursor() as cursor:
            with pytest.raises(psycopg.errors.CheckViolation):
                await cursor.execute(
                    """
                    INSERT INTO session_feedback
                        (id, session_id_opaque, recorded_at, recorded_date,
                         label, turn_count_at_end, source, schema_version, value)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        f"invalid-{uuid4()}",
                        "constraint-session",
                        "2099-05-14T10:00:00Z",
                        "2099-05-14",
                        label,
                        0,
                        source,
                        1,
                        Jsonb({}),
                    ),
                )
