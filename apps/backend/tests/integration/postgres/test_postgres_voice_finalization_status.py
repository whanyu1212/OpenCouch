"""Integration tests for PostgreSQL-backed voice finalization status."""

from __future__ import annotations

import os
from uuid import uuid4

import psycopg
import pytest

from agent.voice.finalization_status import (
    get_voice_finalization_status,
    set_voice_finalization_status,
)

_POSTGRES_TEST_URL_ENV = "OPENCOUCH_TEST_POSTGRES_URL"


def _postgres_database_url() -> str | None:
    """Return the opt-in Postgres DSN for backend integration tests.

    Returns:
        str | None: Configured Postgres DSN, or ``None`` when unavailable.
    """

    return os.getenv(_POSTGRES_TEST_URL_ENV) or os.getenv(
        "OPENCOUCH_MEMORY_DATABASE_URL"
    )


async def _delete_status(dsn: str, thread_id: str) -> None:
    """Delete one test-owned voice finalization row from Postgres.

    Args:
        dsn (str): PostgreSQL connection string.
        thread_id (str): Thread identifier to delete.

    Returns:
        None: Removes the row in place.
    """

    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                DELETE FROM opencouch_voice_finalization_status
                WHERE thread_id = %s
                """,
                (thread_id,),
            )


@pytest.mark.asyncio
async def test_postgres_voice_finalization_status_roundtrips() -> None:
    """Voice finalization status should persist and update in Postgres."""

    dsn = _postgres_database_url()
    if not dsn:
        pytest.skip(
            "Postgres integration DSN not configured; set "
            "OPENCOUCH_TEST_POSTGRES_URL or OPENCOUCH_MEMORY_DATABASE_URL"
        )

    thread_id = f"voice-thread-{uuid4()}"

    try:
        assert await get_voice_finalization_status(thread_id, database_url=dsn) is None

        pending = await set_voice_finalization_status(
            thread_id,
            status="in_progress",
            detail="Saving session memory.",
            database_url=dsn,
        )
        assert pending.thread_id == thread_id
        assert pending.status == "in_progress"
        assert pending.detail == "Saving session memory."

        completed = await set_voice_finalization_status(
            thread_id,
            status="completed",
            detail="Session memory saved.",
            database_url=dsn,
        )
        stored = await get_voice_finalization_status(thread_id, database_url=dsn)

        assert stored is not None
        assert stored.thread_id == thread_id
        assert stored.status == "completed"
        assert stored.detail == "Session memory saved."
        assert stored.updated_at == completed.updated_at
    finally:
        await _delete_status(dsn, thread_id)
