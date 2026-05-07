"""PostgreSQL-backed LiveKit voice finalization status helpers."""

from __future__ import annotations

from psycopg.rows import dict_row
import psycopg

from agent.memory.hashing import iso_now
from agent.voice.finalization_status import (
    VOICE_FINALIZATION_STATUS_DDL,
    VoiceFinalizationState,
    VoiceFinalizationStatus,
)


async def _ensure_status_table(
    conn: psycopg.AsyncConnection[dict[str, object]],
) -> None:
    """Create the shared Postgres status table when it doesn't exist yet."""

    async with conn.transaction():
        async with conn.cursor() as cursor:
            await cursor.execute(VOICE_FINALIZATION_STATUS_DDL)


async def get_postgres_voice_finalization_status(
    thread_id: str,
    *,
    database_url: str,
) -> VoiceFinalizationStatus | None:
    """Return the current disconnect finalization status from Postgres.

    Args:
        thread_id (str): Thread identifier to query.
        database_url (str): PostgreSQL connection string.

    Returns:
        VoiceFinalizationStatus | None: Stored status when present, else ``None``.
    """

    async with await psycopg.AsyncConnection.connect(
        database_url,
        row_factory=dict_row,
        autocommit=True,
    ) as conn:
        await _ensure_status_table(conn)
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT status, detail, updated_at
                FROM opencouch_voice_finalization_status
                WHERE thread_id = %s
                """,
                (thread_id,),
            )
            row = await cursor.fetchone()

    if row is None:
        return None

    return VoiceFinalizationStatus(
        thread_id=thread_id,
        status=str(row["status"]),  # type: ignore[arg-type]
        detail=str(row["detail"]) if row["detail"] is not None else None,
        updated_at=str(row["updated_at"]),
    )


async def set_postgres_voice_finalization_status(
    thread_id: str,
    *,
    status: VoiceFinalizationState,
    detail: str | None = None,
    database_url: str,
) -> VoiceFinalizationStatus:
    """Upsert the disconnect finalization status in Postgres.

    Args:
        thread_id (str): Thread identifier to update.
        status (VoiceFinalizationState): Finalization status.
        detail (str | None): Optional detail string.
        database_url (str): PostgreSQL connection string.

    Returns:
        VoiceFinalizationStatus: The stored status object.
    """

    updated_at = iso_now()

    async with await psycopg.AsyncConnection.connect(
        database_url,
        row_factory=dict_row,
        autocommit=True,
    ) as conn:
        await _ensure_status_table(conn)
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO opencouch_voice_finalization_status(
                    thread_id,
                    status,
                    detail,
                    updated_at
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(thread_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    detail = EXCLUDED.detail,
                    updated_at = EXCLUDED.updated_at
                """,
                (thread_id, status, detail, updated_at),
            )

    return VoiceFinalizationStatus(
        thread_id=thread_id,
        status=status,
        detail=detail,
        updated_at=updated_at,
    )
