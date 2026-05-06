"""Shared LiveKit voice finalization status for worker/UI coordination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import aiosqlite

from agent.memory.hashing import iso_now
from agent.persistence import DEFAULT_MEMORY_DB_PATH
from config import get_settings

VoiceFinalizationState = Literal["in_progress", "completed", "failed"]

VOICE_FINALIZATION_STATUS_DDL = """
CREATE TABLE IF NOT EXISTS opencouch_voice_finalization_status (
    thread_id TEXT PRIMARY KEY,
    status TEXT NOT NULL
        CHECK (status IN ('in_progress', 'completed', 'failed')),
    detail TEXT,
    updated_at TEXT NOT NULL
);
"""


@dataclass(slots=True)
class VoiceFinalizationStatus:
    """Persisted disconnect-time memory finalization status for one thread."""

    thread_id: str
    status: VoiceFinalizationState
    detail: str | None
    updated_at: str


async def _ensure_status_table(conn: aiosqlite.Connection) -> None:
    """Create the shared SQLite status table when it doesn't exist yet."""

    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute(VOICE_FINALIZATION_STATUS_DDL)
    await conn.commit()


def _resolve_postgres_database_url(
    *,
    sqlite_path: str | Path | None,
    database_url: str | None,
) -> str | None:
    """Return the Postgres DSN to use for voice status persistence.

    Args:
        sqlite_path (str | Path | None): Legacy explicit SQLite path override.
            When set, this keeps voice finalization on the compatibility
            SQLite path instead of the recommended Postgres backend.
        database_url (str | None): Explicit PostgreSQL connection string.

    Returns:
        str | None: PostgreSQL DSN when Postgres should be used, else ``None``.

    Raises:
        ValueError: If Postgres is selected through settings but no DSN exists.
    """

    if database_url is not None:
        return database_url
    if sqlite_path is not None:
        return None

    settings = get_settings()
    if settings.persistence_backend != "postgres":
        return None
    if not settings.memory_database_url:
        raise ValueError(
            "OPENCOUCH_MEMORY_DATABASE_URL is required when "
            "OPENCOUCH_PERSISTENCE_BACKEND=postgres"
        )
    return settings.memory_database_url


async def _get_sqlite_voice_finalization_status(
    thread_id: str,
    *,
    sqlite_path: str | Path | None = None,
) -> VoiceFinalizationStatus | None:
    """Return the current disconnect finalization status from the legacy SQLite fallback."""

    db_path = Path(sqlite_path or DEFAULT_MEMORY_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as conn:
        await _ensure_status_table(conn)
        async with conn.execute(
            """
            SELECT status, detail, updated_at
            FROM opencouch_voice_finalization_status
            WHERE thread_id = ?
            """,
            (thread_id,),
        ) as cursor:
            row = await cursor.fetchone()

    if row is None:
        return None

    return VoiceFinalizationStatus(
        thread_id=thread_id,
        status=str(row[0]),  # type: ignore[arg-type]
        detail=str(row[1]) if row[1] is not None else None,
        updated_at=str(row[2]),
    )


async def _set_sqlite_voice_finalization_status(
    thread_id: str,
    *,
    status: VoiceFinalizationState,
    detail: str | None = None,
    sqlite_path: str | Path | None = None,
) -> VoiceFinalizationStatus:
    """Upsert the disconnect finalization status in the legacy SQLite fallback."""

    db_path = Path(sqlite_path or DEFAULT_MEMORY_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    updated_at = iso_now()

    async with aiosqlite.connect(db_path) as conn:
        await _ensure_status_table(conn)
        await conn.execute(
            """
            INSERT INTO opencouch_voice_finalization_status(
                thread_id,
                status,
                detail,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                status = excluded.status,
                detail = excluded.detail,
                updated_at = excluded.updated_at
            """,
            (thread_id, status, detail, updated_at),
        )
        await conn.commit()

    return VoiceFinalizationStatus(
        thread_id=thread_id,
        status=status,
        detail=detail,
        updated_at=updated_at,
    )


async def get_voice_finalization_status(
    thread_id: str,
    *,
    sqlite_path: str | Path | None = None,
    database_url: str | None = None,
) -> VoiceFinalizationStatus | None:
    """Return the current disconnect finalization status for one thread."""

    resolved_database_url = _resolve_postgres_database_url(
        sqlite_path=sqlite_path,
        database_url=database_url,
    )
    if resolved_database_url is not None:
        from agent.voice.postgres_finalization_status import (
            get_postgres_voice_finalization_status,
        )

        return await get_postgres_voice_finalization_status(
            thread_id,
            database_url=resolved_database_url,
        )

    return await _get_sqlite_voice_finalization_status(
        thread_id,
        sqlite_path=sqlite_path,
    )


async def set_voice_finalization_status(
    thread_id: str,
    *,
    status: VoiceFinalizationState,
    detail: str | None = None,
    sqlite_path: str | Path | None = None,
    database_url: str | None = None,
) -> VoiceFinalizationStatus:
    """Upsert the disconnect finalization status for one thread."""

    resolved_database_url = _resolve_postgres_database_url(
        sqlite_path=sqlite_path,
        database_url=database_url,
    )
    if resolved_database_url is not None:
        from agent.voice.postgres_finalization_status import (
            set_postgres_voice_finalization_status,
        )

        return await set_postgres_voice_finalization_status(
            thread_id,
            status=status,
            detail=detail,
            database_url=resolved_database_url,
        )

    return await _set_sqlite_voice_finalization_status(
        thread_id,
        status=status,
        detail=detail,
        sqlite_path=sqlite_path,
    )
