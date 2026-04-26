"""Shared LiveKit voice finalization status for worker/UI coordination."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import aiosqlite

from agent.memory.hashing import iso_now
from agent.persistence import DEFAULT_MEMORY_DB_PATH

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
    """Create the shared status table when it doesn't exist yet."""

    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute(VOICE_FINALIZATION_STATUS_DDL)
    await conn.commit()


async def get_voice_finalization_status(
    thread_id: str,
    *,
    sqlite_path: str | Path | None = None,
) -> VoiceFinalizationStatus | None:
    """Return the current disconnect finalization status for one thread."""

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


async def set_voice_finalization_status(
    thread_id: str,
    *,
    status: VoiceFinalizationState,
    detail: str | None = None,
    sqlite_path: str | Path | None = None,
) -> VoiceFinalizationStatus:
    """Upsert the disconnect finalization status for one thread."""

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
