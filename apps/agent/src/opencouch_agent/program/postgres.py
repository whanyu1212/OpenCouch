"""Postgres-backed per-user program records (HY-334)."""

from datetime import datetime

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import AwareDatetime, BaseModel

from opencouch_agent.program.records import (
    ProgramPosition,
    Worry,
    WorrySort,
    WorrySource,
    WorryStatus,
    check_review,
    parked_worry,
    require_timezone,
)

# No foreign keys to conversations: records belong to the user, not a thread.
# The CHECKs repeat the dataclasses' rules, so a bad row can't be written even
# by another client.
SCHEMA = """
CREATE TABLE IF NOT EXISTS program_positions (
    user_id     TEXT PRIMARY KEY,
    enrolled_at TIMESTAMPTZ NOT NULL,
    week        INTEGER NOT NULL CHECK (week >= 1),
    step        TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS worries (
    -- Insertion order, to break ties between worries caught the same moment.
    seq         BIGSERIAL,
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    source      TEXT NOT NULL CHECK (source IN ('quick_capture', 'chat')),
    status      TEXT NOT NULL CHECK (status IN ('parked', 'reviewed')),
    sort        TEXT CHECK (sort IN ('solvable', 'hypothetical')),
    next_step   TEXT,
    CHECK (status = 'reviewed' OR sort IS NULL),
    -- `sort IS NOT NULL` too: a NULL sort would make the comparison unknown,
    -- which a CHECK lets through.
    CHECK (next_step IS NULL OR (sort IS NOT NULL AND sort = 'solvable'))
);

CREATE INDEX IF NOT EXISTS worries_user_id_captured_at_idx
    ON worries (user_id, captured_at);
"""

_SAVE_POSITION = """
INSERT INTO program_positions (user_id, enrolled_at, week, step)
VALUES (%s, %s, %s, %s)
ON CONFLICT (user_id) DO UPDATE
SET enrolled_at = EXCLUDED.enrolled_at,
    week = EXCLUDED.week,
    step = EXCLUDED.step,
    updated_at = now()
"""

_WORRY_COLUMNS = "id, text, captured_at, source, status, sort, next_step"


class PostgresRecordStore:
    """Program records in Postgres, every query scoped to one user."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create_schema(self) -> None:
        """Create the record tables if they don't exist."""
        async with self._pool.connection() as connection:
            await connection.execute(SCHEMA)

    async def position(self, user_id: str) -> ProgramPosition | None:
        """The user's place in the program; see `RecordStore.position`."""
        async with self._pool.connection() as connection:
            cursor = connection.cursor(row_factory=dict_row)
            await cursor.execute(
                "SELECT enrolled_at, week, step FROM program_positions"
                " WHERE user_id = %s",
                (user_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        checked = _PositionRow.model_validate(row)
        return ProgramPosition(**checked.model_dump())

    async def save_position(self, user_id: str, position: ProgramPosition) -> None:
        """Replace the user's place; see `RecordStore.save_position`."""
        async with self._pool.connection() as connection:
            await connection.execute(
                _SAVE_POSITION,
                (user_id, position.enrolled_at, position.week, position.step),
            )

    async def add_worry(
        self, user_id: str, text: str, *, source: WorrySource, captured_at: datetime
    ) -> Worry:
        """Park a new worry; see `RecordStore.add_worry`."""
        worry = parked_worry(text, source=source, captured_at=captured_at)
        async with self._pool.connection() as connection:
            await connection.execute(
                "INSERT INTO worries (id, user_id, text, captured_at, source, status)"
                " VALUES (%s, %s, %s, %s, %s, %s)",
                (worry.id, user_id, text, captured_at, source, worry.status),
            )
        return worry

    async def worries(
        self,
        user_id: str,
        *,
        status: WorryStatus | None = None,
        since: datetime | None = None,
    ) -> list[Worry]:
        """The user's worries; see `RecordStore.worries`."""
        if since is not None:
            require_timezone(since)
        async with self._pool.connection() as connection:
            cursor = connection.cursor(row_factory=dict_row)
            await cursor.execute(
                f"SELECT {_WORRY_COLUMNS} FROM worries"
                " WHERE user_id = %(user_id)s"
                " AND (%(status)s::text IS NULL OR status = %(status)s)"
                " AND (%(since)s::timestamptz IS NULL OR captured_at >= %(since)s)"
                " ORDER BY captured_at, seq",
                {"user_id": user_id, "status": status, "since": since},
            )
            rows = await cursor.fetchall()
        return [_worry(row) for row in rows]

    async def review_worry(
        self,
        user_id: str,
        worry_id: str,
        *,
        sort: WorrySort | None,
        next_step: str | None = None,
    ) -> Worry | None:
        """Mark a worry reviewed; see `RecordStore.review_worry`."""
        check_review(sort, next_step)
        async with self._pool.connection() as connection:
            cursor = connection.cursor(row_factory=dict_row)
            await cursor.execute(
                "UPDATE worries SET status = 'reviewed', sort = %s, next_step = %s"
                f" WHERE id = %s AND user_id = %s AND status = 'parked'"
                f" RETURNING {_WORRY_COLUMNS}",
                (sort, next_step, worry_id, user_id),
            )
            row = await cursor.fetchone()
        return None if row is None else _worry(row)

    async def delete_worry(self, user_id: str, worry_id: str) -> bool:
        """Delete a worry; see `RecordStore.delete_worry`."""
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM worries WHERE id = %s AND user_id = %s",
                (worry_id, user_id),
            )
            return cursor.rowcount > 0

    async def delete_user(self, user_id: str) -> None:
        """Delete everything; see `RecordStore.delete_user`."""
        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute(
                "DELETE FROM worries WHERE user_id = %s", (user_id,)
            )
            await connection.execute(
                "DELETE FROM program_positions WHERE user_id = %s", (user_id,)
            )


class _PositionRow(BaseModel):
    """A `program_positions` row, checked before it becomes a `ProgramPosition`."""

    enrolled_at: AwareDatetime
    week: int
    step: str


class _WorryRow(BaseModel):
    """A `worries` row, checked before it becomes a `Worry`."""

    id: str
    text: str
    captured_at: AwareDatetime
    source: WorrySource
    status: WorryStatus
    sort: WorrySort | None
    next_step: str | None


def _worry(row: object) -> Worry:
    checked = _WorryRow.model_validate(row)
    return Worry(**checked.model_dump())
