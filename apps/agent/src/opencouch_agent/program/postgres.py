"""Postgres-backed per-user program records (HY-334)."""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from opencouch_agent.program.models import (
    CheckIn,
    FormulationEntry,
    FormulationKind,
    ProgramPosition,
    Questionnaire,
    Worry,
    WorrySort,
    WorrySource,
    WorryStatus,
    check_entry_text,
    check_review,
    confirmed_entry,
    parked_worry,
    require_timezone,
    scored_check_in,
)
from opencouch_agent.program.postgres_schema import (
    CHECK_IN_COLUMNS,
    ENTRY_COLUMNS,
    SCHEMA,
    WORRY_COLUMNS,
    check_in_from,
    entry_from,
    position_from,
    worry_from,
)

_SAVE_POSITION = """
INSERT INTO program_positions (user_id, enrolled_at, week, step)
VALUES (%s, %s, %s, %s)
ON CONFLICT (user_id) DO UPDATE
SET enrolled_at = EXCLUDED.enrolled_at,
    week = EXCLUDED.week,
    step = EXCLUDED.step,
    updated_at = now()
"""

# Psycopg takes either positional or named parameters.
_Params = Sequence[object] | Mapping[str, object]


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
        rows = await self._rows(
            "SELECT enrolled_at, week, step FROM program_positions WHERE user_id = %s",
            (user_id,),
        )
        return position_from(rows[0]) if rows else None

    async def save_position(self, user_id: str, position: ProgramPosition) -> None:
        """Replace the user's place; see `RecordStore.save_position`."""
        await self._run(
            _SAVE_POSITION,
            (user_id, position.enrolled_at, position.week, position.step),
        )

    async def add_worry(
        self, user_id: str, text: str, *, source: WorrySource, captured_at: datetime
    ) -> Worry:
        """Park a new worry; see `RecordStore.add_worry`."""
        worry = parked_worry(text, source=source, captured_at=captured_at)
        await self._run(
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
        rows = await self._rows(
            f"SELECT {WORRY_COLUMNS} FROM worries"
            " WHERE user_id = %(user_id)s"
            " AND (%(status)s::text IS NULL OR status = %(status)s)"
            " AND (%(since)s::timestamptz IS NULL OR captured_at >= %(since)s)"
            " ORDER BY captured_at, seq",
            {"user_id": user_id, "status": status, "since": since},
        )
        return [worry_from(row) for row in rows]

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
        rows = await self._rows(
            "UPDATE worries SET status = 'reviewed', sort = %s, next_step = %s"
            " WHERE id = %s AND user_id = %s AND status = 'parked'"
            f" RETURNING {WORRY_COLUMNS}",
            (sort, next_step, worry_id, user_id),
        )
        return worry_from(rows[0]) if rows else None

    async def delete_worry(self, user_id: str, worry_id: str) -> bool:
        """Delete a worry; see `RecordStore.delete_worry`."""
        return await self._delete("worries", user_id, worry_id)

    async def add_check_in(
        self,
        user_id: str,
        questionnaire: Questionnaire,
        answers: tuple[int, ...],
        *,
        taken_at: datetime,
    ) -> CheckIn:
        """Save a check-in; see `RecordStore.add_check_in`."""
        check_in = scored_check_in(questionnaire, answers, taken_at=taken_at)
        await self._run(
            "INSERT INTO check_ins (id, user_id, questionnaire, answers, taken_at)"
            " VALUES (%s, %s, %s, %s, %s)",
            (check_in.id, user_id, questionnaire, list(check_in.answers), taken_at),
        )
        return check_in

    async def check_ins(
        self, user_id: str, *, questionnaire: Questionnaire | None = None
    ) -> list[CheckIn]:
        """The user's check-ins; see `RecordStore.check_ins`."""
        rows = await self._rows(
            f"SELECT {CHECK_IN_COLUMNS} FROM check_ins"
            " WHERE user_id = %(user_id)s"
            " AND (%(questionnaire)s::text IS NULL"
            " OR questionnaire = %(questionnaire)s)"
            " ORDER BY taken_at, seq",
            {"user_id": user_id, "questionnaire": questionnaire},
        )
        return [check_in_from(row) for row in rows]

    async def delete_check_in(self, user_id: str, check_in_id: str) -> bool:
        """Delete a check-in; see `RecordStore.delete_check_in`."""
        return await self._delete("check_ins", user_id, check_in_id)

    async def add_formulation_entry(
        self, user_id: str, kind: FormulationKind, text: str, *, confirmed_at: datetime
    ) -> FormulationEntry:
        """Save an entry; see `RecordStore.add_formulation_entry`."""
        entry = confirmed_entry(kind, text, confirmed_at=confirmed_at)
        await self._run(
            "INSERT INTO formulation_entries (id, user_id, kind, text, confirmed_at)"
            " VALUES (%s, %s, %s, %s, %s)",
            (entry.id, user_id, kind, text, confirmed_at),
        )
        return entry

    async def formulation(self, user_id: str) -> list[FormulationEntry]:
        """The user's worry map; see `RecordStore.formulation`."""
        rows = await self._rows(
            f"SELECT {ENTRY_COLUMNS} FROM formulation_entries"
            " WHERE user_id = %s ORDER BY confirmed_at, seq",
            (user_id,),
        )
        return [entry_from(row) for row in rows]

    async def edit_formulation_entry(
        self, user_id: str, entry_id: str, text: str
    ) -> FormulationEntry | None:
        """Reword an entry; see `RecordStore.edit_formulation_entry`."""
        check_entry_text(text)
        rows = await self._rows(
            "UPDATE formulation_entries SET text = %s, updated_at = now()"
            f" WHERE id = %s AND user_id = %s RETURNING {ENTRY_COLUMNS}",
            (text, entry_id, user_id),
        )
        return entry_from(rows[0]) if rows else None

    async def delete_formulation_entry(self, user_id: str, entry_id: str) -> bool:
        """Delete an entry; see `RecordStore.delete_formulation_entry`."""
        return await self._delete("formulation_entries", user_id, entry_id)

    async def delete_user(self, user_id: str) -> None:
        """Delete everything; see `RecordStore.delete_user`."""
        async with self._pool.connection() as connection, connection.transaction():
            for table in (
                "worries",
                "check_ins",
                "formulation_entries",
                "program_positions",
            ):
                # The table names are fixed above, never from input.
                await connection.execute(
                    f"DELETE FROM {table} WHERE user_id = %s",
                    (user_id,),
                )

    async def _rows(self, query: str, params: _Params) -> list[dict[str, Any]]:
        # `Any`: the column values are checked by each record's row model.
        async with self._pool.connection() as connection:
            cursor = connection.cursor(row_factory=dict_row)
            await cursor.execute(query, params)
            return await cursor.fetchall()

    async def _run(self, query: str, params: _Params) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(query, params)

    async def _delete(self, table: str, user_id: str, record_id: str) -> bool:
        # Only ever called with this module's table names, never from input.
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                f"DELETE FROM {table} WHERE id = %s AND user_id = %s",
                (record_id, user_id),
            )
            return cursor.rowcount > 0
