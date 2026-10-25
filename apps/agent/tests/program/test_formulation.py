"""The worry map: confirmed themes and safety behaviors (HY-334)."""

import uuid
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from opencouch_agent.program.models import ProgramPosition
from opencouch_agent.program.postgres import PostgresRecordStore
from opencouch_agent.program.records import RecordStore
from tests.program.conftest import POSTGRES_URL

MONDAY = datetime(2026, 10, 26, 18, 0, tzinfo=UTC)
TUESDAY = MONDAY + timedelta(days=1)


async def test_saves_entries_oldest_first(records: RecordStore, alice: str) -> None:
    later = await records.add_formulation_entry(
        alice, "safety_behavior", "checking my bank app", confirmed_at=TUESDAY
    )
    earlier = await records.add_formulation_entry(
        alice, "worry_theme", "money", confirmed_at=MONDAY
    )

    assert await records.formulation(alice) == [earlier, later]


async def test_edit_rewords_an_entry(records: RecordStore, alice: str) -> None:
    entry = await records.add_formulation_entry(
        alice, "worry_theme", "money", confirmed_at=MONDAY
    )

    edited = await records.edit_formulation_entry(alice, entry.id, "money and rent")

    assert edited is not None
    assert (edited.text, edited.kind, edited.confirmed_at) == (
        "money and rent",
        "worry_theme",
        MONDAY,
    )
    assert await records.formulation(alice) == [edited]


@pytest.mark.parametrize("known", [True, False], ids=["known", "unknown"])
async def test_blank_text_is_refused(
    records: RecordStore, alice: str, known: bool
) -> None:
    entry = await records.add_formulation_entry(
        alice, "worry_theme", "money", confirmed_at=MONDAY
    )

    with pytest.raises(ValueError, match="text"):
        await records.edit_formulation_entry(
            alice, entry.id if known else "no-such-entry", "   "
        )
    with pytest.raises(ValueError, match="text"):
        await records.add_formulation_entry(
            alice, "worry_theme", "", confirmed_at=MONDAY
        )


async def test_another_users_entries_are_out_of_reach(
    records: RecordStore, alice: str, bob: str
) -> None:
    entry = await records.add_formulation_entry(
        alice, "worry_theme", "health", confirmed_at=MONDAY
    )

    assert await records.formulation(bob) == []
    assert await records.edit_formulation_entry(bob, entry.id, "mine now") is None
    assert not await records.delete_formulation_entry(bob, entry.id)
    assert await records.formulation(alice) == [entry]


async def test_delete_removes_one_entry(records: RecordStore, alice: str) -> None:
    entry = await records.add_formulation_entry(
        alice, "worry_theme", "health", confirmed_at=MONDAY
    )

    assert await records.delete_formulation_entry(alice, entry.id)
    assert await records.formulation(alice) == []


async def test_delete_user_removes_every_kind_of_record(
    records: RecordStore, alice: str, bob: str
) -> None:
    for user in (alice, bob):
        await records.save_position(user, ProgramPosition(MONDAY, 1, "lesson"))
        await records.add_worry(user, "a worry", source="chat", captured_at=MONDAY)
        await records.add_check_in(user, "gad7", (0,) * 7, taken_at=MONDAY)
        await records.add_formulation_entry(
            user, "worry_theme", "money", confirmed_at=MONDAY
        )

    await records.delete_user(alice)

    assert await records.position(alice) is None
    assert await records.worries(alice) == []
    assert await records.check_ins(alice) == []
    assert await records.formulation(alice) == []
    assert len(await records.formulation(bob)) == 1
    assert len(await records.check_ins(bob)) == 1


@pytest.mark.parametrize(
    "text", ["", "   ", "\t", "\n"], ids=["empty", "spaces", "tab", "newline"]
)
async def test_postgres_refuses_blank_entry_text(text: str) -> None:
    if not POSTGRES_URL:
        pytest.skip("OPENCOUCH_TEST_POSTGRES_URL is not set")
    async with AsyncConnectionPool(POSTGRES_URL) as pool:
        await PostgresRecordStore(pool).create_schema()
        async with pool.connection() as connection:
            with pytest.raises(psycopg.errors.CheckViolation):
                await connection.execute(
                    "INSERT INTO formulation_entries"
                    " (id, user_id, kind, text, confirmed_at)"
                    " VALUES (%s, %s, 'worry_theme', %s, now())",
                    (str(uuid.uuid4()), f"carol-{uuid.uuid4()}", text),
                )
