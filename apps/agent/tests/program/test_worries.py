"""Program position and worries, the same rules in memory and in Postgres (HY-334)."""

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from opencouch_agent.program.models import ProgramPosition, Worry, WorrySort
from opencouch_agent.program.postgres import PostgresRecordStore
from opencouch_agent.program.records import RecordStore
from tests.program.conftest import POSTGRES_URL

MORNING = datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
EVENING = MORNING + timedelta(hours=9)


async def test_position_starts_empty_and_saves(
    records: RecordStore, alice: str
) -> None:
    assert await records.position(alice) is None

    await records.save_position(alice, ProgramPosition(MORNING, 1, "lesson"))
    await records.save_position(alice, ProgramPosition(MORNING, 2, "worry_time"))

    assert await records.position(alice) == ProgramPosition(MORNING, 2, "worry_time")


async def test_a_new_worry_is_parked(records: RecordStore, alice: str) -> None:
    worry = await records.add_worry(
        alice, "what if I fail", source="quick_capture", captured_at=MORNING
    )

    assert (worry.status, worry.sort, worry.next_step) == ("parked", None, None)
    assert await records.worries(alice) == [worry]


async def test_worries_filter_by_status_and_time(
    records: RecordStore, alice: str
) -> None:
    early = await records.add_worry(alice, "early", source="chat", captured_at=MORNING)
    late = await records.add_worry(alice, "late", source="chat", captured_at=EVENING)
    await records.review_worry(alice, early.id, sort="hypothetical")

    assert [w.text for w in await records.worries(alice)] == ["early", "late"]
    assert await records.worries(alice, status="parked") == [late]
    assert await records.worries(alice, since=EVENING) == [late]


@pytest.mark.parametrize(
    ("sort", "next_step"),
    [("solvable", "email the landlord"), ("hypothetical", None), (None, None)],
    ids=["solvable", "hypothetical", "no longer matters"],
)
async def test_review_records_the_outcome(
    records: RecordStore, alice: str, sort: WorrySort | None, next_step: str | None
) -> None:
    worry = await records.add_worry(alice, "rent", source="chat", captured_at=MORNING)

    reviewed = await records.review_worry(
        alice,
        worry.id,
        sort=sort,
        next_step=next_step,
    )

    assert reviewed is not None
    assert (reviewed.status, reviewed.sort, reviewed.next_step) == (
        "reviewed",
        sort,
        next_step,
    )
    assert await records.worries(alice) == [reviewed]


@pytest.mark.parametrize("known", [True, False], ids=["known", "unknown"])
async def test_only_a_solvable_worry_has_a_next_step(
    records: RecordStore, alice: str, known: bool
) -> None:
    worry = await records.add_worry(alice, "rent", source="chat", captured_at=MORNING)
    worry_id = worry.id if known else "no-such-worry"

    with pytest.raises(ValueError, match="solvable"):
        await records.review_worry(
            alice, worry_id, sort="hypothetical", next_step="do it"
        )


async def test_a_reviewed_worry_isnt_reviewed_again(
    records: RecordStore, alice: str
) -> None:
    worry = await records.add_worry(alice, "rent", source="chat", captured_at=MORNING)
    first = await records.review_worry(
        alice, worry.id, sort="solvable", next_step="email the landlord"
    )

    again = await records.review_worry(alice, worry.id, sort=None)

    assert again is None
    assert await records.worries(alice) == [first]


async def test_worries_caught_together_keep_their_order(
    records: RecordStore, alice: str
) -> None:
    for text in ("first", "second", "third"):
        await records.add_worry(alice, text, source="chat", captured_at=MORNING)

    assert [w.text for w in await records.worries(alice)] == [
        "first",
        "second",
        "third",
    ]


async def test_since_needs_a_timezone(records: RecordStore, alice: str) -> None:
    with pytest.raises(ValueError, match="timezone"):
        await records.worries(alice, since=datetime(2026, 10, 25, 18))


async def test_another_users_worries_are_out_of_reach(
    records: RecordStore, alice: str, bob: str
) -> None:
    worry = await records.add_worry(alice, "mine", source="chat", captured_at=MORNING)

    assert await records.worries(bob) == []
    assert await records.review_worry(bob, worry.id, sort="hypothetical") is None
    assert not await records.delete_worry(bob, worry.id)
    assert await records.worries(alice) == [worry]


async def test_delete_removes_one_worry(records: RecordStore, alice: str) -> None:
    worry = await records.add_worry(alice, "gone", source="chat", captured_at=MORNING)

    assert await records.delete_worry(alice, worry.id)
    assert not await records.delete_worry(alice, worry.id)
    assert await records.worries(alice) == []


@pytest.mark.parametrize(
    "build",
    [
        lambda: ProgramPosition(MORNING, 0, "lesson"),
        lambda: ProgramPosition(datetime(2026, 10, 25), 1, "lesson"),
        lambda: Worry("id", "w", MORNING, "chat", status="parked", sort="solvable"),
    ],
    ids=["week zero", "naive time", "parked but sorted"],
)
def test_records_reject_impossible_values(build: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        build()


async def test_postgres_refuses_a_next_step_without_a_sort() -> None:
    if not POSTGRES_URL:
        pytest.skip("OPENCOUCH_TEST_POSTGRES_URL is not set")
    async with AsyncConnectionPool(POSTGRES_URL) as pool:
        await PostgresRecordStore(pool).create_schema()
        async with pool.connection() as connection:
            with pytest.raises(psycopg.errors.CheckViolation):
                await connection.execute(
                    "INSERT INTO worries"
                    " (id, user_id, text, captured_at, source, status, next_step)"
                    " VALUES (%s, %s, 'w', now(), 'chat', 'reviewed', 'a step')",
                    (str(uuid.uuid4()), f"carol-{uuid.uuid4()}"),
                )
