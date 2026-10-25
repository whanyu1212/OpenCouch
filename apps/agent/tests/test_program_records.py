"""Per-user program records, the same rules in memory and in Postgres (HY-334).

The Postgres runs need `OPENCOUCH_TEST_POSTGRES_URL`; they're skipped otherwise.
"""

import os
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta

import pytest
from psycopg_pool import AsyncConnectionPool

from opencouch_agent.program.postgres import PostgresRecordStore
from opencouch_agent.program.records import (
    InMemoryRecordStore,
    ProgramPosition,
    RecordStore,
    Worry,
    WorrySort,
)

POSTGRES_URL = os.environ.get("OPENCOUCH_TEST_POSTGRES_URL")
MORNING = datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
EVENING = MORNING + timedelta(hours=9)


@pytest.fixture(params=["memory", "postgres"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[RecordStore]:
    if request.param == "memory":
        yield InMemoryRecordStore()
        return
    if not POSTGRES_URL:
        pytest.skip("OPENCOUCH_TEST_POSTGRES_URL is not set")
    async with AsyncConnectionPool(POSTGRES_URL) as pool:
        postgres = PostgresRecordStore(pool)
        await postgres.create_schema()
        yield postgres


@pytest.fixture
async def alice(store: RecordStore) -> AsyncIterator[str]:
    # Unique per test, so Postgres runs don't see each other's rows, and
    # deleted afterwards so they don't pile up.
    user = f"alice-{uuid.uuid4()}"
    yield user
    await store.delete_user(user)


@pytest.fixture
async def bob(store: RecordStore) -> AsyncIterator[str]:
    user = f"bob-{uuid.uuid4()}"
    yield user
    await store.delete_user(user)


async def test_position_starts_empty_and_saves(store: RecordStore, alice: str) -> None:
    assert await store.position(alice) is None

    await store.save_position(alice, ProgramPosition(MORNING, 1, "lesson"))
    await store.save_position(alice, ProgramPosition(MORNING, 2, "worry_time"))

    assert await store.position(alice) == ProgramPosition(MORNING, 2, "worry_time")


async def test_a_new_worry_is_parked(store: RecordStore, alice: str) -> None:
    worry = await store.add_worry(
        alice, "what if I fail", source="quick_capture", captured_at=MORNING
    )

    assert (worry.status, worry.sort, worry.next_step) == ("parked", None, None)
    assert await store.worries(alice) == [worry]


async def test_worries_filter_by_status_and_time(
    store: RecordStore, alice: str
) -> None:
    early = await store.add_worry(alice, "early", source="chat", captured_at=MORNING)
    late = await store.add_worry(alice, "late", source="chat", captured_at=EVENING)
    await store.review_worry(alice, early.id, sort="hypothetical")

    assert [w.text for w in await store.worries(alice)] == ["early", "late"]
    assert await store.worries(alice, status="parked") == [late]
    assert await store.worries(alice, since=EVENING) == [late]


@pytest.mark.parametrize(
    ("sort", "next_step"),
    [("solvable", "email the landlord"), ("hypothetical", None), (None, None)],
    ids=["solvable", "hypothetical", "no longer matters"],
)
async def test_review_records_the_outcome(
    store: RecordStore, alice: str, sort: WorrySort | None, next_step: str | None
) -> None:
    worry = await store.add_worry(alice, "rent", source="chat", captured_at=MORNING)

    reviewed = await store.review_worry(
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
    assert await store.worries(alice) == [reviewed]


@pytest.mark.parametrize("known", [True, False], ids=["known", "unknown"])
async def test_only_a_solvable_worry_has_a_next_step(
    store: RecordStore, alice: str, known: bool
) -> None:
    worry = await store.add_worry(alice, "rent", source="chat", captured_at=MORNING)
    worry_id = worry.id if known else "no-such-worry"

    with pytest.raises(ValueError, match="solvable"):
        await store.review_worry(
            alice, worry_id, sort="hypothetical", next_step="do it"
        )


async def test_a_reviewed_worry_isnt_reviewed_again(
    store: RecordStore, alice: str
) -> None:
    worry = await store.add_worry(alice, "rent", source="chat", captured_at=MORNING)
    first = await store.review_worry(
        alice, worry.id, sort="solvable", next_step="email the landlord"
    )

    again = await store.review_worry(alice, worry.id, sort=None)

    assert again is None
    assert await store.worries(alice) == [first]


async def test_worries_caught_together_keep_their_order(
    store: RecordStore, alice: str
) -> None:
    for text in ("first", "second", "third"):
        await store.add_worry(alice, text, source="chat", captured_at=MORNING)

    assert [w.text for w in await store.worries(alice)] == ["first", "second", "third"]


async def test_since_needs_a_timezone(store: RecordStore, alice: str) -> None:
    with pytest.raises(ValueError, match="timezone"):
        await store.worries(alice, since=datetime(2026, 10, 25, 18))


async def test_another_users_worries_are_out_of_reach(
    store: RecordStore, alice: str, bob: str
) -> None:
    worry = await store.add_worry(alice, "mine", source="chat", captured_at=MORNING)

    assert await store.worries(bob) == []
    assert await store.review_worry(bob, worry.id, sort="hypothetical") is None
    assert not await store.delete_worry(bob, worry.id)
    assert await store.worries(alice) == [worry]


async def test_delete_removes_one_worry(store: RecordStore, alice: str) -> None:
    worry = await store.add_worry(alice, "gone", source="chat", captured_at=MORNING)

    assert await store.delete_worry(alice, worry.id)
    assert not await store.delete_worry(alice, worry.id)
    assert await store.worries(alice) == []


async def test_delete_user_removes_only_their_records(
    store: RecordStore, alice: str, bob: str
) -> None:
    for user in (alice, bob):
        await store.save_position(user, ProgramPosition(MORNING, 1, "lesson"))
        await store.add_worry(user, "a worry", source="chat", captured_at=MORNING)

    await store.delete_user(alice)

    assert await store.position(alice) is None
    assert await store.worries(alice) == []
    assert await store.position(bob) is not None
    assert len(await store.worries(bob)) == 1


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
