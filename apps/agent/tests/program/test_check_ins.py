"""Check-ins such as the weekly GAD-7, in memory and in Postgres (HY-334)."""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from opencouch_agent.program.models import CheckIn
from opencouch_agent.program.postgres import PostgresRecordStore
from opencouch_agent.program.records import RecordStore
from tests.program.conftest import POSTGRES_URL

WEEK_ONE = datetime(2026, 10, 25, 9, 0, tzinfo=UTC)
WEEK_TWO = WEEK_ONE + timedelta(days=7)
ANSWERS = (1, 2, 0, 3, 1, 1, 2)


async def test_saves_answers_with_their_total(records: RecordStore, alice: str) -> None:
    check_in = await records.add_check_in(alice, "gad7", ANSWERS, taken_at=WEEK_ONE)

    assert (check_in.answers, check_in.total) == (ANSWERS, 10)
    assert await records.check_ins(alice) == [check_in]


async def test_lists_oldest_first(records: RecordStore, alice: str) -> None:
    later = await records.add_check_in(alice, "gad7", ANSWERS, taken_at=WEEK_TWO)
    earlier = await records.add_check_in(alice, "gad7", ANSWERS, taken_at=WEEK_ONE)

    assert await records.check_ins(alice, questionnaire="gad7") == [earlier, later]


@pytest.mark.parametrize(
    "answers",
    [
        (1, 2, 0),
        (1, 2, 0, 3, 1, 1, 2, 0),
        (1, 2, 0, 4, 1, 1, 2),
        (1, 2, 0, -1, 1, 1, 2),
    ],
    ids=["too few", "too many", "above 3", "negative"],
)
async def test_refuses_answers_that_dont_fit(
    records: RecordStore, alice: str, answers: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="gad7"):
        await records.add_check_in(alice, "gad7", answers, taken_at=WEEK_ONE)

    assert await records.check_ins(alice) == []


async def test_another_users_check_ins_are_out_of_reach(
    records: RecordStore, alice: str, bob: str
) -> None:
    check_in = await records.add_check_in(alice, "gad7", ANSWERS, taken_at=WEEK_ONE)

    assert await records.check_ins(bob) == []
    assert not await records.delete_check_in(bob, check_in.id)
    assert await records.check_ins(alice) == [check_in]


async def test_delete_removes_one_check_in(records: RecordStore, alice: str) -> None:
    check_in = await records.add_check_in(alice, "gad7", ANSWERS, taken_at=WEEK_ONE)

    assert await records.delete_check_in(alice, check_in.id)
    assert await records.check_ins(alice) == []


def test_total_must_match_the_answers() -> None:
    with pytest.raises(ValueError, match="sum"):
        CheckIn("id", "gad7", ANSWERS, 11, WEEK_ONE)


async def test_refuses_true_and_false_as_answers(
    records: RecordStore, alice: str
) -> None:
    with pytest.raises(ValueError, match="whole numbers"):
        await records.add_check_in(alice, "gad7", (True,) * 7, taken_at=WEEK_ONE)


async def test_answers_given_as_a_list_read_back_equal(
    records: RecordStore, alice: str
) -> None:
    # `Any`: a list breaks the declared tuple type, as a loose caller might.
    answers: Any = list(ANSWERS)

    check_in = await records.add_check_in(alice, "gad7", answers, taken_at=WEEK_ONE)

    assert check_in.answers == ANSWERS
    assert await records.check_ins(alice) == [check_in]


@pytest.mark.parametrize(
    "answers",
    ["{9}", "{1,1,1,1,1,1,4}", "{1,1,1,1,1,1,NULL}"],
    ids=["one", "4", "null"],
)
async def test_postgres_refuses_a_bad_gad7_row(answers: str) -> None:
    if not POSTGRES_URL:
        pytest.skip("OPENCOUCH_TEST_POSTGRES_URL is not set")
    async with AsyncConnectionPool(POSTGRES_URL) as pool:
        await PostgresRecordStore(pool).create_schema()
        async with pool.connection() as connection:
            with pytest.raises(psycopg.errors.CheckViolation):
                await connection.execute(
                    "INSERT INTO check_ins"
                    " (id, user_id, questionnaire, answers, taken_at)"
                    " VALUES (%s, %s, 'gad7', %s::integer[], now())",
                    (str(uuid.uuid4()), f"carol-{uuid.uuid4()}", answers),
                )
