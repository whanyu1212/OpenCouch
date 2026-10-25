"""Integration tests for the Postgres session store.

Set `OPENCOUCH_TEST_POSTGRES_URL` to run them; they are skipped otherwise.
"""

import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from opencouch_agent.beta.postgres import PostgresAccessRequests
from opencouch_agent.safety import RiskLevel
from opencouch_agent.safety.risk import RiskChange, RiskState
from opencouch_agent.sessions.postgres import PostgresSessionStore

POSTGRES_URL = os.environ.get("OPENCOUCH_TEST_POSTGRES_URL")
LEASE = timedelta(seconds=60)
IN_CRISIS = RiskState(
    level=RiskLevel.CRISIS,
    peak=RiskLevel.CRISIS,
    score=0.7,
    turns=1,
    last_risk_at=datetime(2026, 10, 23, 9, 0, tzinfo=UTC),
    last_turn_at=datetime(2026, 10, 23, 9, 0, tzinfo=UTC),
    changes=(RiskChange(1, RiskLevel.NONE, RiskLevel.CRISIS, "CRISIS: intent"),),
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="OPENCOUCH_TEST_POSTGRES_URL is not set"
)


@pytest.fixture
async def store() -> AsyncIterator[PostgresSessionStore]:
    assert POSTGRES_URL
    async with AsyncConnectionPool(POSTGRES_URL) as pool:
        postgres_store = PostgresSessionStore(pool)
        await postgres_store.create_schema()
        yield postgres_store


def _thread_id() -> str:
    return f"test-{uuid.uuid4()}"


async def _record(
    store: PostgresSessionStore,
    thread_id: str,
    messages: list[ModelMessage],
    risk: RiskState | None = None,
) -> None:
    await store.claim_turn(thread_id, "token", LEASE)
    assert await store.record_user_turn(
        thread_id, "token", messages, risk or RiskState()
    )
    await store.release_turn(thread_id, "token")


async def test_unknown_thread_loads_empty(store: PostgresSessionStore) -> None:
    session = await store.load(_thread_id())

    assert session.history == []
    assert session.state == {}
    assert session.risk == RiskState()


async def test_messages_round_trip_in_order(store: PostgresSessionStore) -> None:
    thread_id = _thread_id()
    first_turn: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(parts=[TextPart("reply one")]),
    ]
    second_turn: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("second")])]

    await _record(store, thread_id, first_turn)
    await _record(store, thread_id, second_turn)
    session = await store.load(thread_id)

    assert session.history == [*first_turn, *second_turn]


async def test_save_run_replaces_state(store: PostgresSessionStore) -> None:
    thread_id = _thread_id()
    await store.claim_turn(thread_id, "token", timedelta(seconds=60))

    await store.save_run(thread_id, "token", [], {"mode": "chat"})
    await store.save_run(thread_id, "token", [], {"mode": "exercise"})

    assert (await store.load(thread_id)).state == {"mode": "exercise"}


async def test_save_run_appends_after_user_message(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()
    user_turn: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("hi")])]
    reply: list[ModelMessage] = [ModelResponse(parts=[TextPart("hello")])]
    await store.claim_turn(thread_id, "token", LEASE)

    assert await store.record_user_turn(thread_id, "token", user_turn, RiskState())
    await store.save_run(thread_id, "token", reply, {"mode": "chat"})
    session = await store.load(thread_id)

    assert session.history == [*user_turn, *reply]
    assert session.state == {"mode": "chat"}


async def test_save_run_writes_nothing_after_the_lease_was_taken_over(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()
    await store.claim_turn(thread_id, "slow", timedelta(seconds=-1))
    await store.claim_turn(thread_id, "newer", timedelta(seconds=60))
    await store.save_run(thread_id, "newer", [], {"mode": "newer"})

    saved = await store.save_run(
        thread_id,
        "slow",
        [ModelResponse(parts=[TextPart("late reply")])],
        {"mode": "stale"},
    )

    session = await store.load(thread_id)
    assert not saved
    assert session.history == []
    assert session.state == {"mode": "newer"}


async def test_create_schema_adds_lease_columns_to_an_older_table(
    store: PostgresSessionStore,
) -> None:
    # Recreate the table as the first release created it, without lease columns.
    async with store._pool.connection() as connection:
        await connection.execute(
            "ALTER TABLE conversations"
            " DROP COLUMN active_turn_token, DROP COLUMN turn_lease_expires_at"
        )

    await store.create_schema()

    thread_id = _thread_id()
    assert await store.claim_turn(thread_id, "token", timedelta(seconds=60))


async def test_create_schema_is_idempotent(store: PostgresSessionStore) -> None:
    await store.create_schema()


async def test_claim_fails_while_another_turn_holds_the_thread(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()

    assert await store.claim_turn(thread_id, "first", timedelta(seconds=60))
    assert not await store.claim_turn(thread_id, "second", timedelta(seconds=60))


async def test_expired_lease_can_be_reclaimed(store: PostgresSessionStore) -> None:
    thread_id = _thread_id()
    await store.claim_turn(thread_id, "crashed", timedelta(seconds=-1))

    assert await store.claim_turn(thread_id, "next", timedelta(seconds=60))


async def test_release_with_a_stale_token_keeps_the_newer_claim(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()
    await store.claim_turn(thread_id, "old", timedelta(seconds=-1))
    await store.claim_turn(thread_id, "new", timedelta(seconds=60))

    await store.release_turn(thread_id, "old")

    assert not await store.claim_turn(thread_id, "third", timedelta(seconds=60))


async def test_release_frees_the_thread(store: PostgresSessionStore) -> None:
    thread_id = _thread_id()
    await store.claim_turn(thread_id, "first", timedelta(seconds=60))

    await store.release_turn(thread_id, "first")

    assert await store.claim_turn(thread_id, "second", timedelta(seconds=60))


async def test_a_turn_is_in_progress_only_while_its_lease_holds(
    store: PostgresSessionStore,
) -> None:
    held, expired = _thread_id(), _thread_id()
    await store.claim_turn(held, "running", timedelta(seconds=60))
    await store.claim_turn(expired, "crashed", timedelta(seconds=-1))

    assert await store.turn_in_progress(held)
    assert not await store.turn_in_progress(expired)
    await store.release_turn(held, "running")
    assert not await store.turn_in_progress(held)
    assert not await store.turn_in_progress(_thread_id())


async def test_claiming_keeps_existing_history(store: PostgresSessionStore) -> None:
    thread_id = _thread_id()
    await _record(store, thread_id, [ModelRequest(parts=[UserPromptPart("hi")])])

    await store.claim_turn(thread_id, "token", timedelta(seconds=60))

    assert len((await store.load(thread_id)).history) == 1


async def test_recording_a_user_turn_saves_the_risk(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()

    await _record(store, thread_id, [], IN_CRISIS)

    assert (await store.load(thread_id)).risk == IN_CRISIS


async def test_recording_a_user_turn_writes_nothing_after_losing_the_lease(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()
    await store.claim_turn(thread_id, "slow", timedelta(seconds=-1))
    await store.claim_turn(thread_id, "newer", LEASE)

    recorded = await store.record_user_turn(
        thread_id, "slow", [ModelRequest(parts=[UserPromptPart("late")])], IN_CRISIS
    )

    session = await store.load(thread_id)
    assert not recorded
    assert session.history == []
    assert session.risk == RiskState()


async def test_a_claimed_thread_with_no_screened_turn_has_no_risk(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()

    await store.claim_turn(thread_id, "token", LEASE)

    assert (await store.load(thread_id)).risk == RiskState()


async def test_an_unreadable_risk_record_fails_the_load(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()
    await _record(store, thread_id, [], IN_CRISIS)
    async with store._pool.connection() as connection:
        await connection.execute(
            "UPDATE conversations SET risk = %s WHERE thread_id = %s",
            (Jsonb({"version": 999, "state": {}}), thread_id),
        )

    with pytest.raises(ValidationError):
        await store.load(thread_id)


async def test_create_schema_adds_the_risk_column_to_an_older_table(
    store: PostgresSessionStore,
) -> None:
    async with store._pool.connection() as connection:
        await connection.execute("ALTER TABLE conversations DROP COLUMN risk")

    await store.create_schema()

    thread_id = _thread_id()
    await _record(store, thread_id, [], IN_CRISIS)
    assert (await store.load(thread_id)).risk == IN_CRISIS


async def test_the_first_claim_makes_a_thread_its_owners(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()

    assert await store.claim_thread(thread_id, "alice")
    assert await store.claim_thread(thread_id, "alice")
    assert not await store.claim_thread(thread_id, "bob")


async def test_only_the_owner_can_read_a_thread(store: PostgresSessionStore) -> None:
    thread_id = _thread_id()
    await store.claim_thread(thread_id, "alice")

    assert await store.can_read(thread_id, "alice")
    assert not await store.can_read(thread_id, "bob")
    assert await store.can_read(_thread_id(), "bob")


async def test_a_thread_from_before_sign_in_belongs_to_nobody(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()
    # Threads saved before HY-221 have no owner.
    await store.claim_turn(thread_id, "token", LEASE)

    assert not await store.claim_thread(thread_id, "alice")
    assert not await store.can_read(thread_id, "alice")


async def test_access_requests_round_trip() -> None:
    assert POSTGRES_URL
    email = f"{uuid.uuid4()}@example.com"
    async with AsyncConnectionPool(POSTGRES_URL) as pool:
        requests = PostgresAccessRequests(pool)
        await requests.create_schema()

        await requests.add(email)
        await requests.add(email)
        listed = [address for address, _ in await requests.all()]

        assert listed.count(email) == 1
        assert await requests.remove(email)
        assert not await requests.remove(email)
