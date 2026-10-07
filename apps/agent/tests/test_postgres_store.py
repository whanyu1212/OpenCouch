"""Integration tests for the Postgres session store.

Set `OPENCOUCH_TEST_POSTGRES_URL` to run them; they are skipped otherwise.
"""

import os
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from psycopg_pool import AsyncConnectionPool
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from opencouch_agent.sessions.postgres import PostgresSessionStore

POSTGRES_URL = os.environ.get("OPENCOUCH_TEST_POSTGRES_URL")

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


async def test_unknown_thread_loads_empty(store: PostgresSessionStore) -> None:
    session = await store.load(_thread_id())

    assert session.history == []
    assert session.state == {}


async def test_messages_round_trip_in_order(store: PostgresSessionStore) -> None:
    thread_id = _thread_id()
    first_turn: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(parts=[TextPart("reply one")]),
    ]
    second_turn: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("second")])]

    await store.append_messages(thread_id, first_turn)
    await store.append_messages(thread_id, second_turn)
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
    await store.claim_turn(thread_id, "token", timedelta(seconds=60))

    await store.append_messages(thread_id, user_turn)
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


async def test_claiming_keeps_existing_history(store: PostgresSessionStore) -> None:
    thread_id = _thread_id()
    await store.append_messages(thread_id, [ModelRequest(parts=[UserPromptPart("hi")])])

    await store.claim_turn(thread_id, "token", timedelta(seconds=60))

    assert len((await store.load(thread_id)).history) == 1
