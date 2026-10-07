"""Integration tests for the Postgres session store.

Set `OPENCOUCH_TEST_POSTGRES_URL` to run them; they are skipped otherwise.
"""

import os
import uuid
from collections.abc import AsyncIterator

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

    await store.save_run(thread_id, [], {"mode": "chat"})
    await store.save_run(thread_id, [], {"mode": "exercise"})

    assert (await store.load(thread_id)).state == {"mode": "exercise"}


async def test_save_run_appends_after_user_message(
    store: PostgresSessionStore,
) -> None:
    thread_id = _thread_id()
    user_turn: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("hi")])]
    reply: list[ModelMessage] = [ModelResponse(parts=[TextPart("hello")])]

    await store.append_messages(thread_id, user_turn)
    await store.save_run(thread_id, reply, {"mode": "chat"})
    session = await store.load(thread_id)

    assert session.history == [*user_turn, *reply]
    assert session.state == {"mode": "chat"}


async def test_create_schema_is_idempotent(store: PostgresSessionStore) -> None:
    await store.create_schema()
