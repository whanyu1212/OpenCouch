import asyncio
import json
from collections.abc import AsyncIterator
from datetime import timedelta

import httpx
from ag_ui.core import BaseEvent, RunStartedEvent, TextMessageStartEvent
from pydantic_ai.messages import ModelMessage, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from opencouch_agent.agent import build_companion_agent
from opencouch_agent.config import Settings
from opencouch_agent.main import create_app
from opencouch_agent.sessions import InMemorySessionStore
from opencouch_agent.turn import _guard_turn
from tests.ag_ui_requests import run_input, user_message

LEASE = timedelta(seconds=60)


def _client(
    store: InMemorySessionStore,
    model: FunctionModel,
    *,
    turn_timeout_seconds: float = 90.0,
) -> httpx.AsyncClient:
    app = create_app(
        settings=Settings(turn_timeout_seconds=turn_timeout_seconds),
        agent=build_companion_agent(model),
        store=store,
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _prompts_and_replies(messages: list[ModelMessage]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart | TextPart) and isinstance(
                part.content, str
            ):
                texts.append(part.content)
    return texts


def _event_types(response: httpx.Response) -> list[str]:
    return [
        json.loads(line.removeprefix("data: "))["type"]
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


async def test_second_turn_on_a_busy_thread_gets_409_and_stores_nothing() -> None:
    store = InMemorySessionStore()
    model_started = asyncio.Event()
    finish_reply = asyncio.Event()

    async def slow_reply(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        model_started.set()
        await finish_reply.wait()
        yield "done"

    async with _client(store, FunctionModel(stream_function=slow_reply)) as http:
        first = asyncio.create_task(
            http.post("/api/agent", json=run_input(user_message("first", "m1")))
        )
        await asyncio.wait_for(model_started.wait(), timeout=5)

        second = await http.post(
            "/api/agent", json=run_input(user_message("second", "m2"))
        )
        finish_reply.set()
        first_response = await first
        third = await http.post(
            "/api/agent", json=run_input(user_message("third", "m3"))
        )

    assert second.status_code == 409
    assert first_response.status_code == 200
    assert third.status_code == 200
    history = (await store.load("thread-1")).history
    assert _prompts_and_replies(history) == ["first", "done", "third", "done"]


async def test_other_threads_are_not_blocked() -> None:
    store = InMemorySessionStore()
    await store.claim_turn("busy-thread", "someone-else", LEASE)

    async def reply(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        yield "hi"

    async with _client(store, FunctionModel(stream_function=reply)) as http:
        response = await http.post(
            "/api/agent", json=run_input(user_message("hello"), thread_id="free")
        )

    assert response.status_code == 200


async def test_turn_is_released_after_the_run_fails() -> None:
    store = InMemorySessionStore()

    async def fail(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError("model unavailable")
        yield  # Unreachable; makes this an async generator.

    async with _client(store, FunctionModel(stream_function=fail)) as http:
        await http.post("/api/agent", json=run_input(user_message("one", "m1")))
        retry = await http.post("/api/agent", json=run_input(user_message("two", "m2")))

    assert retry.status_code == 200


async def test_turn_that_runs_too_long_ends_with_run_error_and_is_released() -> None:
    store = InMemorySessionStore()

    async def hang(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        await asyncio.Event().wait()
        yield "never"

    async with _client(
        store, FunctionModel(stream_function=hang), turn_timeout_seconds=0.05
    ) as http:
        timed_out = await http.post(
            "/api/agent", json=run_input(user_message("one", "m1"))
        )
        retry = await http.post("/api/agent", json=run_input(user_message("two", "m2")))

    assert _event_types(timed_out)[-1] == "RUN_ERROR"
    assert "turn_timeout" in timed_out.text
    assert retry.status_code == 200
    # The timed-out run was stopped, so no late reply was persisted.
    history = (await store.load("thread-1")).history
    assert _prompts_and_replies(history) == ["one", "two"]


async def test_turn_is_released_when_the_client_stops_reading() -> None:
    store = InMemorySessionStore()
    await store.claim_turn("thread-1", "token", LEASE)
    inner_closed = False

    async def events() -> AsyncIterator[BaseEvent]:
        nonlocal inner_closed
        try:
            yield RunStartedEvent(thread_id="thread-1", run_id="run-1")
            yield TextMessageStartEvent(message_id="msg-1")
        finally:
            inner_closed = True

    guarded = _guard_turn(
        events(),
        store=store,
        thread_id="thread-1",
        lease_token="token",
        turn_timeout=timedelta(seconds=90),
    )
    await anext(guarded)
    await guarded.aclose()  # What Starlette does when the client disconnects.

    assert inner_closed
    assert await store.claim_turn("thread-1", "next", LEASE)


async def test_claim_fails_while_another_turn_holds_the_thread() -> None:
    store = InMemorySessionStore()

    assert await store.claim_turn("t", "first", LEASE)
    assert not await store.claim_turn("t", "second", LEASE)


async def test_expired_lease_can_be_reclaimed() -> None:
    store = InMemorySessionStore()
    await store.claim_turn("t", "crashed", timedelta(seconds=-1))

    assert await store.claim_turn("t", "next", LEASE)


async def test_release_with_a_stale_token_keeps_the_newer_claim() -> None:
    store = InMemorySessionStore()
    await store.claim_turn("t", "old", timedelta(seconds=-1))
    await store.claim_turn("t", "new", LEASE)

    await store.release_turn("t", "old")

    assert not await store.claim_turn("t", "third", LEASE)
