"""A turn's reply is saved however its run ends (HY-204)."""

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from datetime import timedelta
from typing import Any

import httpx
import pytest
from ag_ui.core import BaseEvent, TextMessageContentEvent
from pydantic_ai import CancellationToken
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.ui.ag_ui import AGUIAdapter

from opencouch_agent import turn as turn_module
from opencouch_agent.agent import CompanionDeps, build_companion_agent
from opencouch_agent.api.adapter import ServerOwnedAGUIAdapter
from opencouch_agent.config import Settings
from opencouch_agent.main import create_app
from opencouch_agent.run_recorder import RunRecorder
from opencouch_agent.safety.audit import InMemoryCrisisAuditLog
from opencouch_agent.safety.crisis import CrisisResponder, build_crisis_agent
from opencouch_agent.safety.crisis_resources import load_shipped_catalog
from opencouch_agent.sessions import InMemorySessionStore
from opencouch_agent.turn import (
    TurnAgents,
    _ActiveRun,
    _guard_turn,
    _start_run,
    _UserTurn,
)
from tests.ag_ui_requests import event_types, run_input, user_message
from tests.apps import build_app
from tests.fake_jev import ScriptedJev

LEASE = timedelta(seconds=60)
SHOWN = "I hear you. Please call "


async def shown_then_error(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
    yield "I hear you. "
    yield "Please call "
    raise RuntimeError("model unavailable")


async def shown_then_stall(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
    yield "I hear you. "
    yield "Please call "
    await asyncio.sleep(30)
    yield "never sent"


def _client(
    store: InMemorySessionStore, model: FunctionModel, *, turn_timeout: float = 90.0
) -> httpx.AsyncClient:
    app = build_app(store, model, settings=Settings(turn_timeout_seconds=turn_timeout))
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _reply(history: Sequence[ModelMessage]) -> ModelResponse:
    last = history[-1]
    assert isinstance(last, ModelResponse)
    return last


def _text(response: ModelResponse) -> str:
    return "".join(p.content for p in response.parts if isinstance(p, TextPart))


async def test_a_reply_that_errors_saves_what_the_user_was_shown() -> None:
    store = InMemorySessionStore()

    async with _client(store, FunctionModel(stream_function=shown_then_error)) as http:
        response = await http.post("/api/agent", json=run_input(user_message("hi")))

    reply = _reply((await store.load("thread-1")).history)
    assert event_types(response.text)[-1] == "RUN_ERROR"
    assert _text(reply) == SHOWN
    assert reply.state == "interrupted"


async def test_a_reply_cut_off_by_the_time_limit_saves_its_partial_reply() -> None:
    store = InMemorySessionStore()
    model = FunctionModel(stream_function=shown_then_stall)

    async with _client(store, model, turn_timeout=0.5) as http:
        response = await http.post("/api/agent", json=run_input(user_message("hi")))

    reply = _reply((await store.load("thread-1")).history)
    assert event_types(response.text)[-1] == "RUN_ERROR"
    assert _text(reply) == SHOWN
    assert reply.state == "interrupted"


async def test_a_reply_cut_off_by_a_disconnect_saves_its_partial_reply() -> None:
    store = InMemorySessionStore()
    guarded = await _guarded_turn(
        store, FunctionModel(stream_function=shown_then_stall)
    )

    shown = ""
    async for event in guarded:
        if isinstance(event, TextMessageContentEvent):
            shown += event.delta
        if shown == SHOWN:
            break
    await guarded.aclose()  # What Starlette does when the client disconnects.

    reply = _reply((await store.load("thread-1")).history)
    assert _text(reply) == SHOWN
    assert reply.state == "interrupted"
    assert await store.claim_turn("thread-1", "next", LEASE)


async def test_a_cut_off_reply_keeps_the_tool_calls_that_finished() -> None:
    store = InMemorySessionStore()

    async def tool_then_stall(
        messages: list[ModelMessage], _: AgentInfo
    ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        returned = any(
            isinstance(part, ToolReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if not returned:
            yield {0: DeltaToolCall(name="get_time", json_args="{}", tool_call_id="c1")}
            return
        async for chunk in shown_then_stall(messages, _):
            yield chunk

    agent = build_companion_agent(FunctionModel(stream_function=tool_then_stall))

    @agent.tool_plain
    def get_time() -> str:
        return "23:00"

    app = create_app(
        settings=Settings(turn_timeout_seconds=0.5),
        agent=agent,
        crisis_agent=build_crisis_agent(TestModel()),
        jev=ScriptedJev(),
        store=store,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        await http.post("/api/agent", json=run_input(user_message("hi")))

    history = (await store.load("thread-1")).history
    parts = [part for message in history for part in message.parts]
    assert any(isinstance(part, ToolCallPart) for part in parts)
    assert any(isinstance(part, ToolReturnPart) for part in parts)
    assert _text(_reply(history)) == SHOWN
    assert _reply(history).state == "interrupted"


async def test_a_run_cut_off_before_any_reply_saves_no_empty_reply() -> None:
    store = InMemorySessionStore()

    async def stall(_: list[ModelMessage], __: AgentInfo) -> AsyncIterator[str]:
        await asyncio.sleep(30)
        yield "never sent"

    async with _client(
        store, FunctionModel(stream_function=stall), turn_timeout=0.2
    ) as http:
        await http.post("/api/agent", json=run_input(user_message("hi")))

    history = (await store.load("thread-1")).history
    assert not any(isinstance(message, ModelResponse) for message in history)


async def test_a_run_that_ignores_cancellation_still_saves_what_was_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(turn_module, "CANCEL_GRACE_SECONDS", 0.1)
    store = InMemorySessionStore()
    await store.claim_turn("thread-1", "token", LEASE)

    async def stuck() -> AsyncIterator[BaseEvent]:
        yield TextMessageContentEvent(message_id="msg-1", delta=SHOWN)
        await asyncio.sleep(10)

    run = _ActiveRun(stuck(), CancellationToken(), _recorder(store))
    guarded = _guard_turn(
        run,
        store=store,
        turn=_UserTurn("thread-1", "token", []),
        turn_timeout=timedelta(seconds=90),
    )
    started = time.monotonic()
    await anext(guarded)
    await guarded.aclose()

    assert time.monotonic() - started < 2
    assert _text(_reply((await store.load("thread-1")).history)) == SHOWN
    assert await store.claim_turn("thread-1", "next", LEASE)


async def test_the_time_limit_holds_even_when_the_run_ignores_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(turn_module, "CANCEL_GRACE_SECONDS", 1.0)
    store = InMemorySessionStore()
    await store.claim_turn("thread-1", "token", LEASE)

    async def stuck() -> AsyncIterator[BaseEvent]:
        yield TextMessageContentEvent(message_id="msg-1", delta=SHOWN)
        await asyncio.sleep(10)

    run = _ActiveRun(stuck(), CancellationToken(), _recorder(store))
    guarded = _guard_turn(
        run,
        store=store,
        turn=_UserTurn("thread-1", "token", []),
        turn_timeout=timedelta(seconds=0.2),
    )
    started = time.monotonic()
    types = []
    async for event in guarded:
        types.append(event.type)
        if event.type == "RUN_ERROR":
            error_after = time.monotonic() - started

    # The client hears about the timeout at the deadline; only cleanup waits
    # out the grace period.
    assert types == ["TEXT_MESSAGE_CONTENT", "RUN_ERROR"]
    assert error_after < 0.6
    assert _text(_reply((await store.load("thread-1")).history)) == SHOWN
    assert await store.claim_turn("thread-1", "next", LEASE)


async def test_a_disconnect_while_waiting_on_the_model_saves_what_was_shown() -> None:
    store = InMemorySessionStore()
    guarded = await _guarded_turn(
        store, FunctionModel(stream_function=shown_then_stall)
    )
    shown_everything = asyncio.Event()

    async def read() -> None:
        shown = ""
        async for event in guarded:
            if isinstance(event, TextMessageContentEvent):
                shown += event.delta
            if shown == SHOWN:
                shown_everything.set()

    reader = asyncio.create_task(read())
    await asyncio.wait_for(shown_everything.wait(), timeout=5)
    await asyncio.sleep(0.05)  # The reader is now waiting on the stalled model.
    reader.cancel()  # What Starlette does when the client disconnects.
    with contextlib.suppress(asyncio.CancelledError):
        await reader

    reply = _reply((await store.load("thread-1")).history)
    assert _text(reply) == SHOWN
    assert reply.state == "interrupted"
    assert await store.claim_turn("thread-1", "next", LEASE)


async def test_a_failed_turn_that_lost_its_lease_saves_nothing() -> None:
    store = InMemorySessionStore()
    model_started = asyncio.Event()
    fail_now = asyncio.Event()

    async def wait_then_error(
        _: list[ModelMessage], __: AgentInfo
    ) -> AsyncIterator[str]:
        model_started.set()
        yield "I hear you. "
        await fail_now.wait()
        raise RuntimeError("model unavailable")

    async with _client(store, FunctionModel(stream_function=wait_then_error)) as http:
        first = asyncio.create_task(
            http.post("/api/agent", json=run_input(user_message("hi")))
        )
        await asyncio.wait_for(model_started.wait(), timeout=5)
        # Another turn takes the thread after this one's lease expired.
        store._turn_leases["thread-1"] = ("newer-turn", time.monotonic() + 60)
        fail_now.set()
        await first

    history = (await store.load("thread-1")).history
    assert not any(isinstance(message, ModelResponse) for message in history)


async def test_the_next_turn_sees_the_interrupted_reply() -> None:
    store = InMemorySessionStore()
    seen: list[ModelMessage] = []
    calls = 0

    async def fail_then_record(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            async for chunk in shown_then_error(messages, info):
                yield chunk
        seen.extend(messages)
        yield "ok"

    async with _client(store, FunctionModel(stream_function=fail_then_record)) as http:
        await http.post("/api/agent", json=run_input(user_message("hi", "m1")))
        second = await http.post(
            "/api/agent", json=run_input(user_message("still there?", "m2"))
        )

    interrupted = [m for m in seen if isinstance(m, ModelResponse)]
    assert second.status_code == 200
    assert [(_text(m), m.state) for m in interrupted] == [(SHOWN, "interrupted")]


async def _guarded_turn(
    store: InMemorySessionStore, model: FunctionModel
) -> AsyncGenerator[BaseEvent]:
    """A claimed turn started through the real pipeline, before streaming."""
    await store.claim_turn("thread-1", "token", LEASE)
    agent = build_companion_agent(model)
    body = json.dumps(run_input(user_message("hi"))).encode()
    adapter = ServerOwnedAGUIAdapter(
        agent=agent, run_input=AGUIAdapter.build_run_input(body)
    )
    turn = _UserTurn("thread-1", "token", adapter.new_user_messages())
    crisis = CrisisResponder.from_catalog(
        build_crisis_agent(TestModel()), load_shipped_catalog()
    )
    agents = TurnAgents(agent, crisis)
    _, run = await _start_run(
        adapter,
        turn,
        agents=agents,
        jev=ScriptedJev(),
        store=store,
        audit=InMemoryCrisisAuditLog(),
    )
    return _guard_turn(run, store=store, turn=turn, turn_timeout=timedelta(seconds=90))


class _CountingStore(InMemorySessionStore):
    def __init__(self, *, fail_first: bool = False) -> None:
        super().__init__()
        self.saves: list[Sequence[ModelMessage]] = []
        self._fail_next = fail_first

    async def save_run(
        self,
        thread_id: str,
        lease_token: str,
        messages: Sequence[ModelMessage],
        state: dict[str, Any],
    ) -> bool:
        if self._fail_next:
            self._fail_next = False
            raise ConnectionError("database unavailable")
        self.saves.append(messages)
        return await super().save_run(thread_id, lease_token, messages, state)


def _recorder(store: InMemorySessionStore) -> RunRecorder:
    return RunRecorder(store, "thread-1", "token", CompanionDeps())


async def test_a_run_that_was_saved_is_not_saved_again_by_the_fallback() -> None:
    store = _CountingStore()
    await store.claim_turn("thread-1", "token", LEASE)
    recorder = _recorder(store)
    result = await build_companion_agent(TestModel()).run("hi", deps=CompanionDeps())

    await recorder.save_completed(result)
    await recorder.save_shown()

    assert len(store.saves) == 1


async def test_the_fallback_retries_a_save_that_raised() -> None:
    store = _CountingStore(fail_first=True)
    await store.claim_turn("thread-1", "token", LEASE)
    recorder = _recorder(store)
    result = await build_companion_agent(TestModel()).run("hi", deps=CompanionDeps())

    with pytest.raises(ConnectionError):
        await recorder.save_completed(result)
    await recorder.save_shown()

    assert store.saves == [[]]


async def test_a_run_that_showed_nothing_saves_no_reply() -> None:
    store = _CountingStore()
    await store.claim_turn("thread-1", "token", LEASE)

    await _recorder(store).save_shown()

    assert store.saves == [[]]
