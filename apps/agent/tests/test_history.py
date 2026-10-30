"""A conversation's saved history can be shown again after a reload (HY-350)."""

import asyncio
import time
from datetime import timedelta

from fastapi.testclient import TestClient
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel

from opencouch_agent.agent import CompanionState
from opencouch_agent.api.history import shown_history, wait_for_turn
from opencouch_agent.safety.risk import RiskState
from opencouch_agent.sessions import InMemorySessionStore
from tests.ag_ui_requests import run_input, user_message
from tests.apps import build_app
from tests.fake_jev import ScriptedJev

CRISIS_MESSAGE = "i'm going to take all my pills tonight"
CRISIS_READING = {"intent": 0.9, "ideation": 0.8, "latest_adds_risk": 0.9}


def test_history_lists_what_was_said_with_the_clients_message_ids(
    client: TestClient,
) -> None:
    client.post("/api/agent", json=run_input(user_message("first", "m1")))
    client.post("/api/agent", json=run_input(user_message("second", "m2")))

    response = client.get("/api/threads/thread-1/messages")

    assert response.status_code == 200
    messages = response.json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "first"),
        ("assistant", "I'm here with you."),
        ("user", "second"),
        ("assistant", "I'm here with you."),
    ]
    assert [m["id"] for m in messages if m["role"] == "user"] == ["m1", "m2"]
    assert len({m["id"] for m in messages}) == len(messages)


def test_an_unknown_thread_has_no_history(client: TestClient) -> None:
    response = client.get("/api/threads/never-started/messages")

    assert response.status_code == 200
    assert response.json() == {
        "messages": [],
        **CompanionState().model_dump(mode="json"),
    }


def test_history_never_includes_risk_or_other_server_data() -> None:
    # The resource card's keys are the one crisis detail sent: the person
    # was shown that card in their own conversation. The risk never is.
    store = InMemorySessionStore()
    jev = ScriptedJev({CRISIS_MESSAGE: CRISIS_READING})
    client = TestClient(build_app(store, TestModel(), jev=jev))
    client.post("/api/agent", json=run_input(user_message(CRISIS_MESSAGE)))

    response = client.get("/api/threads/thread-1/messages")

    assert set(response.json()) == {
        "messages",
        "crisis_resources",
        "crisis_resources_until",
        "get_help_resources",
        "crisis_check_in",
    }
    for message in response.json()["messages"]:
        assert set(message) == {"id", "role", "content", "interrupted"}


def test_tool_calls_are_left_out_and_cut_off_replies_are_marked() -> None:
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart("what time is it?")]),
        ModelResponse(parts=[ToolCallPart("get_time", "{}", "c1")]),
        ModelRequest(parts=[ToolReturnPart("get_time", "23:00", "c1")]),
        ModelResponse(parts=[TextPart("It's late. ")]),
        ModelRequest(parts=[UserPromptPart("can't sleep")]),
        ModelResponse(parts=[TextPart("Let's try")], state="interrupted"),
    ]

    shown = shown_history(history).messages

    assert [(m.role, m.content, m.interrupted) for m in shown] == [
        ("user", "what time is it?", False),
        ("assistant", "It's late. ", False),
        ("user", "can't sleep", False),
        ("assistant", "Let's try", True),
    ]


def test_each_prompt_in_one_request_gets_its_own_id() -> None:
    history = [ModelRequest(parts=[UserPromptPart("one"), UserPromptPart("two")])]

    shown = shown_history(history).messages

    assert [m.id for m in shown] == ["history-0", "history-0-1"]


async def test_loading_waits_for_a_stopping_turn_to_save_its_cut_off_reply() -> None:
    store = InMemorySessionStore()
    await store.claim_turn("thread-1", "turn", timedelta(seconds=60))
    await store.record_user_turn(
        "thread-1", "turn", [ModelRequest(parts=[UserPromptPart("hi")])], RiskState()
    )

    async def stop_turn() -> None:
        await asyncio.sleep(0.05)
        reply = ModelResponse(parts=[TextPart("Let's")], state="interrupted")
        await store.save_run("thread-1", "turn", [reply], {})
        await store.release_turn("thread-1", "turn")

    stopping = asyncio.create_task(stop_turn())
    await wait_for_turn(store, "thread-1", timedelta(seconds=5))
    shown = shown_history((await store.load("thread-1")).history).messages
    await stopping

    assert [(m.content, m.interrupted) for m in shown] == [
        ("hi", False),
        ("Let's", True),
    ]


async def test_loading_stops_waiting_for_a_turn_running_elsewhere() -> None:
    store = InMemorySessionStore()
    await store.claim_turn("thread-1", "other-tab", timedelta(seconds=60))

    started = time.monotonic()
    await wait_for_turn(store, "thread-1", timedelta(seconds=0.2))

    assert time.monotonic() - started < 1


async def test_a_turn_whose_lease_expired_does_not_hold_up_loading() -> None:
    store = InMemorySessionStore()
    await store.claim_turn("thread-1", "crashed", timedelta(seconds=-1))

    assert not await store.turn_in_progress("thread-1")
