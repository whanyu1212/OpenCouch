"""A conversation's saved history can be shown again after a reload (HY-350)."""

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

from opencouch_agent.api.history import shown_history
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
    assert response.json() == {"messages": []}


def test_history_never_includes_risk_or_other_server_data() -> None:
    store = InMemorySessionStore()
    jev = ScriptedJev({CRISIS_MESSAGE: CRISIS_READING})
    client = TestClient(build_app(store, TestModel(), jev=jev))
    client.post("/api/agent", json=run_input(user_message(CRISIS_MESSAGE)))

    response = client.get("/api/threads/thread-1/messages")

    assert set(response.json()) == {"messages"}
    for message in response.json()["messages"]:
        assert set(message) == {"id", "role", "content", "interrupted"}
    assert "crisis" not in response.text.lower()


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
