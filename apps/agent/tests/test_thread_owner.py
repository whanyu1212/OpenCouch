"""Each conversation thread belongs to the user who started it (HY-221)."""

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from opencouch_agent.sessions import InMemorySessionStore
from tests.ag_ui_requests import reply_text, run_input, user_message
from tests.apps import build_app
from tests.fake_jev import ScriptedJev


class _NamedUser:
    """Signs in whoever the bearer token names: `Bearer alice` is alice."""

    async def user_id(self, authorization: str | None) -> str:
        return (authorization or "").removeprefix("Bearer ")


def _as(user: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {user}"}


@pytest.fixture
def jev() -> ScriptedJev:
    return ScriptedJev()


@pytest.fixture
def store() -> InMemorySessionStore:
    return InMemorySessionStore()


@pytest.fixture
def client(store: InMemorySessionStore, jev: ScriptedJev) -> TestClient:
    companion = TestModel(custom_output_text="I'm here with you.")
    return TestClient(build_app(store, companion, jev=jev, verifier=_NamedUser()))


def _send(client: TestClient, user: str, text: str, message_id: str) -> int:
    response = client.post(
        "/api/agent",
        json=run_input(user_message(text, message_id)),
        headers=_as(user),
    )
    return response.status_code


async def test_someone_elses_thread_cant_be_written_to_or_screened(
    client: TestClient, store: InMemorySessionStore, jev: ScriptedJev
) -> None:
    assert _send(client, "alice", "my worry", "m1") == 200

    status = _send(client, "bob", "let me in", "m2")

    assert status == 404
    assert jev.screened == ["my worry"]
    history = (await store.load("thread-1")).history
    assert "let me in" not in str(history)


async def test_someone_elses_thread_cant_be_read(client: TestClient) -> None:
    _send(client, "alice", "my worry", "m1")

    theirs = client.get("/api/threads/thread-1/messages", headers=_as("bob"))
    mine = client.get("/api/threads/thread-1/messages", headers=_as("alice"))

    assert theirs.status_code == 404
    assert "my worry" not in theirs.text
    assert [message["content"] for message in mine.json()["messages"]] == [
        "my worry",
        "I'm here with you.",
    ]


async def test_a_refused_stranger_leaves_the_thread_free_for_its_owner(
    client: TestClient, store: InMemorySessionStore
) -> None:
    _send(client, "alice", "my worry", "m1")
    _send(client, "bob", "let me in", "m2")

    assert not await store.turn_in_progress("thread-1")
    assert _send(client, "alice", "still here", "m3") == 200


def test_a_new_thread_reads_as_empty(client: TestClient) -> None:
    response = client.get("/api/threads/new-thread/messages", headers=_as("bob"))

    assert response.status_code == 200
    assert response.json()["messages"] == []


def test_the_first_turn_makes_a_new_thread_its_senders(client: TestClient) -> None:
    status = _send(client, "bob", "hello", "m1")

    assert status == 200
    reply = client.post(
        "/api/agent",
        json=run_input(user_message("again", "m2")),
        headers=_as("bob"),
    )
    assert reply_text(reply) == "I'm here with you."
