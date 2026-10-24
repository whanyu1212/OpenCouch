"""Oversized requests and messages are refused before anything is screened or
stored (HY-347)."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from opencouch_agent.sessions import InMemorySessionStore
from opencouch_agent.turn import MAX_MESSAGE_CHARS, MAX_REQUEST_BYTES
from tests.ag_ui_requests import run_input, user_message
from tests.apps import build_app
from tests.fake_jev import ScriptedJev


@pytest.fixture
def jev() -> ScriptedJev:
    return ScriptedJev()


@pytest.fixture
def client(store: InMemorySessionStore, jev: ScriptedJev) -> TestClient:
    return TestClient(build_app(store, TestModel(), jev=jev))


async def test_an_oversized_request_is_refused_before_anything_runs(
    client: TestClient, store: InMemorySessionStore, jev: ScriptedJev
) -> None:
    body = run_input(user_message("hi"))
    body["forwardedProps"] = {"padding": "x" * MAX_REQUEST_BYTES}

    response = client.post("/api/agent", json=body)

    assert response.status_code == 413
    assert jev.screened == []
    assert (await store.load("thread-1")).history == []


def _streamed(body: dict[str, object]) -> Iterator[bytes]:
    """The body sent in chunks, without a Content-Length, as a proxy may."""
    encoded = json.dumps(body).encode()
    for start in range(0, len(encoded), 4096):
        yield encoded[start : start + 4096]


def test_an_oversized_streamed_request_is_refused_before_it_is_screened(
    client: TestClient, jev: ScriptedJev
) -> None:
    body = run_input(user_message("hi"))
    body["forwardedProps"] = {"padding": "x" * MAX_REQUEST_BYTES}

    response = client.post(
        "/api/agent",
        content=_streamed(body),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert jev.screened == []


def test_a_streamed_request_within_the_limit_is_answered(
    client: TestClient, jev: ScriptedJev
) -> None:
    response = client.post(
        "/api/agent",
        content=_streamed(run_input(user_message("hi"))),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 200
    assert jev.screened == ["hi"]


async def test_a_message_over_the_limit_is_refused_before_it_is_screened(
    client: TestClient, store: InMemorySessionStore, jev: ScriptedJev
) -> None:
    message = "a" * (MAX_MESSAGE_CHARS + 1)

    response = client.post("/api/agent", json=run_input(user_message(message)))

    assert response.status_code == 422
    assert jev.screened == []
    assert (await store.load("thread-1")).history == []


def test_a_message_at_the_limit_is_screened_and_answered(
    client: TestClient, jev: ScriptedJev
) -> None:
    message = "a" * MAX_MESSAGE_CHARS

    response = client.post("/api/agent", json=run_input(user_message(message)))

    assert response.status_code == 200
    assert jev.screened == [message]
