"""Requests for early access: public, private about who asked, and checked (HY-353)."""

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from opencouch_agent.beta.access_requests import InMemoryAccessRequests
from opencouch_agent.beta.invites import BetaInvites
from opencouch_agent.sessions import InMemorySessionStore
from tests.apps import build_app


@pytest.fixture
def requests() -> InMemoryAccessRequests:
    return InMemoryAccessRequests()


@pytest.fixture
def client(requests: InMemoryAccessRequests) -> TestClient:
    # Nobody invited and no sign-in: asking needs neither.
    app = build_app(
        InMemorySessionStore(),
        TestModel(),
        invites=BetaInvites(),
        access_requests=requests,
    )
    return TestClient(app)


def _ask(client: TestClient, email: str) -> int:
    return client.post("/api/access-requests", json={"email": email}).status_code


def test_records_a_normalised_address(
    client: TestClient, requests: InMemoryAccessRequests
) -> None:
    assert _ask(client, "  Carol@Example.com ") == 202
    assert requests.emails == {"carol@example.com"}


def test_asking_twice_looks_the_same(
    client: TestClient, requests: InMemoryAccessRequests
) -> None:
    first = client.post("/api/access-requests", json={"email": "carol@example.com"})
    second = client.post("/api/access-requests", json={"email": "CAROL@example.com"})

    assert first.json() == second.json() == {"status": "received"}
    assert requests.emails == {"carol@example.com"}


@pytest.mark.parametrize(
    "email",
    [
        "",
        "carol",
        "carol@",
        "@example.com",
        "carol@example",
        "a@b@c.com",
        "car ol@x.com",
        "a@.com",
        "a@x.",
        "a@x..com",
        "a@x.c",
        f"{'a' * 250}@x.com",
    ],
    ids=[
        "empty",
        "no at",
        "no domain",
        "no local part",
        "no dot",
        "two ats",
        "space",
        "empty label",
        "trailing dot",
        "double dot",
        "one-letter tld",
        "too long",
    ],
)
def test_rejects_what_cant_be_an_address(
    client: TestClient, requests: InMemoryAccessRequests, email: str
) -> None:
    assert _ask(client, email) == 422
    assert requests.emails == set()
