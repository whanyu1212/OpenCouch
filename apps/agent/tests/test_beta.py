"""The private beta: only invited, verified emails use the coach (HY-353)."""

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from opencouch_agent.api.identity import SignedInUser
from opencouch_agent.beta.invites import build_beta_invites
from opencouch_agent.sessions import InMemorySessionStore
from tests.ag_ui_requests import run_input, user_message
from tests.apps import build_app

INVITES = build_beta_invites(
    invites=" Alice@Example.com , bob@example.com,,", everyone=False, model="test"
)


@pytest.mark.parametrize(
    ("email", "allowed"),
    [
        ("alice@example.com", True),
        ("  ALICE@example.COM ", True),
        ("bob@example.com", True),
        ("carol@example.com", False),
        (None, False),
    ],
    ids=["invited", "case and spaces", "second invite", "not invited", "unverified"],
)
def test_allows_only_invited_verified_emails(email: str | None, allowed: bool) -> None:
    assert INVITES.allows(email) is allowed


@pytest.mark.parametrize("invites", [None, "", " , "], ids=["unset", "empty", "blank"])
def test_no_invites_lets_nobody_in(invites: str | None) -> None:
    beta = build_beta_invites(invites=invites, everyone=False, model="openai:x")

    assert not beta.allows("alice@example.com")


def test_open_beta_never_runs_with_a_real_model() -> None:
    with pytest.raises(ValueError, match="OPENCOUCH_MODEL=test"):
        build_beta_invites(invites=None, everyone=True, model="openai:x")


def test_open_beta_lets_everyone_in() -> None:
    beta = build_beta_invites(invites=None, everyone=True, model="test")

    assert beta.allows(None)


class _Caller:
    """Signs in whoever the bearer token names, with that address verified."""

    async def identify(self, authorization: str | None) -> SignedInUser:
        email = (authorization or "").removeprefix("Bearer ")
        return SignedInUser(email, email)


@pytest.mark.parametrize(
    ("caller", "status"),
    [("alice@example.com", 200), ("carol@example.com", 403)],
    ids=["invited", "not invited"],
)
def test_only_invited_users_reach_the_coach(caller: str, status: int) -> None:
    app = build_app(
        InMemorySessionStore(), TestModel(), verifier=_Caller(), invites=INVITES
    )
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {caller}"}

    turn = client.post(
        "/api/agent", json=run_input(user_message("hi")), headers=headers
    )
    history = client.get("/api/threads/thread-1/messages", headers=headers)

    assert turn.status_code == history.status_code == status
    if status == 403:
        assert turn.json()["detail"]["code"] == "not_invited"
