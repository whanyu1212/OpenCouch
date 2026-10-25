"""Sign-in checks: Neon Auth tokens, and which routes need them."""

import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from opencouch_agent.api.identity import (
    LOCAL_USER_ID,
    FakeUserVerifier,
    IdentityUnavailableError,
    NeonAuthVerifier,
    NotAuthenticatedError,
    build_user_verifier,
)
from opencouch_agent.sessions import InMemorySessionStore
from tests.ag_ui_requests import run_input, user_message
from tests.apps import build_app

BASE_URL = "https://ep-test.neonauth.example.com/neondb/auth"
ORIGIN = "https://ep-test.neonauth.example.com"
KEY = Ed25519PrivateKey.generate()
OTHER_KEY = Ed25519PrivateKey.generate()


def _jwk(key: Ed25519PrivateKey) -> dict[str, Any]:
    public: dict[str, Any] = json.loads(
        jwt.algorithms.OKPAlgorithm.to_jwk(key.public_key())
    )
    return public | {"kid": "key-1"}


def _public_jwk(key: Ed25519PrivateKey) -> jwt.PyJWK:
    return jwt.PyJWK(_jwk(key))


def _token(key: Ed25519PrivateKey = KEY, **claims: Any) -> str:
    now = int(time.time())
    payload = {"sub": "user-123", "iss": ORIGIN, "aud": ORIGIN, "iat": now}
    payload |= {"exp": now + 900} | claims
    return jwt.encode(payload, key, algorithm="EdDSA", headers={"kid": "key-1"})


NO_EXPIRY = jwt.encode(
    {"sub": "user-123", "iss": ORIGIN, "aud": ORIGIN},
    KEY,
    algorithm="EdDSA",
    headers={"kid": "key-1"},
)


def _verifier() -> NeonAuthVerifier:
    return NeonAuthVerifier(BASE_URL, signing_key=lambda _: _public_jwk(KEY))


async def test_a_valid_token_gives_its_subject() -> None:
    assert await _verifier().user_id(f"Bearer {_token()}") == "user-123"


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "",
        "Bearer",
        "Bearer   ",
        f"Basic {_token()}",
        "Bearer not-a-jwt",
        f"Bearer {_token(exp=int(time.time()) - 60)}",
        f"Bearer {_token(iss='https://elsewhere.example.com')}",
        f"Bearer {_token(aud='https://elsewhere.example.com')}",
        f"Bearer {_token(OTHER_KEY)}",
        f"Bearer {_token(sub='')}",
        f"Bearer {NO_EXPIRY}",
    ],
    ids=[
        "no header",
        "empty header",
        "no token",
        "blank token",
        "not bearer",
        "malformed",
        "expired",
        "wrong issuer",
        "wrong audience",
        "wrong key",
        "empty subject",
        "no expiry",
    ],
)
async def test_anything_but_a_valid_neon_auth_token_is_rejected(
    authorization: str | None,
) -> None:
    with pytest.raises(NotAuthenticatedError):
        await _verifier().user_id(authorization)


async def test_a_token_signed_with_another_algorithm_is_rejected() -> None:
    now = int(time.time())
    payload = {"sub": "user-123", "iss": ORIGIN, "aud": ORIGIN, "exp": now + 900}
    forged = jwt.encode(
        payload, "a-shared-secret-of-at-least-32-bytes!", algorithm="HS256"
    )

    with pytest.raises(NotAuthenticatedError):
        await _verifier().user_id(f"Bearer {forged}")


async def test_an_anonymous_token_is_not_a_sign_in() -> None:
    with pytest.raises(NotAuthenticatedError):
        await _verifier().user_id(f"Bearer {_token(role='anonymous')}")


async def test_a_token_whose_header_cant_be_read_is_rejected() -> None:
    def unreadable(_: str) -> jwt.PyJWK:
        raise jwt.DecodeError("Invalid header padding")

    verifier = NeonAuthVerifier(BASE_URL, signing_key=unreadable)

    with pytest.raises(NotAuthenticatedError):
        await verifier.user_id("Bearer not-a-jwt")


@pytest.mark.parametrize(
    "failure",
    [
        jwt.PyJWKClientConnectionError("connection refused"),
        jwt.PyJWKClientError("The JWKS endpoint did not contain any signing keys"),
        json.JSONDecodeError("Expecting value", "<html>", 0),
    ],
    ids=["unreachable", "no usable keys", "not json"],
)
async def test_unusable_keys_mean_sign_in_cant_be_checked(failure: Exception) -> None:
    def broken(_: str) -> jwt.PyJWK:
        raise failure

    verifier = NeonAuthVerifier(BASE_URL, signing_key=broken)

    with pytest.raises(IdentityUnavailableError):
        await verifier.user_id(f"Bearer {_token()}")


async def test_the_verifier_reads_neon_auths_jwks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched: list[str] = []

    def fetch_data(client: jwt.PyJWKClient) -> dict[str, Any]:
        fetched.append(client.uri)
        return {"keys": [_jwk(KEY)]}

    monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", fetch_data)

    assert await NeonAuthVerifier(BASE_URL).user_id(f"Bearer {_token()}") == "user-123"
    assert fetched == [f"{BASE_URL}/.well-known/jwks.json"]


def test_without_a_neon_auth_url_the_service_refuses_to_start() -> None:
    with pytest.raises(ValueError, match="NEON_AUTH_BASE_URL"):
        build_user_verifier(base_url=None, fake=False, model="openai:gpt-6.1-sol")


def test_the_fake_verifier_never_runs_next_to_a_real_model() -> None:
    with pytest.raises(ValueError, match="OPENCOUCH_MODEL=test"):
        build_user_verifier(base_url=None, fake=True, model="openai:gpt-6.1-sol")


async def test_the_fake_verifier_treats_every_request_as_the_local_user() -> None:
    verifier = build_user_verifier(base_url=None, fake=True, model="test")

    assert isinstance(verifier, FakeUserVerifier)
    assert await verifier.user_id(None) == LOCAL_USER_ID


def test_a_neon_auth_url_gives_the_real_verifier() -> None:
    verifier = build_user_verifier(base_url=BASE_URL, fake=False, model="test")

    assert isinstance(verifier, NeonAuthVerifier)


def _client() -> TestClient:
    app = build_app(
        InMemorySessionStore(),
        TestModel(custom_output_text="I'm here with you."),
        verifier=_verifier(),
    )
    return TestClient(app)


def test_the_agent_and_history_need_a_sign_in() -> None:
    client = _client()

    turn = client.post("/api/agent", json=run_input(user_message("hi")))
    history = client.get("/api/threads/thread-1/messages")

    assert turn.status_code == history.status_code == 401
    assert turn.headers["www-authenticate"] == "Bearer"


def test_a_signed_in_request_reaches_the_agent() -> None:
    headers = {"Authorization": f"Bearer {_token()}"}

    response = _client().post(
        "/api/agent", json=run_input(user_message("hi")), headers=headers
    )

    assert response.status_code == 200
    assert "RUN_FINISHED" in response.text


def test_crisis_resources_and_health_stay_public() -> None:
    client = _client()

    assert client.get("/api/crisis-resources").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_sign_in_that_cant_be_checked_is_a_503_not_a_sign_out() -> None:
    def unreachable(_: str) -> jwt.PyJWK:
        raise jwt.PyJWKClientConnectionError("connection refused")

    app = build_app(
        InMemorySessionStore(),
        TestModel(),
        verifier=NeonAuthVerifier(BASE_URL, signing_key=unreachable),
    )
    headers = {"Authorization": f"Bearer {_token()}"}

    response = TestClient(app).post(
        "/api/agent", json=run_input(user_message("hi")), headers=headers
    )

    assert response.status_code == 503
