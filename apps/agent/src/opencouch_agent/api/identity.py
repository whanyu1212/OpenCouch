"""Who is calling: the user ID behind each request to a protected route (HY-207).

The web app signs people in with Neon Auth (managed Better Auth) and sends a
short-lived Neon Auth JWT as `Authorization: Bearer <token>`. The agent checks
it against Neon Auth's published keys and trusts only its `sub`.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Final, Protocol
from urllib.parse import urlsplit

import jwt
from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# The ID every request gets from the fake verifier, for local runs.
LOCAL_USER_ID: Final = "local-user"
# Fetching the keys happens at most once per key rotation, but it mustn't
# hold a turn up for long when Neon Auth is unreachable.
JWKS_TIMEOUT_SECONDS: Final = 5.0

SigningKeyLookup = Callable[[str], jwt.PyJWK]
"""The public key a token was signed with, found from the token's header."""


class NotAuthenticatedError(Exception):
    """The request carries no valid sign-in."""


class IdentityUnavailableError(Exception):
    """Sign-ins can't be checked right now, because Neon Auth's keys are unreachable."""


class UserVerifier(Protocol):
    """Finds the verified user behind a request."""

    async def user_id(self, authorization: str | None) -> str:
        """The user ID for a request's `Authorization` header.

        Raises:
            NotAuthenticatedError: If the header is missing or its token isn't valid.
            IdentityUnavailableError: If the token can't be checked right now.
        """
        ...


class NeonAuthVerifier:
    """Checks Neon Auth JWTs: EdDSA-signed, and issued for our Neon Auth origin."""

    def __init__(
        self, base_url: str, *, signing_key: SigningKeyLookup | None = None
    ) -> None:
        """Set up the verifier.

        Args:
            base_url: The Neon Auth URL, including its path.
            signing_key: Finds a token's public key. Defaults to Neon Auth's
                JWKS at `<base_url>/.well-known/jwks.json`, cached.
        """
        parts = urlsplit(base_url)
        # Neon Auth sets both `iss` and `aud` to its own origin.
        self._origin = f"{parts.scheme}://{parts.netloc}"
        if signing_key is None:
            # The key set is cached for 5 minutes and refetched for an unknown
            # key ID, so a rotated or revoked key stops working within minutes.
            jwks = jwt.PyJWKClient(
                f"{base_url.rstrip('/')}/.well-known/jwks.json",
                timeout=JWKS_TIMEOUT_SECONDS,
            )
            signing_key = jwks.get_signing_key_from_jwt
        self._signing_key = signing_key

    async def user_id(self, authorization: str | None) -> str:
        """The user ID in a valid Neon Auth token; see `UserVerifier.user_id`."""
        token = _bearer_token(authorization)
        key = await self._key_for(token)
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=["EdDSA"],
                issuer=self._origin,
                audience=self._origin,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError as error:
            # Only the error type: the token itself is a credential.
            logger.info("Rejected a sign-in token: %s", type(error).__name__)
            raise NotAuthenticatedError from error
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject:
            raise NotAuthenticatedError
        # Neon Auth can also issue tokens to visitors who never signed in.
        if claims.get("role") == "anonymous":
            raise NotAuthenticatedError
        return subject

    async def _key_for(self, token: str) -> jwt.PyJWK:
        try:
            # The lookup may fetch the key set, which blocks.
            return await asyncio.to_thread(self._signing_key, token)
        except jwt.DecodeError as error:
            raise NotAuthenticatedError from error
        except (jwt.PyJWKClientError, json.JSONDecodeError) as error:
            # A key set that can't be fetched or read, or lacks the token's key,
            # is Neon Auth's problem, not a sign-out. A forged key ID lands here
            # too, and gets in no further.
            raise IdentityUnavailableError from error


class FakeUserVerifier:
    """Treats every request as `LOCAL_USER_ID`. For tests and local runs only."""

    async def user_id(self, authorization: str | None) -> str:
        """Always `LOCAL_USER_ID`, whatever the request carries."""
        return LOCAL_USER_ID


def build_user_verifier(
    *, base_url: str | None, fake: bool, model: str
) -> UserVerifier:
    """The verifier for the service's settings.

    Args:
        base_url: The Neon Auth URL, if set.
        fake: Whether the fake verifier was asked for explicitly.
        model: The agent's model string.

    Raises:
        ValueError: If the fake is asked for with a real model, or no Neon
            Auth URL is set and the fake wasn't asked for.
    """
    # Same rule as the fake Jev: the fake never stands in for a missing URL on
    # its own, and never runs next to a real model, so a misconfigured deploy
    # fails to start instead of serving anyone who asks.
    if fake:
        if model != "test":
            raise ValueError("OPENCOUCH_FAKE_AUTH only runs with OPENCOUCH_MODEL=test")
        logger.warning("Using the fake user verifier: every request is one local user")
        return FakeUserVerifier()
    if not base_url:
        raise ValueError(
            "NEON_AUTH_BASE_URL is required: every conversation belongs to a "
            "signed-in user. For a local run with OPENCOUCH_MODEL=test, set "
            "OPENCOUCH_FAKE_AUTH=true."
        )
    return NeonAuthVerifier(base_url)


def signed_in_user(verifier: UserVerifier) -> Callable[..., Awaitable[str]]:
    """A FastAPI dependency giving the request's verified user ID.

    It raises `HTTPException` 401 without a valid sign-in, and 503 if the
    sign-in can't be checked right now.
    """

    async def user_id(authorization: Annotated[str | None, Header()] = None) -> str:
        return await _verified_user_id(verifier, authorization)

    return user_id


async def _verified_user_id(verifier: UserVerifier, authorization: str | None) -> str:
    try:
        return await verifier.user_id(authorization)
    except NotAuthenticatedError as error:
        raise HTTPException(
            status_code=401,
            detail="Sign in to continue.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from error
    except IdentityUnavailableError as error:
        logger.warning(
            "Neon Auth's signing keys couldn't be used: %s",
            type(error.__cause__).__name__,
        )
        raise HTTPException(
            status_code=503, detail="Sign-in can't be checked right now."
        ) from error


def _bearer_token(authorization: str | None) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise NotAuthenticatedError
    return token.strip()
