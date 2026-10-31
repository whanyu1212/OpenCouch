"""Thin client over Jev, so the rest of the service never touches the SDK.

Jev is called through Vercel AI Gateway's TypeSafe-compatible endpoint, using
TypeSafe's own SDK. Every request carries the user's recent messages, so
nothing here may log a request or response body.
"""

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Final, Protocol

import httpx2
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Noul,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeRateLimitError,
)

logger = logging.getLogger(__name__)

GATEWAY_BASE_URL: Final = "https://ai-gateway.vercel.sh/typesafe"
GATEWAY_MODEL: Final = "typesafe-ai/jev"

# Successful Jev calls take up to ~1.5s, so an attempt that hasn't answered in
# 2s is likely stuck. In FirstRead's evals, 503s cleared on a quick retry, while
# 429s asked for ~38s waits that a live turn can't afford: retry once, fast,
# and ignore Retry-After. A timed-out attempt isn't retried, because a retry
# after 2s couldn't finish inside the caller's deadline anyway.
LIVE_ATTEMPT_TIMEOUT_SECONDS: Final = 2.0
LIVE_RETRY: Final = RetryPolicy(
    max_retries=1,
    backoff_initial=0.25,
    backoff_max=0.25,
    backoff_jitter=0.0,
    respect_retry_after=False,
    api_timeout_error=False,
    timeout=None,
)


class FailureReason(StrEnum):
    """Why a Jev call failed, for logs and the audit record."""

    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class JevClient(Protocol):
    """Asks Jev yes/no questions about one piece of text."""

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        """Return Jev's probability of "yes", from 0 to 1, keyed by question name."""
        ...


class TypeSafeJevClient:
    """`JevClient` backed by TypeSafe's SDK, tuned for live turns by default.

    Batch callers such as the evals pass a more patient `retry` and
    `attempt_timeout_seconds`; live turns keep the defaults.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = GATEWAY_BASE_URL,
        model: str = GATEWAY_MODEL,
        retry: RetryPolicy = LIVE_RETRY,
        attempt_timeout_seconds: float = LIVE_ATTEMPT_TIMEOUT_SECONDS,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        _keep_request_bodies_out_of_logs(logging.getLogger("typesafe_sdk"))
        # Model IDs only, never request text; batch evals persist these for drift.
        self.served_models: set[str] = set()
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            retry=retry,
            timeout=attempt_timeout_seconds,
            transport=transport,
        )

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        """Ask every question in one request.

        Raises:
            KeyError: If Jev's response has no answer for a question.
            TypeSafeError: If the request fails after the live retry.
        """
        response = await self._client.system_one(state=state, questions=questions)
        self.served_models.add(response.model)
        return {name: response.nouls[name].noul for name in questions}

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


class FakeJevClient:
    """Answers "no" to every question, for local runs without a Jev key.

    Every message reads as no risk, so screening runs but protects nothing.
    `build_jev_client` only allows it with the test model.
    """

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        """Return 0 for every question."""
        logger.warning("Fake Jev: this turn was not screened")
        return dict.fromkeys(questions, 0.0)


def build_jev_client(*, api_key: str | None, fake: bool, model: str) -> JevClient:
    """The Jev client for the service's settings.

    Args:
        api_key: The Vercel AI Gateway key, if set.
        fake: Whether the fake client was asked for explicitly.
        model: The agent's model string.

    Raises:
        ValueError: If the fake is asked for with a real model, or no key is
            set and the fake wasn't asked for.
    """
    # Safety: the fake never stands in for a missing key on its own, and never
    # runs next to a real model, so a misconfigured deploy fails to start
    # instead of serving unscreened turns.
    if fake:
        if model != "test":
            raise ValueError("OPENCOUCH_FAKE_JEV only runs with OPENCOUCH_MODEL=test")
        logger.warning("Using the fake Jev client: no turn is really screened")
        return FakeJevClient()
    if not api_key:
        raise ValueError(
            "AI_GATEWAY_API_KEY is required: every turn is screened by Jev. For a "
            "local run with OPENCOUCH_MODEL=test, set OPENCOUCH_FAKE_JEV=true."
        )
    return TypeSafeJevClient(api_key=api_key)


def failure_reason(error: BaseException) -> FailureReason:
    """Classify a failed Jev call."""
    # Check timeouts first: the SDK's timeout error is also a connection error.
    if isinstance(error, TypeSafeAPITimeoutError | TimeoutError):
        return FailureReason.TIMEOUT
    if isinstance(error, TypeSafeRateLimitError):
        return FailureReason.RATE_LIMITED
    if isinstance(error, TypeSafeAPIConnectionError):
        return FailureReason.UNAVAILABLE
    if isinstance(error, TypeSafeAPIError) and error.status >= 500:
        return FailureReason.UNAVAILABLE
    return FailureReason.ERROR


class _DropDebugRecords(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO


def _keep_request_bodies_out_of_logs(sdk_logger: logging.Logger) -> None:
    # Privacy: at DEBUG the SDK logs full request and response bodies, i.e. the
    # user's messages. A filter rather than a level, so it still holds if
    # logging is reconfigured to DEBUG later.
    if not any(isinstance(f, _DropDebugRecords) for f in sdk_logger.filters):
        sdk_logger.addFilter(_DropDebugRecords())
