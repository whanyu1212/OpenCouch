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
    """`JevClient` backed by TypeSafe's SDK, tuned for live turns."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = GATEWAY_BASE_URL,
        model: str = GATEWAY_MODEL,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        _keep_request_bodies_out_of_logs(logging.getLogger("typesafe_sdk"))
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            retry=LIVE_RETRY,
            timeout=LIVE_ATTEMPT_TIMEOUT_SECONDS,
            transport=transport,
        )

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        """Ask every question in one request.

        Raises:
            KeyError: If Jev's response has no answer for a question.
            TypeSafeError: If the request fails after the live retry.
        """
        response = await self._client.system_one(state=state, questions=questions)
        return {name: response.nouls[name].noul for name in questions}

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


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
