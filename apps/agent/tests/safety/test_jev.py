import json
import logging
from collections.abc import Callable

import httpx2
import pytest
from typesafe_sdk import (
    Noul,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeRateLimitError,
)

from opencouch_agent.safety.jev import (
    GATEWAY_MODEL,
    FailureReason,
    TypeSafeJevClient,
    failure_reason,
)

QUESTIONS = {"ideation": Noul(instructions="q1"), "intent": Noul(instructions="q2")}

Handler = Callable[[httpx2.Request], httpx2.Response]


def answered(probabilities: dict[str, float]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": GATEWAY_MODEL,
            "usage": {"input_tokens": 10, "output_tokens": 2},
            "answers": {
                name: {"type": "noul", "noul": probability}
                for name, probability in probabilities.items()
            },
        },
    )


def jev_client(handler: Handler) -> TypeSafeJevClient:
    return TypeSafeJevClient(
        api_key="test-key", transport=httpx2.MockTransport(handler)
    )


async def test_ask_sends_every_question_in_one_request_through_the_gateway() -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return answered({"ideation": 0.12, "intent": 0.03})

    client = jev_client(handler)
    probabilities = await client.ask("the transcript", QUESTIONS)
    await client.aclose()

    assert probabilities == {"ideation": 0.12, "intent": 0.03}
    [request] = requests
    assert str(request.url) == "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == GATEWAY_MODEL
    assert body["state"] == "the transcript"
    assert set(body["questions"]) == {"ideation", "intent"}


async def test_ask_retries_once_after_a_server_error() -> None:
    responses = iter([httpx2.Response(503), answered({"ideation": 0, "intent": 0})])
    client = jev_client(lambda _: next(responses))
    assert await client.ask("state", QUESTIONS) == {"ideation": 0, "intent": 0}


async def test_ask_gives_up_after_one_retry() -> None:
    calls = 0

    def handler(_: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(429, headers={"retry-after": "38"})

    with pytest.raises(TypeSafeRateLimitError):
        await jev_client(handler).ask("state", QUESTIONS)
    assert calls == 2


async def test_ask_fails_when_an_answer_is_missing() -> None:
    client = jev_client(lambda _: answered({"ideation": 0.1}))
    with pytest.raises(KeyError):
        await client.ask("state", QUESTIONS)


def test_client_keeps_request_bodies_out_of_logs_even_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    jev_client(lambda _: answered({}))
    jev_client(lambda _: answered({}))
    sdk_logger = logging.getLogger("typesafe_sdk")
    with caplog.at_level(logging.DEBUG, logger="typesafe_sdk"):
        sdk_logger.debug("POST body=%r", "I want to die")
        sdk_logger.info("POST retry 1")
    assert "die" not in caplog.text
    assert "retry 1" in caplog.text
    assert len(sdk_logger.filters) == len(set(map(type, sdk_logger.filters)))


async def test_a_timed_out_attempt_is_not_retried() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        raise httpx2.ReadTimeout("stuck", request=request)

    with pytest.raises(TypeSafeAPITimeoutError):
        await jev_client(handler).ask("state", QUESTIONS)
    assert calls == 1


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (TypeSafeAPITimeoutError(2.0), FailureReason.TIMEOUT),
        (TimeoutError(), FailureReason.TIMEOUT),
        (
            TypeSafeRateLimitError(429, None, httpx2.Headers()),
            FailureReason.RATE_LIMITED,
        ),
        (TypeSafeAPIConnectionError("down"), FailureReason.UNAVAILABLE),
        (TypeSafeAPIError(503, None, httpx2.Headers()), FailureReason.UNAVAILABLE),
        (TypeSafeAPIError(403, None, httpx2.Headers()), FailureReason.ERROR),
        (KeyError("intent"), FailureReason.ERROR),
    ],
)
def test_failure_reason_classifies_errors(
    error: BaseException, reason: FailureReason
) -> None:
    assert failure_reason(error) is reason
