"""Bounded structured judge calls with auditable, credit-only failover."""

import re
import time
from collections.abc import Sequence
from contextlib import suppress

from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from evals.judge.models import JudgeAttempt
from evals.judge.providers import JudgeRoute, MissingKeyError


class JudgeError(RuntimeError):
    """A safe, unscorable judge failure with all attempted routes attached."""

    def __init__(self, failure_class: str, attempts: list[JudgeAttempt]) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class
        self.attempts = attempts


class _ResponseError(RuntimeError):
    pass


def _same_model(requested: str, served: str | None) -> bool:
    expected = requested.removeprefix("anthropic/").replace("5.5", "5-5")
    actual = (served or "").removeprefix("anthropic/").replace("5.5", "5-5")
    return bool(
        re.fullmatch(re.escape(expected) + r"(?:-\d{4}-?\d{2}-?\d{2})?", actual)
    )


def _credit_exhausted(error: ModelHTTPError, route: JudgeRoute) -> bool:
    # Provider payloads are untrusted; inspect only known fields and never save them.
    body = error.body
    details = body.get("error", body) if isinstance(body, dict) else {}
    if not isinstance(details, dict):
        details = {}
    codes = {error.provider_error_code, error.provider_error_type}
    codes.update(
        value for key in ("code", "type") if isinstance(value := details.get(key), str)
    )
    if route.route == "gateway":
        return error.status_code == 402 and bool(
            codes & {"insufficient_funds", "quota_for_entity_exceeded"}
        )
    if route.route == "openai":
        return error.status_code == 429 and "insufficient_quota" in codes
    message = details.get("message", "")
    return (
        error.status_code == 400
        and isinstance(message, str)
        and any(
            phrase in message.lower()
            for phrase in (
                "credit balance is too low",
                "credit balance too low",
                "spend limit has been exceeded",
                "spend limit exceeded",
            )
        )
    )


class _Recorder(WrapperModel):
    def __init__(self, route: JudgeRoute, attempt: JudgeAttempt) -> None:
        super().__init__(route.model)
        self.attempt = attempt

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )
        attempt = self.attempt
        attempt.served_model = response.model_name
        usage = response.usage
        attempt.input_tokens = usage.input_tokens
        attempt.output_tokens = usage.output_tokens
        attempt.cache_read_tokens = usage.cache_read_tokens
        attempt.cache_write_tokens = usage.cache_write_tokens
        attempt.reasoning_tokens = usage.output_reasoning_tokens
        if response.model_name:
            with suppress(ValueError, LookupError):
                attempt.cost_usd = float(response.cost().total_price)
        if not _same_model(attempt.requested_model, response.model_name):
            raise _ResponseError("served_model_mismatch")
        if response.finish_reason == "content_filter" or (
            response.provider_details and response.provider_details.get("refusal")
        ):
            raise _ResponseError("refusal")
        if response.finish_reason == "length":
            raise _ResponseError("truncation")
        return response


def _failure(error: Exception, route: JudgeRoute) -> str:
    if isinstance(error, ModelHTTPError):
        if _credit_exhausted(error, route):
            return "credit_exhausted"
        return f"http_{error.status_code}"
    if isinstance(error, MissingKeyError):
        return "missing_api_key"
    if isinstance(error, _ResponseError):
        return str(error)
    if isinstance(error, UnexpectedModelBehavior):
        return "invalid_output"
    if isinstance(error, TimeoutError):
        return "timeout"
    return "provider_error"


class JudgeClient:
    """Request one validated verdict, falling back only after confirmed exhaustion."""

    def __init__(self, routes: Sequence[JudgeRoute], *, system_prompt: str) -> None:
        self.routes = tuple(routes)
        self.system_prompt = system_prompt

    async def request[OutputT: BaseModel](
        self, prompt: str, output_type: type[OutputT]
    ) -> tuple[OutputT, list[JudgeAttempt]]:
        """Return a typed verdict and safe records, or raise an unscorable error."""
        attempts: list[JudgeAttempt] = []
        for route in self.routes:
            attempt = JudgeAttempt(
                route=route.route,
                requested_model=route.requested_model,
                fallback_reason="credit_exhausted" if attempts else None,
            )
            attempts.append(attempt)
            start = time.perf_counter()
            try:
                agent = Agent(
                    _Recorder(route, attempt),
                    output_type=NativeOutput(output_type),
                    system_prompt=self.system_prompt,
                    retries=0,
                    model_settings=route.settings,
                )
                result = await agent.run(
                    prompt, usage_limits=UsageLimits(request_limit=1)
                )
            except Exception as error:
                failure = _failure(error, route)
                attempt.failure_class = failure
                attempt.status = (
                    "credit_exhausted" if failure == "credit_exhausted" else "error"
                )
                if failure != "credit_exhausted":
                    raise JudgeError(failure, attempts) from None
            else:
                return result.output, attempts
            finally:
                attempt.elapsed_ms = round((time.perf_counter() - start) * 1000)
        raise JudgeError("credit_exhausted" if attempts else "no_routes", attempts)
