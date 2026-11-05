"""Judge calls are bounded and fail over only for confirmed credit exhaustion."""

import asyncio
from dataclasses import replace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

from evals.judge.client import JudgeClient, JudgeError
from evals.judge.models import Route
from evals.judge.providers import JudgeRoute, build_routes


@pytest.fixture(autouse=True)
def available_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ModelResponse, "cost", Mock(return_value=Mock(total_price=0.42))
    )


class Verdict(BaseModel):
    score: int


def route(
    name: Route = "anthropic",
    *,
    error: Exception | None = None,
    served: str = "claude-opus-5-5",
    text: str = '{"score": 3}',
    finish: str | None = None,
) -> JudgeRoute:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.output_mode == "native"
        assert info.model_settings is not None
        assert info.model_settings["max_retries"] == 0
        if error:
            raise error
        response = ModelResponse(
            [TextPart(text)],
            model_name=served,
            usage=RequestUsage(
                input_tokens=11,
                output_tokens=7,
                cache_read_tokens=2,
                cache_write_tokens=1,
                output_reasoning_tokens=3,
            ),
        )
        if finish:
            from typing import cast

            from pydantic_ai.messages import FinishReason

            response.finish_reason = cast(FinishReason, finish)
        return response

    return JudgeRoute(
        name,
        FunctionModel(
            respond,
            model_name=served,
            profile=ModelProfile(supports_json_schema_output=True),
        ),
        "claude-opus-5-5",
        {"max_retries": 0},
    )


async def test_native_output_records_usage() -> None:
    result, attempts = await JudgeClient([route()], system_prompt="Judge").request(
        "x", Verdict
    )
    assert result.score == 3
    attempt = attempts[0]
    assert (attempt.input_tokens, attempt.output_tokens) == (11, 7)
    assert (
        attempt.cache_read_tokens,
        attempt.cache_write_tokens,
        attempt.reasoning_tokens,
    ) == (2, 1, 3)
    assert attempt.status == "ok"
    assert attempt.cost_usd == 0.42


async def test_confirmed_credit_errors_exhaust_both_routes_before_openai() -> None:
    routes = [
        route(
            error=ModelHTTPError(
                400, "x", {"error": {"message": "Your credit balance is too low"}}
            )
        ),
        route(
            "gateway",
            error=ModelHTTPError(402, "x", provider_error_code="insufficient_funds"),
        ),
        route("openai"),
    ]
    _, attempts = await JudgeClient(routes, system_prompt="Judge").request("x", Verdict)
    assert [attempt.status for attempt in attempts] == [
        "credit_exhausted",
        "credit_exhausted",
        "ok",
    ]
    assert attempts[-1].fallback_reason == "credit_exhausted"


@pytest.mark.parametrize(
    "status,code",
    [(400, None), (401, None), (403, None), (429, "rate_limit_exceeded"), (500, None)],
)
async def test_other_http_errors_never_fall_back(status: int, code: str | None) -> None:
    with pytest.raises(JudgeError) as caught:
        await JudgeClient(
            [
                route(error=ModelHTTPError(status, "x", provider_error_code=code)),
                route("gateway"),
            ],
            system_prompt="Judge",
        ).request("x", Verdict)
    assert len(caught.value.attempts) == 1
    assert caught.value.failure_class == f"http_{status}"


@pytest.mark.parametrize(
    "kwargs,failure",
    [
        ({"text": "invalid"}, "invalid_output"),
        ({"text": '{"score": 3}', "finish": "content_filter"}, "refusal"),
        ({"text": '{"score": 3}', "finish": "length"}, "truncation"),
        ({"served": "claude-opus-5-6"}, "served_model_mismatch"),
    ],
)
async def test_unscorable_responses_keep_usage_and_stop(
    kwargs: dict[str, str], failure: str
) -> None:
    with pytest.raises(JudgeError) as caught:
        await JudgeClient(
            [
                route(
                    text=kwargs.get("text", '{"score": 3}'),
                    finish=kwargs.get("finish"),
                    served=kwargs.get("served", "claude-opus-5-5"),
                ),
                route("gateway"),
            ],
            system_prompt="Judge",
        ).request("x", Verdict)
    assert caught.value.failure_class == failure
    assert len(caught.value.attempts) == 1
    assert caught.value.attempts[0].output_tokens == 7


async def test_cancellation_propagates() -> None:
    async def cancel(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise asyncio.CancelledError

    cancelled = replace(
        route(),
        model=FunctionModel(
            cancel, profile=ModelProfile(supports_json_schema_output=True)
        ),
    )
    with pytest.raises(asyncio.CancelledError):
        await JudgeClient([cancelled], system_prompt="Judge").request("x", Verdict)


async def test_missing_intermediate_key_stops_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    gateway = build_routes("gateway")[0]
    exhausted = route(
        error=ModelHTTPError(400, "x", {"error": {"message": "credit balance too low"}})
    )
    with pytest.raises(JudgeError) as caught:
        await JudgeClient(
            [exhausted, gateway, route("openai")], system_prompt="Judge"
        ).request("x", Verdict)
    assert caught.value.failure_class == "missing_api_key"
    assert len(caught.value.attempts) == 2


def test_factory_models_settings_and_explicit_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in ("ANTHROPIC_API_KEY", "AI_GATEWAY_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "test-key")
    routes = build_routes()
    assert [item.requested_model for item in routes] == [
        "claude-opus-5-5",
        "anthropic/claude-opus-5.5",
        "gpt-6.1-sol",
    ]
    for item in routes:
        assert item.settings["max_retries"] == 0
        assert item.settings["max_tokens"] == 4096
        assert "temperature" not in item.settings
    assert routes[1].model.profile.get("supports_json_schema_output")
    assert routes[1].model.profile.get("anthropic_supports_adaptive_thinking")
    assert len(build_routes("openai")) == 1


@pytest.mark.parametrize(
    "name,status,code",
    [
        ("gateway", 402, "quota_for_entity_exceeded"),
        ("openai", 429, "insufficient_quota"),
    ],
)
async def test_exhausted_last_route_is_unscorable(
    name: Route, status: int, code: str
) -> None:
    with pytest.raises(JudgeError) as caught:
        await JudgeClient(
            [route(name, error=ModelHTTPError(status, "x", {"error": {"type": code}}))],
            system_prompt="Judge",
        ).request("x", Verdict)
    assert caught.value.failure_class == "credit_exhausted"
    assert caught.value.attempts[0].status == "credit_exhausted"


@pytest.mark.parametrize(
    "served", ["anthropic/claude-opus-5.5", "claude-opus-5-5-20261104"]
)
async def test_canonical_spelling_and_dated_snapshot_are_allowed(served: str) -> None:
    result, _ = await JudgeClient(
        [route(served=served)], system_prompt="Judge"
    ).request("x", Verdict)
    assert result.score == 3


async def test_missing_primary_key_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(JudgeError) as caught:
        await JudgeClient(build_routes(), system_prompt="Judge").request("x", Verdict)
    assert caught.value.failure_class == "missing_api_key"
    assert len(caught.value.attempts) == 1
    assert str(caught.value) == "missing_api_key"


@pytest.mark.parametrize("status", [200, 500])
@pytest.mark.parametrize("route_name", ["anthropic", "gateway"])
async def test_sdk_anthropic_uses_native_schema_and_bounded_settings(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    route_name: Route,
) -> None:
    import json

    import httpx2
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.profiles.anthropic import anthropic_model_profile
    from pydantic_ai.providers.anthropic import AnthropicProvider

    calls: list[dict[str, object]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        calls.append(body)
        if status != 200:
            return httpx2.Response(
                status, json={"error": {"type": "api_error", "message": "test"}}
            )
        return httpx2.Response(
            200,
            json={
                "id": "message-test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": '{"score": 3}'}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    original = build_routes(route_name)[0]
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        model = AnthropicModel(
            original.requested_model,
            provider=AnthropicProvider(
                api_key="test-key",
                http_client=http,
                base_url="https://ai-gateway.vercel.sh"
                if route_name == "gateway"
                else "https://api.anthropic.com",
            ),
            profile=anthropic_model_profile("claude-opus-5-5"),
        )
        client = JudgeClient([replace(original, model=model)], system_prompt="Judge")
        if status != 200:
            with pytest.raises(JudgeError):
                await client.request("x", Verdict)
            assert len(calls) == 1
            return
        result, _ = await client.request("x", Verdict)
    assert result.score == 3
    assert len(calls) == 1
    assert calls[0]["model"] == original.requested_model
    assert calls[0]["max_tokens"] == 4096
    assert calls[0]["thinking"] == {"type": "adaptive"}
    assert "temperature" not in calls[0]
    assert "tool_choice" not in calls[0]
    assert calls[0]["output_config"] == {
        "effort": "high",
        "format": {
            "type": "json_schema",
            "schema": {
                "properties": {"score": {"type": "integer"}},
                "required": ["score"],
                "type": "object",
                "additionalProperties": False,
            },
        },
    }


@pytest.mark.parametrize("error", [ValueError("unavailable"), LookupError("unknown")])
async def test_unavailable_pricing_keeps_verdict(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(ModelResponse, "cost", Mock(side_effect=error))
    result, attempts = await JudgeClient([route()], system_prompt="Judge").request(
        "x", Verdict
    )
    assert result.score == 3
    assert attempts[0].cost_usd is None
    assert attempts[0].output_tokens == 7


@pytest.mark.parametrize(
    "name,status,body",
    [
        ("anthropic", 400, "credit balance too low"),
        ("anthropic", 400, {"error": "credit balance too low"}),
        ("anthropic", 400, {"error": {"message": 3}}),
        ("anthropic", 400, {"error": {"message": "invalid request"}}),
        ("gateway", 402, {"error": {"code": "other_error"}}),
        ("openai", 429, {"error": {"type": "rate_limit_exceeded"}}),
    ],
)
async def test_ambiguous_provider_payload_never_confirms_exhaustion(
    name: Route, status: int, body: object
) -> None:
    with pytest.raises(JudgeError) as caught:
        await JudgeClient(
            [route(name, error=ModelHTTPError(status, "x", body)), route("openai")],
            system_prompt="Judge",
        ).request("x", Verdict)
    assert caught.value.failure_class == f"http_{status}"
    assert len(caught.value.attempts) == 1


@pytest.mark.parametrize(
    "error,failure",
    [
        (TimeoutError("private detail"), "timeout"),
        (RuntimeError("private detail"), "provider_error"),
    ],
)
async def test_local_errors_are_safe_and_stop_fallback(
    error: Exception, failure: str
) -> None:
    with pytest.raises(JudgeError) as caught:
        await JudgeClient(
            [route(error=error), route("gateway")], system_prompt="Judge"
        ).request("x", Verdict)
    assert caught.value.failure_class == failure
    assert "private detail" not in str(caught.value)
    assert len(caught.value.attempts) == 1


async def test_empty_chain_is_unscorable() -> None:
    with pytest.raises(JudgeError, match="no_routes"):
        await JudgeClient([], system_prompt="Judge").request("x", Verdict)


@pytest.mark.parametrize("status", [200, 429])
async def test_sdk_openai_responses_uses_direct_native_output_and_safe_quota_error(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    import json

    import httpx2
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    calls: list[dict[str, object]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == "https://api.openai.com/v1/responses"
        calls.append(json.loads(request.content))
        if status == 429:
            return httpx2.Response(
                429,
                json={
                    "error": {
                        "type": "insufficient_quota",
                        "code": "insufficient_quota",
                        "message": "private billing detail",
                    }
                },
            )
        return httpx2.Response(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1793836800,
                "status": "completed",
                "model": "gpt-6.1-sol",
                "error": None,
                "incomplete_details": None,
                "output": [
                    {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"score": 3}',
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 9,
                    "total_tokens": 29,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens_details": {"reasoning_tokens": 3},
                },
                "parallel_tool_calls": False,
                "tools": [],
                "tool_choice": "auto",
            },
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://unexpected.example/v1")
    original = build_routes("openai")[0]
    assert isinstance(original.model, OpenAIResponsesModel)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        sdk = original.model.client.with_options(http_client=http)
        model = OpenAIResponsesModel(
            original.requested_model, provider=OpenAIProvider(openai_client=sdk)
        )
        client = JudgeClient([replace(original, model=model)], system_prompt="Judge")
        if status == 429:
            with pytest.raises(JudgeError) as caught:
                await client.request("x", Verdict)
            assert caught.value.failure_class == "credit_exhausted"
            assert len(caught.value.attempts) == 1
            assert caught.value.attempts[0].status == "credit_exhausted"
            assert "private billing detail" not in str(caught.value)
            assert (
                "private billing detail"
                not in caught.value.attempts[0].model_dump_json()
            )
        else:
            result, attempts = await client.request("x", Verdict)
            assert result.score == 3
            assert len(attempts) == 1
            attempt = attempts[0]
            assert attempt.served_model == "gpt-6.1-sol"
            assert attempt.status == "ok"
            assert (attempt.input_tokens, attempt.output_tokens) == (20, 9)
            assert (attempt.cache_read_tokens, attempt.reasoning_tokens) == (4, 3)
    assert len(calls) == 1
    assert calls[0]["model"] == "gpt-6.1-sol"
    assert calls[0]["reasoning"] == {"effort": "high", "context": "all_turns"}
    assert calls[0]["max_output_tokens"] == 4096
    assert "temperature" not in calls[0]
    assert calls[0].get("tool_choice", "auto") == "auto"
    assert not calls[0].get("tools")
    assert calls[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "Verdict",
            "strict": True,
            "schema": {
                "properties": {"score": {"type": "integer"}},
                "required": ["score"],
                "type": "object",
                "additionalProperties": False,
            },
        }
    }
