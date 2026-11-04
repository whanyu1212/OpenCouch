"""Explicit, credit-only judge provider routing and bounded model settings."""

import os
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from evals.judge.models import Route


class MissingKeyError(RuntimeError):
    """A configured route has no credential available."""


@dataclass(frozen=True)
class JudgeRoute:
    """One explicitly configured provider and its requested model version."""

    route: Route
    model: Model
    requested_model: str
    settings: ModelSettings


def _missing_key(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise MissingKeyError("Judge route credential is missing")


def _route(route: Route, openai_model: str) -> JudgeRoute:
    requested = {
        "anthropic": "claude-opus-5-5",
        "gateway": "anthropic/claude-opus-5.5",
        "openai": openai_model,
    }[route]
    key_name = {
        "anthropic": "ANTHROPIC_API_KEY",
        "gateway": "AI_GATEWAY_API_KEY",
        "openai": "OPENAI_API_KEY",
    }[route]
    settings: ModelSettings
    if route == "openai":
        settings = OpenAIResponsesModelSettings(
            max_tokens=4096,
            timeout=120,
            max_retries=0,
            openai_reasoning_effort="high",
        )
    else:
        settings = AnthropicModelSettings(
            max_tokens=4096,
            timeout=120,
            max_retries=0,
            anthropic_thinking={"type": "adaptive"},
            anthropic_effort="high",
        )
    key = os.environ.get(key_name)
    model: Model
    if not key:
        model = FunctionModel(_missing_key)
    elif route == "openai":
        model = OpenAIResponsesModel(
            requested,
            provider=OpenAIProvider(api_key=key, base_url="https://api.openai.com/v1"),
        )
    else:
        provider = AnthropicProvider(
            api_key=key,
            base_url="https://ai-gateway.vercel.sh"
            if route == "gateway"
            else "https://api.anthropic.com",
        )
        model = AnthropicModel(
            requested,
            provider=provider,
            profile=anthropic_model_profile("claude-opus-5-5"),
        )
    return JudgeRoute(route, model, requested, settings)


def build_routes(
    route: Route = "anthropic", openai_model: str = "gpt-6.1-sol"
) -> list[JudgeRoute]:
    """Build the ordered chain, reading credentials only when called.

    Missing optional credentials stop fallback when that route is reached.
    """
    chain: Sequence[Route] = ("anthropic", "gateway", "openai")
    return [_route(item, openai_model) for item in chain[chain.index(route) :]]
