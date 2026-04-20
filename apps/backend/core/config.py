from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from services.llm.base import BaseLLMClient
from services.llm.factory import create_llm_client
from services.llm.google_genai import DEFAULT_GEMINI_MODEL
from services.llm.openai_client import DEFAULT_OPENAI_MODEL

LLMProvider = Literal["gemini", "openai"]
ResponseModelTier = Literal["fast", "quality"]

# Single source of truth for the default provider when LLM_PROVIDER
# env var is unset. Used by both the Settings dataclass default and
# get_settings() fallback so they always agree.
DEFAULT_LLM_PROVIDER: LLMProvider = "openai"
DEFAULT_OPENAI_QUALITY_MODEL = "gpt-5.4"

_DOTENV_LOADED = False


def load_runtime_env() -> None:
    """Load local `.env` files for runtime configuration.

    Returns:
        None.
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    backend_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]

    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(backend_root / ".env", override=False)
    _DOTENV_LOADED = True


@dataclass(slots=True)
class Settings:
    """Typed runtime configuration for application wiring.

    Keep this focused on provider selection and model defaults, not domain policy.
    """

    llm_provider: LLMProvider = DEFAULT_LLM_PROVIDER
    gemini_model: str = DEFAULT_GEMINI_MODEL
    openai_model: str = DEFAULT_OPENAI_MODEL
    response_fast_provider: LLMProvider = DEFAULT_LLM_PROVIDER
    response_fast_gemini_model: str = DEFAULT_GEMINI_MODEL
    response_fast_openai_model: str = DEFAULT_OPENAI_MODEL
    response_quality_provider: LLMProvider = DEFAULT_LLM_PROVIDER
    response_quality_gemini_model: str = DEFAULT_GEMINI_MODEL
    response_quality_openai_model: str = DEFAULT_OPENAI_QUALITY_MODEL
    langsmith_tracing: bool = False
    langsmith_endpoint: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str | None = None
    langchain_tracing_v2: bool = False
    langchain_endpoint: str | None = None
    langchain_api_key: str | None = None
    langchain_project: str | None = None


def get_settings() -> Settings:
    """Load runtime settings from environment variables.

    Returns:
        A populated `Settings` instance.

    Raises:
        ValueError: If `LLM_PROVIDER` contains an unsupported value.
    """

    load_runtime_env()

    provider = _read_provider_env("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    response_fast_provider = _read_provider_env("RESPONSE_FAST_LLM_PROVIDER", provider)
    response_quality_provider = _read_provider_env(
        "RESPONSE_QUALITY_LLM_PROVIDER",
        provider,
    )

    return Settings(
        llm_provider=provider,  # type: ignore[arg-type]
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        response_fast_provider=response_fast_provider,  # type: ignore[arg-type]
        response_fast_gemini_model=os.getenv(
            "RESPONSE_FAST_GEMINI_MODEL",
            os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        ),
        response_fast_openai_model=os.getenv(
            "RESPONSE_FAST_OPENAI_MODEL",
            os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        ),
        response_quality_provider=response_quality_provider,  # type: ignore[arg-type]
        response_quality_gemini_model=os.getenv(
            "RESPONSE_QUALITY_GEMINI_MODEL",
            os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        ),
        response_quality_openai_model=os.getenv(
            "RESPONSE_QUALITY_OPENAI_MODEL",
            DEFAULT_OPENAI_QUALITY_MODEL,
        ),
        langsmith_tracing=os.getenv("LANGSMITH_TRACING", "").strip().lower()
        in {"1", "true", "yes", "on"},
        langsmith_endpoint=os.getenv("LANGSMITH_ENDPOINT"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY"),
        langsmith_project=os.getenv("LANGSMITH_PROJECT"),
        langchain_tracing_v2=os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
        in {"1", "true", "yes", "on"},
        langchain_endpoint=os.getenv("LANGCHAIN_ENDPOINT"),
        langchain_api_key=os.getenv("LANGCHAIN_API_KEY"),
        langchain_project=os.getenv("LANGCHAIN_PROJECT"),
    )


def _read_provider_env(name: str, fallback: LLMProvider) -> LLMProvider:
    """Read and validate an LLM provider env var."""

    raw = os.getenv(name, fallback).strip().lower()
    if raw not in {"gemini", "openai"}:
        raise ValueError(f"Unsupported {name} value: {raw}")
    return raw  # type: ignore[return-value]


def _resolve_model_for_provider(
    *,
    provider: LLMProvider,
    gemini_model: str,
    openai_model: str,
) -> str:
    """Return the provider-specific model string."""

    return gemini_model if provider == "gemini" else openai_model


def create_configured_control_llm_client(
    settings: Settings | None = None,
) -> BaseLLMClient:
    """Create the pinned control-plane LLM client for the runtime."""

    load_runtime_env()
    settings = settings or get_settings()
    model = _resolve_model_for_provider(
        provider=settings.llm_provider,
        gemini_model=settings.gemini_model,
        openai_model=settings.openai_model,
    )
    return create_llm_client(
        provider=settings.llm_provider,
        model=model,
    )


def create_configured_response_llm_client(
    tier: ResponseModelTier,
    settings: Settings | None = None,
) -> BaseLLMClient:
    """Create the response-writer LLM client for a user-facing tier."""

    load_runtime_env()
    settings = settings or get_settings()
    if tier == "fast":
        provider = settings.response_fast_provider
        model = _resolve_model_for_provider(
            provider=provider,
            gemini_model=settings.response_fast_gemini_model,
            openai_model=settings.response_fast_openai_model,
        )
    else:
        provider = settings.response_quality_provider
        model = _resolve_model_for_provider(
            provider=provider,
            gemini_model=settings.response_quality_gemini_model,
            openai_model=settings.response_quality_openai_model,
        )

    return create_llm_client(provider=provider, model=model)


def create_configured_response_llm_clients(
    settings: Settings | None = None,
) -> dict[ResponseModelTier, BaseLLMClient]:
    """Create both response-tier clients for reuse in the API server."""

    settings = settings or get_settings()
    return {
        "fast": create_configured_response_llm_client("fast", settings),
        "quality": create_configured_response_llm_client("quality", settings),
    }


def create_configured_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Backward-compatible alias for the pinned control-plane client."""

    return create_configured_control_llm_client(settings)
