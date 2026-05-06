"""Runtime configuration helpers for model and tracing setup."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from llm.base import BaseLLMClient
from llm.factory import create_llm_client
from llm.google_genai import DEFAULT_GEMINI_MODEL
from llm.openai_client import DEFAULT_OPENAI_MODEL

LLMProvider = Literal["gemini", "openai"]
ResponseModelTier = Literal["fast", "quality"]
PersistenceBackend = Literal["sqlite", "postgres"]

# Single source of truth for the default provider when LLM_PROVIDER is unset.
DEFAULT_LLM_PROVIDER: LLMProvider = "openai"
DEFAULT_OPENAI_QUALITY_MODEL = "gpt-5.4"
# Postgres is the default persistent backend (Docker compose ships it as the
# primary persistence path). SQLite remains available as an explicit fallback
# via OPENCOUCH_PERSISTENCE_BACKEND=sqlite for local-only installs without
# Docker.
DEFAULT_PERSISTENCE_BACKEND: PersistenceBackend = "postgres"

# Shared, actionable error text raised by every postgres-without-URL guard
# in the runtime. Lives here so the message stays consistent across the
# checkpointer and voice-finalization validators that reference it.
MISSING_MEMORY_DATABASE_URL_MESSAGE = (
    "OPENCOUCH_PERSISTENCE_BACKEND=postgres requires "
    "OPENCOUCH_MEMORY_DATABASE_URL. Add it to your .env — for the "
    "local docker compose stack use "
    "postgresql://opencouch:opencouch@localhost:5432/opencouch "
    "(or @postgres:5432/opencouch from inside the compose network). "
    "For a local-only install without docker, set "
    "OPENCOUCH_PERSISTENCE_BACKEND=sqlite instead."
)

_DOTENV_LOADED = False


def load_runtime_env() -> None:
    """Load local `.env` files for runtime configuration.

    Returns:
        None.
    """

    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return

    backend_root = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[2]

    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(repo_root / ".env.local", override=False)
    load_dotenv(backend_root / ".env", override=False)
    load_dotenv(backend_root / ".env.local", override=False)
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
    persistence_backend: PersistenceBackend = DEFAULT_PERSISTENCE_BACKEND
    memory_database_url: str | None = None
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
        ValueError: If `LLM_PROVIDER` or `OPENCOUCH_PERSISTENCE_BACKEND`
            contains an unsupported value. Postgres-without-URL is validated
            downstream by the runtime constructor, where the error can be
            surfaced with full context.
    """

    load_runtime_env()

    provider = _read_provider_env("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    response_fast_provider = _read_provider_env("RESPONSE_FAST_LLM_PROVIDER", provider)
    response_quality_provider = _read_provider_env(
        "RESPONSE_QUALITY_LLM_PROVIDER",
        provider,
    )
    persistence_backend = _read_persistence_backend_env(
        "OPENCOUCH_PERSISTENCE_BACKEND",
        DEFAULT_PERSISTENCE_BACKEND,
    )

    return Settings(
        llm_provider=provider,
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        response_fast_provider=response_fast_provider,
        response_fast_gemini_model=os.getenv(
            "RESPONSE_FAST_GEMINI_MODEL",
            os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        ),
        response_fast_openai_model=os.getenv(
            "RESPONSE_FAST_OPENAI_MODEL",
            os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        ),
        response_quality_provider=response_quality_provider,
        response_quality_gemini_model=os.getenv(
            "RESPONSE_QUALITY_GEMINI_MODEL",
            os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        ),
        response_quality_openai_model=os.getenv(
            "RESPONSE_QUALITY_OPENAI_MODEL",
            DEFAULT_OPENAI_QUALITY_MODEL,
        ),
        persistence_backend=persistence_backend,
        memory_database_url=os.getenv("OPENCOUCH_MEMORY_DATABASE_URL"),
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


def _read_persistence_backend_env(
    name: str,
    fallback: PersistenceBackend,
) -> PersistenceBackend:
    """Read and validate a persistence backend env var.

    Args:
        name (str): Environment variable name to read.
        fallback (PersistenceBackend): Backend used when the variable is unset.

    Returns:
        PersistenceBackend: Validated backend literal.

    Raises:
        ValueError: If the environment value is not supported.
    """

    raw = os.getenv(name, fallback).strip().lower()
    if raw == "sqlite":
        return "sqlite"
    if raw == "postgres":
        return "postgres"
    raise ValueError(f"Unsupported {name} value: {raw}")


def _read_provider_env(name: str, fallback: LLMProvider) -> LLMProvider:
    """Read and validate an LLM provider env var.

    Args:
        name: Environment variable name to read.
        fallback: Provider used when the variable is unset.

    Returns:
        Validated provider literal.

    Raises:
        ValueError: If the environment value is not supported.
    """

    raw = os.getenv(name, fallback).strip().lower()
    if raw == "gemini":
        return "gemini"
    if raw == "openai":
        return "openai"
    raise ValueError(f"Unsupported {name} value: {raw}")


def _read_bool_env(name: str) -> bool:
    """Read a boolean feature flag from the environment.

    Args:
        name (str): Environment variable name to read.

    Returns:
        bool: True when the variable is set to a truthy value.
    """

    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, fallback: int) -> int:
    """Read an integer from the environment.

    Args:
        name (str): Environment variable name to read.
        fallback (int): Value used when the variable is unset.

    Returns:
        int: Parsed integer value.

    Raises:
        ValueError: If the environment value is not an integer.
    """

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return fallback
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Unsupported {name} value: {raw}") from exc


def _resolve_model_for_provider(
    *,
    provider: LLMProvider,
    gemini_model: str,
    openai_model: str,
) -> str:
    """Return the provider-specific model string.

    Args:
        provider: LLM provider selected for the client.
        gemini_model: Gemini model name to use when provider is Gemini.
        openai_model: OpenAI model name to use when provider is OpenAI.

    Returns:
        Model name for the selected provider.
    """

    return gemini_model if provider == "gemini" else openai_model


def create_configured_control_llm_client(
    settings: Settings | None = None,
) -> BaseLLMClient:
    """Create the pinned control-plane LLM client for the runtime.

    Args:
        settings: Optional preloaded settings. Reads the environment
            when omitted.

    Returns:
        Configured control-plane LLM client.
    """

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
    """Create the response-writer LLM client for a user-facing tier.

    Args:
        tier: Response model tier to configure.
        settings: Optional preloaded settings. Reads the environment
            when omitted.

    Returns:
        Configured response-writer LLM client.
    """

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
    """Create both response-tier clients for reuse in the API server.

    Args:
        settings: Optional preloaded settings. Reads the environment
            when omitted.

    Returns:
        Response-tier clients keyed by tier.
    """

    settings = settings or get_settings()
    return {
        "fast": create_configured_response_llm_client("fast", settings),
        "quality": create_configured_response_llm_client("quality", settings),
    }


def create_configured_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Return the configured control-plane client.

    Args:
        settings: Optional preloaded settings. Reads the environment
            when omitted.

    Returns:
        Configured control-plane LLM client.
    """

    return create_configured_control_llm_client(settings)
