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

# Single source of truth for the default provider when LLM_PROVIDER
# env var is unset. Used by both the Settings dataclass default and
# get_settings() fallback so they always agree.
DEFAULT_LLM_PROVIDER: LLMProvider = "openai"

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

    provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
    if provider not in {"gemini", "openai"}:
        raise ValueError(f"Unsupported LLM_PROVIDER value: {provider}")

    return Settings(
        llm_provider=provider,  # type: ignore[arg-type]
        gemini_model=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        openai_model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
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


def create_configured_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """Create the configured LLM client for the current runtime.

    Args:
        settings: Optional explicit settings object. If omitted, settings are loaded
            from the current environment.

    Returns:
        A configured provider client that implements `BaseLLMClient`.
    """

    load_runtime_env()
    settings = settings or get_settings()
    model = (
        settings.gemini_model
        if settings.llm_provider == "gemini"
        else settings.openai_model
    )
    return create_llm_client(
        provider=settings.llm_provider,
        model=model,
    )
