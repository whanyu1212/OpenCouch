from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from services.llm.base import BaseLLMClient
from services.llm.factory import create_llm_client

LLMProvider = Literal["gemini", "openai"]

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

    llm_provider: LLMProvider = "openai"
    gemini_model: str = "gemini-3-flash-preview"
    openai_model: str = "gpt-5.4-mini"


def get_settings() -> Settings:
    """Load runtime settings from environment variables.

    Returns:
        A populated `Settings` instance.

    Raises:
        ValueError: If `LLM_PROVIDER` contains an unsupported value.
    """

    load_runtime_env()

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in {"gemini", "openai"}:
        raise ValueError(f"Unsupported LLM_PROVIDER value: {provider}")

    return Settings(
        llm_provider=provider,  # type: ignore[arg-type]
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
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
