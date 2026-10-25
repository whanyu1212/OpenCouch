"""Runtime settings, read from `OPENCOUCH_*` environment variables."""

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings, loaded from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="OPENCOUCH_",
        env_file=".env",
        extra="ignore",
    )

    model: str = "openai:gpt-6.1-sol"
    # Unset means an in-memory session store: history is lost on restart.
    # DATABASE_URL is what a Neon database added through Vercel sets.
    database_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENCOUCH_DATABASE_URL", "DATABASE_URL"),
    )
    cors_origins: list[str] = ["http://localhost:3000"]
    # Hard limit on the agent's reply, including streaming it. Screening runs
    # first, under its own 3-second deadline.
    turn_timeout_seconds: float = 90.0
    # Jev is called through Vercel AI Gateway; the evals read the same variable.
    ai_gateway_api_key: SecretStr | None = Field(
        default=None, validation_alias="AI_GATEWAY_API_KEY"
    )
    # Local runs without a key only, and only with the test model.
    fake_jev: bool = False
    # Where sign-ins are checked (HY-207). NEON_AUTH_BASE_URL is what Neon
    # sets when Neon Auth is enabled; the web app reads the same variable.
    neon_auth_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "OPENCOUCH_NEON_AUTH_BASE_URL", "NEON_AUTH_BASE_URL"
        ),
    )
    # Local runs without Neon Auth only, and only with the test model.
    fake_auth: bool = False
    host: str = "0.0.0.0"
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()
