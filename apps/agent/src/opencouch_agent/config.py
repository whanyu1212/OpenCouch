"""Runtime settings, read from `OPENCOUCH_*` environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENCOUCH_",
        env_file=".env",
        extra="ignore",
    )

    model: str = "openai:gpt-5.4-mini"
    database_url: str = "postgresql://opencouch:opencouch@localhost:5432/opencouch"
    cors_origins: list[str] = ["http://localhost:3000"]
    host: str = "0.0.0.0"
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    return Settings()
