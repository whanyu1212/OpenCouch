"""Tests for runtime configuration env parsing."""

from __future__ import annotations

import pytest

import config


def test_get_settings_defaults_to_postgres_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset persistence env vars resolve to the Postgres backend with a null URL.

    The postgres-without-URL combination is rejected later by the runtime
    constructor (see `validate_thread_checkpointer_config`); `get_settings`
    itself returns a DTO and does not enforce the URL contract.
    """

    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.delenv("OPENCOUCH_PERSISTENCE_BACKEND", raising=False)
    monkeypatch.delenv("OPENCOUCH_MEMORY_DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENCOUCH_TEXT_SESSION_BACKEND", raising=False)
    monkeypatch.delenv("OPENCOUCH_TEXT_SESSION_DATABASE_URL", raising=False)

    settings = config.get_settings()

    assert settings.persistence_backend == "postgres"
    assert settings.memory_database_url is None
    assert settings.text_session_backend == "auto"
    assert settings.text_session_database_url is None


def test_get_settings_reads_sqlite_backend_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite remains an explicit fallback via the env override."""

    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("OPENCOUCH_PERSISTENCE_BACKEND", "sqlite")
    monkeypatch.delenv("OPENCOUCH_MEMORY_DATABASE_URL", raising=False)

    settings = config.get_settings()

    assert settings.persistence_backend == "sqlite"
    assert settings.memory_database_url is None


def test_get_settings_reads_postgres_backend_and_database_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured Postgres env vars should round-trip into Settings."""

    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("OPENCOUCH_PERSISTENCE_BACKEND", "postgres")
    monkeypatch.setenv(
        "OPENCOUCH_MEMORY_DATABASE_URL",
        "postgresql://opencouch:opencouch@postgres:5432/opencouch",
    )

    settings = config.get_settings()

    assert settings.persistence_backend == "postgres"
    assert settings.memory_database_url == (
        "postgresql://opencouch:opencouch@postgres:5432/opencouch"
    )


def test_get_settings_reads_text_session_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI SDK session storage can be enabled independently."""

    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("OPENCOUCH_TEXT_SESSION_BACKEND", "sqlalchemy")
    monkeypatch.setenv(
        "OPENCOUCH_TEXT_SESSION_DATABASE_URL",
        "postgresql+asyncpg://opencouch:opencouch@postgres:5432/opencouch",
    )

    settings = config.get_settings()

    assert settings.text_session_backend == "sqlalchemy"
    assert settings.text_session_database_url == (
        "postgresql+asyncpg://opencouch:opencouch@postgres:5432/opencouch"
    )


def test_get_settings_rejects_invalid_text_session_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported OpenAI SDK session backends should raise eagerly."""

    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("OPENCOUCH_TEXT_SESSION_BACKEND", "bogus")

    with pytest.raises(ValueError, match="OPENCOUCH_TEXT_SESSION_BACKEND"):
        config.get_settings()


def test_get_settings_rejects_invalid_persistence_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported persistence backend values should raise eagerly."""

    monkeypatch.setattr(config, "_DOTENV_LOADED", True)
    monkeypatch.setenv("OPENCOUCH_PERSISTENCE_BACKEND", "bogus")

    with pytest.raises(ValueError, match="OPENCOUCH_PERSISTENCE_BACKEND"):
        config.get_settings()
