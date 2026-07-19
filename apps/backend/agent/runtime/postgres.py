"""Postgres configuration validation helpers for runtime-owned stores."""

from __future__ import annotations

from config import MISSING_MEMORY_DATABASE_URL_MESSAGE


def require_postgres_database_url(database_url: str | None) -> str:
    """Return a Postgres DSN or raise the shared operator-facing error."""

    if not database_url:
        raise ValueError(MISSING_MEMORY_DATABASE_URL_MESSAGE)
    return database_url
