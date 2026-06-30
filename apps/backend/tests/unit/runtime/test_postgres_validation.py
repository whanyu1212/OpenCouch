"""Tests for shared Postgres runtime validation helpers."""

from __future__ import annotations

import pytest

from agent.runtime.postgres import require_postgres_database_url
from config import MISSING_MEMORY_DATABASE_URL_MESSAGE


def test_require_postgres_database_url_returns_configured_url() -> None:
    dsn = "postgresql://opencouch:opencouch@localhost:5432/opencouch"

    assert require_postgres_database_url(dsn) == dsn


@pytest.mark.parametrize("database_url", [None, ""])
def test_require_postgres_database_url_raises_shared_message(
    database_url: str | None,
) -> None:
    with pytest.raises(ValueError, match="OPENCOUCH_MEMORY_DATABASE_URL") as exc_info:
        require_postgres_database_url(database_url)

    assert str(exc_info.value) == MISSING_MEMORY_DATABASE_URL_MESSAGE
