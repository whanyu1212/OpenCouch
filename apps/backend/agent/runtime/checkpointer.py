"""Checkpointer factory helpers for the persistent agent runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import MISSING_MEMORY_DATABASE_URL_MESSAGE

ThreadPersistenceBackend = Literal["sqlite", "postgres"]
Checkpointer = AsyncSqliteSaver | AsyncPostgresSaver

ALLOWED_MSGPACK_MODULES = [
    ("agent.models", "Channel"),
    ("agent.models", "CrisisAssessment"),
    ("agent.models", "ResponseCategory"),
]


def validate_thread_checkpointer_config(
    *,
    thread_persistence_backend: ThreadPersistenceBackend,
    thread_database_url: str | None,
) -> None:
    """Validate thread-checkpointer configuration.

    Args:
        thread_persistence_backend (ThreadPersistenceBackend): Checkpointer
            backend to use.
        thread_database_url (str | None): PostgreSQL checkpoint URL.

    Raises:
        ValueError: If PostgreSQL checkpoints are selected without a database
            URL.
    """

    if thread_persistence_backend == "postgres" and not thread_database_url:
        raise ValueError(MISSING_MEMORY_DATABASE_URL_MESSAGE)


def build_checkpoint_serializer() -> JsonPlusSerializer:
    """Build the serializer used by LangGraph checkpoints.

    Returns:
        JsonPlusSerializer: Serializer configured with allowed msgpack modules.
    """

    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def open_checkpointer(
    *,
    thread_persistence_backend: ThreadPersistenceBackend,
    sqlite_path: Path,
    thread_database_url: str | None,
) -> AsyncIterator[Checkpointer]:
    """Open a configured LangGraph checkpointer.

    Args:
        thread_persistence_backend (ThreadPersistenceBackend): Checkpointer
            backend to use.
        sqlite_path (Path): SQLite checkpoint path, or ``Path(":memory:")``.
        thread_database_url (str | None): PostgreSQL checkpoint URL.

    Yields:
        Checkpointer: Open LangGraph checkpointer.

    Raises:
        ValueError: If PostgreSQL checkpoints are selected without a database
            URL.
    """

    if thread_persistence_backend == "sqlite" and sqlite_path != Path(":memory:"):
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    serializer = build_checkpoint_serializer()
    if thread_persistence_backend == "postgres":
        if not thread_database_url:
            raise ValueError(MISSING_MEMORY_DATABASE_URL_MESSAGE)
        context_manager = AsyncPostgresSaver.from_conn_string(
            thread_database_url,
            serde=serializer,
        )
    else:
        context_manager = AsyncSqliteSaver.from_conn_string(str(sqlite_path))

    async with context_manager as checkpointer:
        cast(Any, checkpointer).serde = serializer
        yield checkpointer
