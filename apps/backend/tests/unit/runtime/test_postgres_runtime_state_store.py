from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import psycopg
import pytest

import agent.runtime.state_store as runtime_state_store_module
from agent.runtime.state_store import PostgresRuntimeStateStore


def _connection(
    *, execute_error: Exception | None = None
) -> tuple[MagicMock, AsyncMock]:
    cursor = MagicMock()
    cursor.execute = AsyncMock(side_effect=execute_error)
    cursor_context = MagicMock()
    cursor_context.__aenter__ = AsyncMock(return_value=cursor)
    cursor_context.__aexit__ = AsyncMock(return_value=False)

    connection = MagicMock()
    connection.cursor.return_value = cursor_context
    connection.close = AsyncMock()
    return connection, cursor.execute


@pytest.mark.asyncio
async def test_save_state_reconnects_after_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_connection, failed_execute = _connection(
        execute_error=psycopg.OperationalError("connection dropped")
    )
    replacement_connection, replacement_execute = _connection()
    connect = AsyncMock(side_effect=[failed_connection, replacement_connection])
    monkeypatch.setattr(
        runtime_state_store_module.psycopg.AsyncConnection,
        "connect",
        connect,
    )

    store = PostgresRuntimeStateStore("postgresql://test")
    ensure_schema = AsyncMock()
    monkeypatch.setattr(store, "_ensure_schema", ensure_schema)

    with pytest.raises(psycopg.OperationalError, match="connection dropped"):
        await store.save_state("thread-1", {"session_progress": {"turn_count": 1}})

    assert store._connection is None  # noqa: SLF001
    failed_execute.assert_awaited_once()
    failed_connection.close.assert_awaited_once()

    await store.save_state("thread-1", {"session_progress": {"turn_count": 1}})

    assert store._connection is replacement_connection  # noqa: SLF001
    assert connect.await_count == 2
    assert ensure_schema.await_count == 2
    replacement_execute.assert_awaited_once()

    await store.aclose()
