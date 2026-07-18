"""Runtime-owned text state snapshot stores.

These stores replace legacy session snapshots for OpenCouch-owned state that is
not model-visible SDK session history: latest response metadata, diagnostics,
active exercise state, pending memory action, and other product state used by
API/CLI/debug surfaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any, Literal, Protocol, cast
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from agent.memory.hashing import iso_now
from agent.runtime.postgres import require_postgres_database_url
from agent.state import AgentState

logger = logging.getLogger(__name__)

RuntimeStateBackend = Literal["memory", "postgres"]

RUNTIME_STATE_DDL_POSTGRES = """
CREATE TABLE IF NOT EXISTS opencouch_thread_state (
    thread_id TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    value JSONB NOT NULL
);
"""

RUNTIME_STATE_INDEX_UPDATED_AT = """
CREATE INDEX IF NOT EXISTS idx_opencouch_thread_state_updated_at
    ON opencouch_thread_state(updated_at DESC);
"""


class RuntimeStateStore(Protocol):
    """Storage interface for latest text runtime state snapshots."""

    async def ensure_schema(self) -> None:
        """Create or migrate the backing store schema."""

    async def load_state(self, thread_id: str) -> AgentState | None:
        """Load the latest state snapshot for one thread."""

    async def save_state(self, thread_id: str, state: Mapping[str, Any]) -> None:
        """Persist the latest state snapshot for one thread."""

    async def delete_thread(self, thread_id: str) -> None:
        """Delete one thread's persisted state snapshot."""

    async def list_thread_ids(self, *, limit: int) -> list[str]:
        """List recently updated thread ids."""

    async def aclose(self) -> None:
        """Close store resources."""


class InMemoryRuntimeStateStore:
    """Process-local runtime state store for incognito sessions."""

    def __init__(self) -> None:
        self._states: dict[str, tuple[str, int, dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._write_sequence = 0
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("InMemoryRuntimeStateStore is closed.")

    async def ensure_schema(self) -> None:
        self._ensure_open()

    async def load_state(self, thread_id: str) -> AgentState | None:
        self._ensure_open()
        stored = self._states.get(thread_id)
        if stored is None:
            return None
        return _state_from_payload(json.loads(json.dumps(stored[2])))

    async def save_state(self, thread_id: str, state: Mapping[str, Any]) -> None:
        self._ensure_open()
        payload = json.loads(json.dumps(_state_payload(state)))
        async with self._lock:
            self._ensure_open()
            self._write_sequence += 1
            self._states[thread_id] = (
                iso_now(),
                self._write_sequence,
                payload,
            )

    async def delete_thread(self, thread_id: str) -> None:
        self._ensure_open()
        async with self._lock:
            self._ensure_open()
            self._states.pop(thread_id, None)

    async def list_thread_ids(self, *, limit: int) -> list[str]:
        self._ensure_open()
        ordered = sorted(
            self._states.items(),
            key=lambda item: (item[1][0], item[1][1]),
            reverse=True,
        )
        return [thread_id for thread_id, _ in ordered[:limit]]

    async def aclose(self) -> None:
        if self._closed:
            return
        async with self._lock:
            if self._closed:
                return
            self._states.clear()
            self._closed = True


class PostgresRuntimeStateStore:
    """PostgreSQL-backed latest-state snapshot store."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        self._connection: psycopg.AsyncConnection[dict[str, Any]] | None = None
        self._closed = False
        self._connect_lock = asyncio.Lock()

    async def _ensure_connection(self) -> psycopg.AsyncConnection[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("PostgresRuntimeStateStore is closed.")
        if self._connection is not None:
            return self._connection

        async with self._connect_lock:
            if self._closed:
                raise RuntimeError("PostgresRuntimeStateStore is closed.")
            if self._connection is not None:
                return self._connection

            conn = await psycopg.AsyncConnection.connect(
                self.dsn,
                row_factory=dict_row,
                autocommit=True,
            )
            try:
                await self._ensure_schema(conn)
            except BaseException:
                await conn.close()
                raise
            self._connection = conn
            return self._connection

    @staticmethod
    async def _ensure_schema(
        conn: psycopg.AsyncConnection[dict[str, Any]],
    ) -> None:
        async with conn.transaction():
            async with conn.cursor() as cursor:
                await cursor.execute(RUNTIME_STATE_DDL_POSTGRES)
                await cursor.execute(RUNTIME_STATE_INDEX_UPDATED_AT)

    async def ensure_schema(self) -> None:
        await self._ensure_connection()

    async def load_state(self, thread_id: str) -> AgentState | None:
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT value
                FROM opencouch_thread_state
                WHERE thread_id = %s
                """,
                (thread_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return _state_from_payload(row["value"])

    async def save_state(self, thread_id: str, state: Mapping[str, Any]) -> None:
        conn = await self._ensure_connection()
        payload = _state_payload(state)
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                INSERT INTO opencouch_thread_state(thread_id, updated_at, turn_count, value)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(thread_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    turn_count = excluded.turn_count,
                    value = excluded.value
                """,
                (
                    thread_id,
                    iso_now(),
                    _turn_count(payload),
                    Jsonb(payload),
                ),
            )

    async def delete_thread(self, thread_id: str) -> None:
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                "DELETE FROM opencouch_thread_state WHERE thread_id = %s",
                (thread_id,),
            )

    async def list_thread_ids(self, *, limit: int) -> list[str]:
        conn = await self._ensure_connection()
        async with conn.cursor() as cursor:
            await cursor.execute(
                """
                SELECT thread_id
                FROM opencouch_thread_state
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
        return [str(row["thread_id"]) for row in rows]

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            try:
                await self._connection.close()
            except Exception:
                logger.warning(
                    "PostgresRuntimeStateStore: connection close raised; ignoring",
                    exc_info=True,
                )
            finally:
                self._connection = None


def create_runtime_state_store(
    *,
    backend: RuntimeStateBackend,
    database_url: str | None,
) -> RuntimeStateStore:
    """Create a runtime-owned text state snapshot store."""

    if backend == "memory":
        return InMemoryRuntimeStateStore()
    if backend == "postgres":
        return PostgresRuntimeStateStore(require_postgres_database_url(database_url))
    raise ValueError(f"Unsupported runtime-state backend: {backend}")


def _state_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in state.items():
        if hasattr(value, "model_dump"):
            payload[str(key)] = value.model_dump(mode="json")
        else:
            payload[str(key)] = value
    return payload


def _state_from_payload(payload: Any) -> AgentState:
    from agent.models import Channel, CrisisAssessment

    raw = dict(payload or {})
    crisis = raw.get("crisis")
    if isinstance(crisis, Mapping):
        raw["crisis"] = CrisisAssessment.model_validate(crisis)
    channel = raw.get("channel")
    if isinstance(channel, str):
        try:
            raw["channel"] = Channel(channel)
        except ValueError:
            raw["channel"] = Channel.TEST
    return cast(AgentState, raw)


def _turn_count(payload: Mapping[str, Any]) -> int:
    session_progress = payload.get("session_progress", {}) or {}
    if not isinstance(session_progress, Mapping):
        return 0
    raw_turn_count = session_progress.get("turn_count", 0)
    try:
        return max(0, int(raw_turn_count or 0))
    except (TypeError, ValueError):
        return 0


__all__ = [
    "InMemoryRuntimeStateStore",
    "PostgresRuntimeStateStore",
    "RuntimeStateBackend",
    "RuntimeStateStore",
    "create_runtime_state_store",
]
