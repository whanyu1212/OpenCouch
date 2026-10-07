"""Postgres-backed session store."""

from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from psycopg import AsyncConnection
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from opencouch_agent.sessions.store import Session

# Applied on startup. Revisit migrations (e.g. Alembic) before the memory
# slice adds more tables.
SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    thread_id   TEXT PRIMARY KEY,
    -- Reserved for user identity; threads are unauthenticated for now.
    owner_id    TEXT,
    state       JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- The turn currently holding the thread, if any; see claim_turn.
    active_turn_token     TEXT,
    turn_lease_expires_at TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id          BIGSERIAL PRIMARY KEY,
    thread_id   TEXT NOT NULL REFERENCES conversations (thread_id) ON DELETE CASCADE,
    message     JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversation_messages_thread_id_idx
    ON conversation_messages (thread_id, id);
"""

_ENSURE_CONVERSATION = """
INSERT INTO conversations (thread_id) VALUES (%s)
ON CONFLICT (thread_id) DO UPDATE SET updated_at = now()
"""

# Insert-or-update in one statement, so two turns can't both claim the thread.
# The WHERE clause makes the update a no-op, returning no row, while another
# unexpired turn holds it.
_CLAIM_TURN = """
INSERT INTO conversations (thread_id, active_turn_token, turn_lease_expires_at)
VALUES (%(thread_id)s, %(token)s, now() + %(lease)s)
ON CONFLICT (thread_id) DO UPDATE
SET active_turn_token = EXCLUDED.active_turn_token,
    turn_lease_expires_at = EXCLUDED.turn_lease_expires_at,
    updated_at = now()
WHERE conversations.active_turn_token IS NULL
   OR conversations.turn_lease_expires_at < now()
RETURNING thread_id
"""

_RELEASE_TURN = """
UPDATE conversations
SET active_turn_token = NULL, turn_lease_expires_at = NULL
WHERE thread_id = %s AND active_turn_token = %s
"""


class PostgresSessionStore:
    """Session store persisting history and state in Postgres.

    Each `ModelMessage` is stored as its own row, in the JSON form produced by
    `ModelMessagesTypeAdapter`, so history can be trimmed or paged later.
    """

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create_schema(self) -> None:
        """Create the session tables if they don't exist."""
        async with self._pool.connection() as connection:
            await connection.execute(SCHEMA)

    async def load(self, thread_id: str) -> Session:
        """Load a session, returning an empty one if the thread is new."""
        async with self._pool.connection() as connection:
            state_row = await (
                await connection.execute(
                    "SELECT state FROM conversations WHERE thread_id = %s",
                    (thread_id,),
                )
            ).fetchone()
            message_rows = await (
                await connection.execute(
                    "SELECT message FROM conversation_messages"
                    " WHERE thread_id = %s ORDER BY id",
                    (thread_id,),
                )
            ).fetchall()

        history = ModelMessagesTypeAdapter.validate_python(
            [row[0] for row in message_rows]
        )
        state = state_row[0] if state_row else {}
        return Session(thread_id=thread_id, history=history, state=state)

    async def append_messages(
        self, thread_id: str, messages: Sequence[ModelMessage]
    ) -> None:
        """Append messages to a thread's history, creating the thread if needed."""
        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute(_ENSURE_CONVERSATION, (thread_id,))
            await _insert_messages(connection, thread_id, messages)

    async def save_run(
        self,
        thread_id: str,
        messages: Sequence[ModelMessage],
        state: dict[str, Any],
    ) -> None:
        """Atomically append a run's messages and replace the thread's state."""
        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO conversations (thread_id, state) VALUES (%s, %s)"
                " ON CONFLICT (thread_id)"
                " DO UPDATE SET state = EXCLUDED.state, updated_at = now()",
                (thread_id, Jsonb(state)),
            )
            await _insert_messages(connection, thread_id, messages)

    async def claim_turn(
        self, thread_id: str, lease_token: str, lease: timedelta
    ) -> bool:
        """Claim a thread for one turn; see `SessionStore.claim_turn`."""
        async with self._pool.connection() as connection:
            claimed = await (
                await connection.execute(
                    _CLAIM_TURN,
                    {"thread_id": thread_id, "token": lease_token, "lease": lease},
                )
            ).fetchone()
        return claimed is not None

    async def release_turn(self, thread_id: str, lease_token: str) -> None:
        """Release a turn claim, but only if `lease_token` still holds it."""
        async with self._pool.connection() as connection:
            await connection.execute(_RELEASE_TURN, (thread_id, lease_token))


async def _insert_messages(
    connection: AsyncConnection[Any],
    thread_id: str,
    messages: Sequence[ModelMessage],
) -> None:
    if not messages:
        return
    serialized = ModelMessagesTypeAdapter.dump_python(list(messages), mode="json")
    async with connection.cursor() as cursor:
        await cursor.executemany(
            "INSERT INTO conversation_messages (thread_id, message) VALUES (%s, %s)",
            [(thread_id, Jsonb(message)) for message in serialized],
        )
