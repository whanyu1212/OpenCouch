"""Active-session lifecycle: durable state, store backends, manager.

The active-session subsystem owns the runtime-side bookkeeping that
sits alongside LangGraph's checkpointer: which thread is currently
mid-session, when it was last touched, what mutation token it carries,
and how that state survives a process restart. Three modules carry the
load:

- ``manager``: the in-process owner that orchestrates token rotation,
  expiry checks, and acquisition of session leases.
- ``store``: the persistence Protocol plus the Postgres backend.
- ``sqlite_store``: the SQLite backend that reuses the checkpointer
  connection.

The package is the public surface — callers should ``from
agent.runtime.active_session import X`` rather than reaching into
sibling modules directly.
"""

from __future__ import annotations

from agent.runtime.active_session.manager import (
    ActiveSessionManager,
    PersistedActiveSessionRow,
    PersistedActiveSessionState,
    parse_iso_timestamp,
)
from agent.runtime.active_session.sqlite_store import SqliteActiveSessionStore
from agent.runtime.active_session.store import (
    ACTIVE_SESSION_EXTRA_COLUMNS,
    ACTIVE_SESSION_STATE_DDL,
    ActiveSessionStore,
    PostgresActiveSessionStore,
)

__all__ = [
    "ACTIVE_SESSION_EXTRA_COLUMNS",
    "ACTIVE_SESSION_STATE_DDL",
    "ActiveSessionManager",
    "ActiveSessionStore",
    "PersistedActiveSessionRow",
    "PersistedActiveSessionState",
    "PostgresActiveSessionStore",
    "SqliteActiveSessionStore",
    "parse_iso_timestamp",
]
