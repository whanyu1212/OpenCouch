"""Active-session lifecycle: durable state, store backends, manager.

The active-session subsystem owns the runtime-side bookkeeping that
sits alongside OpenAI SDK session history and runtime state snapshots: which
thread is currently mid-session, when it was last touched, what mutation token
it carries, and how that state survives a process restart. Three modules carry
the load:

- ``manager``: the in-process owner of durable mutation coordination and
  active-session row mechanics.
- ``store``: the persistence Protocol plus ephemeral and Postgres backends.

The package is the public surface — callers should ``from
agent.runtime.session.active_session import X`` rather than reaching into
sibling modules directly.
"""

from __future__ import annotations

from agent.runtime.session.manager import (
    ActiveSessionManager,
    PersistedActiveSessionRow,
    PersistedActiveSessionState,
)
from agent.runtime.session.state import parse_iso_timestamp
from agent.runtime.session.store import (
    ACTIVE_SESSION_EXTRA_COLUMNS,
    ACTIVE_SESSION_STATE_DDL,
    ActiveSessionStore,
    InMemoryActiveSessionStore,
    NullActiveSessionStore,
    PostgresActiveSessionStore,
)

__all__ = [
    "ACTIVE_SESSION_EXTRA_COLUMNS",
    "ACTIVE_SESSION_STATE_DDL",
    "ActiveSessionManager",
    "ActiveSessionStore",
    "InMemoryActiveSessionStore",
    "NullActiveSessionStore",
    "PersistedActiveSessionRow",
    "PersistedActiveSessionState",
    "PostgresActiveSessionStore",
    "parse_iso_timestamp",
]
