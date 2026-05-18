"""Memory persistence modes for the OpenCouch agent.

- ``INCOGNITO`` — ephemeral, in-memory only. Session state is stored in
  ``:memory:`` and memory/audit stores avoid disk writes.
- ``LOCAL`` — persists through the configured local backend and stays on
  the device. Postgres is the recommended durable backend; SQLite remains
  available as a legacy fallback.
- ``SYNCED`` — reserved for a future remote persistence tier. Runtime
  code currently treats it like durable mode while backend sync remains
  unimplemented.

The enum inherits from ``str`` so instances compare equal to their
string value (``MemoryMode.LOCAL == "local"``), which is useful for
CLI argument parsing and JSON serialization.
"""

from __future__ import annotations

from enum import StrEnum


class MemoryMode(StrEnum):
    """Persistence tier for the agent's memory layer.

    See the module docstring for the semantics of each mode.
    """

    INCOGNITO = "incognito"
    LOCAL = "local"
    SYNCED = "synced"
