"""Memory persistence modes for the OpenCouch agent.

The ``MemoryMode`` enum replaces the earlier ``is_guest_mode`` boolean
and represents the three persistence tiers locked in the schema v1
decisions log (see ``agent/memory/schema.yaml`` §1):

- ``INCOGNITO`` — ephemeral, in-memory only. Checkpointer runs against
  ``:memory:``. Nothing survives the runtime instance dying. The crisis
  safety log is the ONLY exception — it always persists regardless of
  mode, with no user identifier attached.
- ``LOCAL`` — persists to local SQLite (and optionally local Neo4j for
  graph memory in phase 3+). Never leaves the device. Default for the
  desktop CLI.
- ``SYNCED`` — persists to a remote Postgres + remote Neo4j backend.
  Requires user accounts and auth. Reserved in the schema so phases 1-3
  can be designed against it; backend not yet implemented.

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
