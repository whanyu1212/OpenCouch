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
from typing import Literal


class MemoryMode(StrEnum):
    """Persistence tier for the agent's memory layer.

    See the module docstring for the semantics of each mode.
    """

    INCOGNITO = "incognito"
    LOCAL = "local"
    SYNCED = "synced"


EffectiveMemoryMode = Literal["incognito", "persistent"]


def resolve_effective_memory_mode(
    runtime_mode: MemoryMode | str | None,
    requested_mode: str | None,
) -> EffectiveMemoryMode:
    """Return the binary memory mode that should govern a single request.

    The runtime mode is authoritative: a request can opt down to incognito
    but never escalate to persistent. If either side asks for incognito,
    the result is incognito.

    Args:
        runtime_mode: The process-level memory mode the runtime was built
            with (``MemoryMode`` or its string form). ``None`` is treated
            as persistent.
        requested_mode: The per-request mode the client sent (``"incognito"``
            or ``"persistent"``). ``None`` defers to ``runtime_mode``.

    Returns:
        ``"incognito"`` when either input is incognito, otherwise
        ``"persistent"``. Non-incognito runtime modes (``LOCAL``,
        ``SYNCED``) collapse to ``"persistent"`` for callers that only
        care about the binary read/write surface.
    """

    if _is_incognito(runtime_mode) or _is_incognito(requested_mode):
        return "incognito"
    return "persistent"


def _is_incognito(value: MemoryMode | str | None) -> bool:
    """Return whether ``value`` represents the incognito mode."""

    if value is None:
        return False
    return str(value).strip().lower() == MemoryMode.INCOGNITO.value
