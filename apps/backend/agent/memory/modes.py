"""Memory persistence modes for the OpenCouch agent.

- ``INCOGNITO`` — ephemeral, in-memory only. Memory and audit stores avoid
  disk writes.
- ``LOCAL`` — persists through Postgres and stays within the configured
  deployment.
- ``SYNCED`` — reserved for a future remote persistence tier. Runtime
  code currently treats it like durable mode while backend sync remains
  unimplemented.

The enum inherits from ``str`` so instances compare equal to their
string value (``MemoryMode.LOCAL == "local"``), which is useful for
CLI argument parsing and JSON serialization.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable


class MemoryMode(StrEnum):
    """Persistence tier for the agent's memory layer.

    See the module docstring for the semantics of each mode.
    """

    INCOGNITO = "incognito"
    LOCAL = "local"
    SYNCED = "synced"


#: Attribute a dependency sets to declare it is safe for incognito runtimes.
EPHEMERAL_CAPABILITY_ATTRIBUTE = "supports_incognito"


@runtime_checkable
class EphemeralCapable(Protocol):
    """Marker for dependencies an incognito runtime may use.

    Incognito runtimes promise never to touch disk, a database, or a remote
    service. Runtime-owned backend selection honors that, but callers can also
    inject their own stores and providers, and an injected dependency would
    otherwise override the mode's choice silently.

    Rather than rejecting a known list of durable implementations, a
    dependency must *opt in* by setting ``supports_incognito = True``. This
    fails closed: an implementation the runtime has never seen — including a
    caller's own durable wrapper — is rejected until it declares itself
    ephemeral. A blocklist would fail open for exactly those cases.

    Declaring this attribute is an assertion by the implementation that it
    performs no durable writes and makes no network calls. In-repo ephemeral
    backends declare it; durable ones deliberately do not.
    """

    supports_incognito: bool


def is_ephemeral_capable(dependency: object) -> bool:
    """Return whether a dependency declares itself incognito-safe.

    The declaration must live on the dependency's **own class**, and the
    capability is never inherited. Two escapes this closes:

    - Reading the attribute off the *instance* would let a stray
      ``store.supports_incognito = True`` anywhere in caller code disable a
      privacy control with no class-level declaration to review.
    - Accepting an *inherited* marker would opt in any subclass of an
      ephemeral store, including a durable wrapper that overrides writes to
      mirror them to disk or a remote service. Subclassing an in-memory
      store is precisely how such a wrapper would be written.

    Each concrete implementation therefore restates the marker, which is the
    point: the assertion "this performs no durable writes and no network
    calls" is about the concrete behavior, so it cannot be delegated to a
    base class that knows nothing about the override.

    The value must be exactly ``True``. Truthy stand-ins are rejected so that
    a partially-migrated or accidentally-typed value cannot grant capability,
    and so ``False`` reads as a deliberate opt-out.

    Args:
        dependency (object): Injected store, backend, or provider.

    Returns:
        bool: ``True`` when the dependency opts in to incognito use.
    """

    declared = vars(type(dependency)).get(EPHEMERAL_CAPABILITY_ATTRIBUTE, False)
    return declared is True


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
