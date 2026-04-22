"""Shared helpers for memory-subsystem records.

Two small utilities that were previously duplicated across
``agent/nodes/crisis_log.py`` (``_hash_session_id``) and
``agent/persistence.py`` (``_iso_now``). They're moved here so new
memory subsystems (e.g. the session-feedback collector) can reuse them
without cross-importing node-private helpers.

Semantics are preserved exactly — the old private names are kept as
module-local aliases at the original sites so existing tests and
internal callers keep working unchanged.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

_NO_SESSION_PLACEHOLDER = "__no_session_id__"


def hash_session_id(session_id: str | None) -> str:
    """Return a stable opaque hash for a session id.

    Args:
        session_id (str | None): Raw session id, or ``None``.

    Returns:
        str: SHA-256 hex digest of the session id or the null-session placeholder.
    """

    source = session_id or _NO_SESSION_PLACEHOLDER
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def iso_now() -> str:
    """Return the current UTC timestamp in the project's ISO-8601 format.

    Returns:
        str: Current UTC timestamp with a trailing ``Z`` suffix.
    """

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
