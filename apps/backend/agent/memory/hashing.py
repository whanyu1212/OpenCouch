"""Shared timestamp and identifier helpers for memory-adjacent records.

Memory, audit, and persistence code use these helpers to avoid
duplicating timestamp formatting and opaque session-id hashing.
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
