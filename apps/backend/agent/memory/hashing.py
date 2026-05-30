"""Shared timestamp and identifier helpers for memory-adjacent records.

Memory, audit, and persistence code use these helpers to avoid
duplicating timestamp formatting and opaque session-id hashing.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

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


def extract_iso_date(timestamp: str) -> str:
    """Return the ``YYYY-MM-DD`` date bucket from an ISO-8601 timestamp.

    Splitting on ``"T"`` yields the date portion of a full timestamp and is a
    no-op for an already-bare date. ``date.fromisoformat`` is called for its
    validation side effect: a malformed prefix raises ``ValueError`` rather than
    letting a bad date bucket reach storage.

    Args:
        timestamp (str): ISO-8601 timestamp (or a bare ``YYYY-MM-DD`` date).

    Returns:
        str: The ``YYYY-MM-DD`` date prefix.

    Raises:
        ValueError: If the date prefix is not a valid ISO-8601 date.
    """

    date_prefix = timestamp.split("T", 1)[0]
    date.fromisoformat(date_prefix)
    return date_prefix
