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
    """Return a SHA-256 hash of the session id, padded if None.

    Used for the ``session_id_opaque`` field on audit records. A stable
    hash means two records from the same session share an opaque
    identifier without exposing the original session id. When
    ``session_id`` is ``None`` (or empty, which is treated the same),
    we hash a placeholder so the field is always populated.
    """

    source = session_id or _NO_SESSION_PLACEHOLDER
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def iso_now() -> str:
    """Return the current UTC time in ISO-8601 format with 'Z' suffix.

    Callers and stored records across the codebase rely on the 'Z'
    suffix rather than the raw ``+00:00`` offset that Python's default
    ``isoformat()`` emits. Keep this helper as the single source of
    truth for that format choice.
    """

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
