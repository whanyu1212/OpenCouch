"""Shared serialization boundary for crisis-log persistence drivers.

Both the SQLite and PostgreSQL crisis-log backends round-trip a
``CrisisLogRecord`` through the same Pydantic boundary: the model is dumped to a
plain JSON-able ``dict`` on write and re-validated from a ``dict`` on read. That
boundary is what these helpers own.

Storage *encoding* deliberately stays in each driver, because it genuinely
differs: SQLite serializes the dict with ``json.dumps(..., default=str)`` into a
``TEXT`` column and must ``json.loads`` it back; PostgreSQL wraps the dict in
``Jsonb`` for a ``JSONB`` column and reads it back as a ``dict`` directly. Keeping
the encode/decode in the drivers and the model boundary here lets the two
backends share the Pydantic contract without entangling their column types.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.audit.models import CrisisLogRecord


def serialize_crisis_record(record: CrisisLogRecord) -> dict[str, Any]:
    """Dump a crisis record to a plain JSON-able dict.

    Args:
        record (CrisisLogRecord): Crisis event record to serialize.

    Returns:
        dict[str, Any]: JSON-mode dump with no Pydantic types, ready for a
            driver to encode for its column (TEXT via ``json.dumps`` or JSONB
            via ``Jsonb``).
    """

    return record.model_dump(mode="json")


def deserialize_crisis_record(payload: Mapping[str, Any]) -> CrisisLogRecord:
    """Validate a stored dict back into a crisis record.

    Args:
        payload (Mapping[str, Any]): Decoded row value (``json.loads`` output for
            SQLite, or the JSONB dict for PostgreSQL).

    Returns:
        CrisisLogRecord: Validated crisis event record.
    """

    return CrisisLogRecord.model_validate(payload)


__all__ = [
    "serialize_crisis_record",
    "deserialize_crisis_record",
]
