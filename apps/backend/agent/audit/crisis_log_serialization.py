"""Pydantic serialization boundary for durable crisis-log records."""

from __future__ import annotations

from typing import Any, Mapping

from agent.audit.models import CrisisLogRecord


def serialize_crisis_record(record: CrisisLogRecord) -> dict[str, Any]:
    """Dump a crisis record to a plain JSON-able dict.

    Args:
        record (CrisisLogRecord): Crisis event record to serialize.

    Returns:
        dict[str, Any]: JSON-mode dump ready for the Postgres JSONB column.
    """

    return record.model_dump(mode="json")


def deserialize_crisis_record(payload: Mapping[str, Any]) -> CrisisLogRecord:
    """Validate a stored dict back into a crisis record.

    Args:
        payload (Mapping[str, Any]): Decoded Postgres JSONB row value.

    Returns:
        CrisisLogRecord: Validated crisis event record.
    """

    return CrisisLogRecord.model_validate(payload)


__all__ = [
    "serialize_crisis_record",
    "deserialize_crisis_record",
]
