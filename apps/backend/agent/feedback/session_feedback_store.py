"""Shared session-feedback table config for the KvStore-backed backends.

The SQLite and PostgreSQL session-feedback backends share one
:class:`~agent.storage.kv_store.KvStore` body; this module holds the
feedback-specific table layout and the record (de)serialization boundary, so
each driver module only supplies its own DDL and dialect.

Three columns (``turn_count_at_end``, ``source``, ``schema_version``) are
written for indexability/operability but are never read back: the authoritative
record is reconstructed from the ``value`` JSON column.
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.feedback.models import SessionFeedbackRecord
from agent.memory.hashing import extract_iso_date
from agent.storage.kv_store import KvTableConfig

SESSION_FEEDBACK_INSERT_COLUMNS: tuple[str, ...] = (
    "id",
    "session_id_opaque",
    "user_id_or_null",
    "recorded_at",
    "recorded_date",
    "label",
    "turn_count_at_end",
    "source",
    "schema_version",
    "value",
)


def _feedback_to_row(record: SessionFeedbackRecord) -> list[object]:
    """Bind a feedback record to its non-value INSERT parameters, in column order.

    Args:
        record (SessionFeedbackRecord): Record being appended.

    Returns:
        list[object]: Bound parameters for every INSERT column except ``value``.
    """

    return [
        record.id,
        record.session_id_opaque,
        record.user_id_or_null,
        record.recorded_at,
        extract_iso_date(record.recorded_at),
        record.label,
        record.turn_count_at_end,
        record.source,
        record.schema_version,
    ]


def _serialize(record: SessionFeedbackRecord) -> Mapping[str, Any]:
    """Dump a feedback record to a JSON-able dict for the value column."""

    return record.model_dump(mode="json")


def _deserialize(payload: Mapping[str, Any]) -> SessionFeedbackRecord:
    """Validate a decoded dict back into a feedback record."""

    return SessionFeedbackRecord.model_validate(payload)


def build_session_feedback_table_config(
    ddls: tuple[str, ...],
) -> KvTableConfig[SessionFeedbackRecord]:
    """Build the session-feedback :class:`KvTableConfig` for a dialect's DDL.

    Args:
        ddls (tuple[str, ...]): The dialect-specific schema DDL tuple.

    Returns:
        KvTableConfig[SessionFeedbackRecord]: Table layout + (de)serialization.
    """

    return KvTableConfig(
        table="session_feedback",
        key_column="session_id_opaque",
        date_column="recorded_date",
        insert_columns=SESSION_FEEDBACK_INSERT_COLUMNS,
        ddls=ddls,
        to_row=_feedback_to_row,
        date_of=lambda record: extract_iso_date(record.recorded_at),
        serialize=_serialize,
        deserialize=_deserialize,
    )


__all__ = [
    "SESSION_FEEDBACK_INSERT_COLUMNS",
    "build_session_feedback_table_config",
]
