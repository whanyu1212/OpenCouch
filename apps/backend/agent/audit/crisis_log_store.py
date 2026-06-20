"""Shared crisis-log table config for the KvStore-backed crisis backends.

The SQLite and PostgreSQL crisis-log backends share one
:class:`~agent.storage.kv_store.KvStore` body; this module holds the
crisis-log-specific table layout and the record (de)serialization boundary, so
each driver module only supplies its own DDL and dialect.
"""

from __future__ import annotations

from agent.audit.crisis_log_serialization import (
    deserialize_crisis_record,
    serialize_crisis_record,
)
from agent.audit.models import CrisisLogRecord
from agent.memory.hashing import extract_iso_date
from agent.storage.kv_store import KvTableConfig

CRISIS_LOG_INSERT_COLUMNS: tuple[str, ...] = (
    "id",
    "session_id_opaque",
    "user_id_or_null",
    "detected_at",
    "detected_date",
    "level",
    "value",
)


def _crisis_to_row(record: CrisisLogRecord) -> list[object]:
    """Bind a crisis record to its non-value INSERT parameters, in column order.

    Args:
        record (CrisisLogRecord): Record being appended.

    Returns:
        list[object]: Bound parameters for every INSERT column except ``value``.
    """

    return [
        record.id,
        record.session_id_opaque,
        record.user_id_or_null,
        record.detected_at,
        extract_iso_date(record.detected_at),
        record.level,
    ]


def build_crisis_log_table_config(
    ddls: tuple[str, ...],
) -> KvTableConfig[CrisisLogRecord]:
    """Build the crisis-log :class:`KvTableConfig` for a given dialect's DDL.

    Args:
        ddls (tuple[str, ...]): The dialect-specific schema DDL tuple.

    Returns:
        KvTableConfig[CrisisLogRecord]: Table layout + (de)serialization.
    """

    return KvTableConfig(
        table="crisis_log",
        key_column="detected_date",
        date_column="detected_date",
        insert_columns=CRISIS_LOG_INSERT_COLUMNS,
        ddls=ddls,
        to_row=_crisis_to_row,
        date_of=lambda record: extract_iso_date(record.detected_at),
        serialize=serialize_crisis_record,
        deserialize=deserialize_crisis_record,
    )


__all__ = [
    "CRISIS_LOG_INSERT_COLUMNS",
    "build_crisis_log_table_config",
]
