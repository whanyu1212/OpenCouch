"""Compatibility imports for the legacy SQLite memory store."""

from agent.memory.legacy.sqlite_store import (
    MEMORY_RECORDS_DDL,
    MEMORY_RECORDS_INDEX_LAST_REF_DDL,
    MEMORY_RECORDS_INDEX_OWNER_KIND_DDL,
    MEMORY_SCHEMA_DDL,
    SqliteMemoryStore,
)

__all__ = [
    "SqliteMemoryStore",
    "MEMORY_SCHEMA_DDL",
    "MEMORY_RECORDS_DDL",
    "MEMORY_RECORDS_INDEX_OWNER_KIND_DDL",
    "MEMORY_RECORDS_INDEX_LAST_REF_DDL",
]
