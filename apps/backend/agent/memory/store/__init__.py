"""Public re-export surface for the memory store package.

Concrete definitions live in :mod:`agent.memory.store.base` (the shared
contract) and :mod:`agent.memory.store.memory` (the in-memory backend).
This module re-exports them so callers keep a stable import surface.
"""

from __future__ import annotations

from agent.memory.store.base import (
    SEARCH_MATCH_THRESHOLD,
    MemoryRecordFilter,
    MemoryStore,
    Namespace,
    StoreRecord,
    memory_record_matches_filter,
)
from agent.memory.store.memory import OpenCouchMemoryStore

__all__ = [
    "SEARCH_MATCH_THRESHOLD",
    "MemoryRecordFilter",
    "MemoryStore",
    "Namespace",
    "OpenCouchMemoryStore",
    "StoreRecord",
    "memory_record_matches_filter",
]
