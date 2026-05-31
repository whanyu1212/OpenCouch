"""Compatibility facade for memory read-path retrieval service."""

from agent.memory.retrieval.service import (
    EPISODIC_MAX_AGE_DAYS,
    EPISODIC_SEARCH_LIMIT,
    SEMANTIC_SEARCH_LIMIT,
    SEMANTIC_WORKING_MEMORY_LIMIT,
    LoadMemoryResult,
    RetrievalPath,
    load_memory_for_turn,
)

__all__ = [
    "EPISODIC_MAX_AGE_DAYS",
    "EPISODIC_SEARCH_LIMIT",
    "SEMANTIC_SEARCH_LIMIT",
    "SEMANTIC_WORKING_MEMORY_LIMIT",
    "LoadMemoryResult",
    "RetrievalPath",
    "load_memory_for_turn",
]
