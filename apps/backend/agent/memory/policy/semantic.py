"""Shared semantic-memory policy constants."""

from __future__ import annotations

SEMANTIC_SESSION_ONLY_CATEGORIES: frozenset[str] = frozenset(
    {
        "loss",
        "trigger",
    }
)
