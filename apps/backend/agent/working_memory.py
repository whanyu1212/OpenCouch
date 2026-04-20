"""Structured working-memory helpers.

This module keeps the durable graph state raw and pushes any
human-readable rendering to the surfaces that need it (prompt builders,
dispatcher prompts, CLI panels). The state carries structured entries;
formatting happens on demand.
"""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict


class SemanticWorkingMemoryEntry(TypedDict, total=False):
    """Semantic fact retrieved for the current turn."""

    type: Literal["semantic"]  # required
    evidence_quote: str  # required
    category: str
    subject: str
    predicate: str
    object: str


class EpisodicWorkingMemoryEntry(TypedDict):
    """Episodic session arc retrieved for the current turn."""

    type: Literal["episodic"]
    summary: str
    primary_themes: list[str]
    is_catch_up: bool


WorkingMemoryEntry: TypeAlias = SemanticWorkingMemoryEntry | EpisodicWorkingMemoryEntry


def make_semantic_working_memory_entry(
    *,
    evidence_quote: str,
    category: str = "",
    subject: str = "",
    predicate: str = "",
    object: str = "",
) -> SemanticWorkingMemoryEntry:
    """Build a semantic working-memory entry."""

    entry: SemanticWorkingMemoryEntry = {
        "type": "semantic",
        "evidence_quote": evidence_quote,
    }
    if category:
        entry["category"] = category
    if subject:
        entry["subject"] = subject
    if predicate:
        entry["predicate"] = predicate
    if object:
        entry["object"] = object
    return entry


def make_episodic_working_memory_entry(
    *,
    summary: str,
    primary_themes: list[str] | None = None,
    is_catch_up: bool,
) -> EpisodicWorkingMemoryEntry:
    """Build an episodic working-memory entry."""

    return {
        "type": "episodic",
        "summary": summary,
        "primary_themes": list(primary_themes or []),
        "is_catch_up": is_catch_up,
    }


def format_working_memory_entry(entry: WorkingMemoryEntry | str) -> str:
    """Render one working-memory entry for a human-facing surface.

    Accepts legacy ``str`` entries defensively so older manual fixtures
    and checkpoints still render cleanly during the migration.
    """

    if isinstance(entry, str):
        return entry

    entry_type = entry.get("type")
    if entry_type == "semantic":
        quote = entry.get("evidence_quote", "").strip()
        if not quote:
            return ""
        category = entry.get("category", "")
        subject = entry.get("subject", "")
        predicate = entry.get("predicate", "")
        obj = entry.get("object", "")
        if subject and predicate and obj:
            pred_label = predicate.replace("_", " ")
            return f"[{category}] {subject} {pred_label} {obj} — '{quote}'"
        return f"Previously noted: {quote}"

    if entry_type == "episodic":
        summary = entry.get("summary", "").strip()
        if not summary:
            return ""
        themes = entry.get("primary_themes") or []
        themes_str = ", ".join(themes) if themes else "untagged"
        return f"Last session ({themes_str}): {summary}"

    return ""


def format_working_memory_entries(
    entries: list[WorkingMemoryEntry] | list[str] | None,
    *,
    limit: int | None = None,
) -> list[str]:
    """Render a list of working-memory entries for prompt/CLI display."""

    if not entries:
        return []

    selected = entries[:limit] if limit is not None else entries
    rendered = [format_working_memory_entry(entry) for entry in selected]
    return [entry for entry in rendered if entry]
