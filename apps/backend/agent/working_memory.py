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


class EpisodicWorkingMemoryEntry(TypedDict, total=False):
    """Episodic session arc retrieved for the current turn."""

    type: Literal["episodic"]  # required
    summary: str  # required
    primary_themes: list[str]  # required
    is_catch_up: bool  # required
    approach_used: str
    approach_context: dict


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
    approach_used: str | None = None,
    approach_context: dict | None = None,
) -> EpisodicWorkingMemoryEntry:
    """Build an episodic working-memory entry."""

    entry: EpisodicWorkingMemoryEntry = {
        "type": "episodic",
        "summary": summary,
        "primary_themes": list(primary_themes or []),
        "is_catch_up": is_catch_up,
    }
    if approach_used:
        entry["approach_used"] = approach_used
    if approach_context:
        entry["approach_context"] = approach_context
    return entry


def _format_approach_context(ctx: dict) -> list[str]:
    """Render approach_context fields as concise human-readable fragments.

    Returns a list of short strings like "Thought: I'm going to get fired"
    or "Action step: speak up in one meeting". Empty/null fields are skipped.
    """

    parts: list[str] = []
    # Map of context keys → human-readable labels. Order determines
    # rendering order. Keys not present in ctx are silently skipped.
    _LABELS = {
        # CBT
        "thought_examined": "Thought",
        "action_step": "Action step",
        "tool_used": "Tool",
        # MI
        "readiness_stage": "Readiness",
        "change_talk_themes": "Change talk",
        "sustain_talk_themes": "Sustain talk",
        # ACT
        "values_identified": "Values",
        "fusion_patterns": "Fusion",
        "committed_action": "Committed action",
        # Grief
        "person_lost": "Person lost",
        "relationship": "Relationship",
        "time_since_loss": "Time since loss",
        # IPT
        "problem_area": "Problem area",
        "key_relationship": "Key relationship",
        "communication_step_planned": "Communication step",
        # DBT
        "skills_used": "Skills used",
        "primary_domain": "Domain",
        # PFA
        "crisis_type": "Crisis type",
        "support_connected": "Support connected",
    }
    for key, label in _LABELS.items():
        value = ctx.get(key)
        if not value:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v)
            if not value:
                continue
        parts.append(f"{label}: {value}")
    return parts


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
        modality = entry.get("approach_used")
        # Build the parenthetical: themes + modality label
        tag_parts = list(themes) if themes else []
        if modality and modality != "none":
            tag_parts.append(modality.upper().replace("_", " "))
        tags_str = ", ".join(tag_parts) if tag_parts else "untagged"
        base = f"Last session ({tags_str}): {summary}"

        # Append modality-specific context as a concise suffix
        ctx = entry.get("approach_context") or {}
        ctx_parts = _format_approach_context(ctx) if ctx else []
        if ctx_parts:
            base += " [" + "; ".join(ctx_parts) + "]"
        return base

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
