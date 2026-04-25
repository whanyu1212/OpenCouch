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

    type: Literal["semantic"]
    evidence_quote: str
    category: str
    subject: str
    predicate: str
    object: str


class EpisodicWorkingMemoryEntry(TypedDict, total=False):
    """Episodic session arc retrieved for the current turn."""

    type: Literal["episodic"]
    summary: str
    primary_themes: list[str]
    is_catch_up: bool
    approach_used: str
    approach_context: dict[str, object]


WorkingMemoryEntry: TypeAlias = SemanticWorkingMemoryEntry | EpisodicWorkingMemoryEntry


def make_semantic_working_memory_entry(
    *,
    evidence_quote: str,
    category: str = "",
    subject: str = "",
    predicate: str = "",
    object: str = "",
) -> SemanticWorkingMemoryEntry:
    """Build a semantic working-memory entry.

    Args:
        evidence_quote: Source quote that justified the semantic memory.
        category: Optional semantic category label.
        subject: Optional subject identifier.
        predicate: Optional relationship predicate.
        object: Optional object identifier.

    Returns:
        Structured entry for graph state.
    """

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
    approach_context: dict[str, object] | None = None,
) -> EpisodicWorkingMemoryEntry:
    """Build an episodic working-memory entry.

    Args:
        summary: Session summary text to expose as working memory.
        primary_themes: Optional theme labels for the session.
        is_catch_up: Whether this entry is first-turn catch-up context.
        approach_used: Optional modality label from the stored session.
        approach_context: Optional modality-specific session details.

    Returns:
        Structured entry for graph state.
    """

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


def _format_approach_context(ctx: dict[str, object]) -> list[str]:
    """Render approach_context fields as concise human-readable fragments.

    Args:
        ctx: Stored approach-context payload from an episodic arc.

    Returns:
        Short display fragments for recognized context fields.
    """

    parts: list[str] = []
    _LABELS = {
        "thought_examined": "Thought",
        "action_step": "Action step",
        "tool_used": "Tool",
        "readiness_stage": "Readiness",
        "change_talk_themes": "Change talk",
        "sustain_talk_themes": "Sustain talk",
        "values_identified": "Values",
        "fusion_patterns": "Fusion",
        "committed_action": "Committed action",
        "person_lost": "Person lost",
        "relationship": "Relationship",
        "time_since_loss": "Time since loss",
        "problem_area": "Problem area",
        "key_relationship": "Key relationship",
        "communication_step_planned": "Communication step",
        "skills_used": "Skills used",
        "primary_domain": "Domain",
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

    Args:
        entry: Structured or legacy working-memory entry.

    Returns:
        Human-readable rendering, or an empty string for unsupported entries.
    """

    if isinstance(entry, str):
        return entry

    entry_type = entry.get("type")
    if entry_type == "semantic":
        quote_raw = entry.get("evidence_quote")
        quote = quote_raw.strip() if isinstance(quote_raw, str) else ""
        if not quote:
            return ""
        category_raw = entry.get("category")
        subject_raw = entry.get("subject")
        predicate_raw = entry.get("predicate")
        object_raw = entry.get("object")
        category = category_raw if isinstance(category_raw, str) else ""
        subject = subject_raw if isinstance(subject_raw, str) else ""
        predicate = predicate_raw if isinstance(predicate_raw, str) else ""
        obj = object_raw if isinstance(object_raw, str) else ""
        if subject and predicate and obj:
            pred_label = predicate.replace("_", " ")
            return f"[{category}] {subject} {pred_label} {obj} — '{quote}'"
        return f"Previously noted: {quote}"

    if entry_type == "episodic":
        summary_raw = entry.get("summary")
        summary = summary_raw.strip() if isinstance(summary_raw, str) else ""
        if not summary:
            return ""
        themes_raw = entry.get("primary_themes")
        modality_raw = entry.get("approach_used")
        tag_parts = (
            [theme for theme in themes_raw if isinstance(theme, str)]
            if isinstance(themes_raw, list)
            else []
        )
        modality = modality_raw if isinstance(modality_raw, str) else ""
        if modality and modality != "none":
            tag_parts.append(modality.upper().replace("_", " "))
        tags_str = ", ".join(tag_parts) if tag_parts else "untagged"
        base = f"Last session ({tags_str}): {summary}"

        ctx = entry.get("approach_context")
        ctx_parts = _format_approach_context(ctx) if isinstance(ctx, dict) else []
        if ctx_parts:
            base += " [" + "; ".join(ctx_parts) + "]"
        return base

    return ""


def format_working_memory_entries(
    entries: list[WorkingMemoryEntry] | list[str] | None,
    *,
    limit: int | None = None,
) -> list[str]:
    """Render working-memory entries for prompt or CLI display.

    Args:
        entries: Entries to render.
        limit: Optional maximum number of entries to render.

    Returns:
        Non-empty rendered entry strings.
    """

    if not entries:
        return []

    selected = entries[:limit] if limit is not None else entries
    rendered = [format_working_memory_entry(entry) for entry in selected]
    return [entry for entry in rendered if entry]
