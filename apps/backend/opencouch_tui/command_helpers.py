"""Shared slash-command parsing and plain-text formatting helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from agent.runtime import ThreadSummary


def parse_optional_count_arg(
    tokens: list[str],
    *,
    default: int,
) -> int | None:
    """Parse an optional numeric count argument from slash-command tokens.

    Args:
        tokens: Full slash-command token list, including the command name.
        default: Default count when no explicit argument was provided.

    Returns:
        Parsed positive count, or ``None`` when the tokens are invalid.
    """

    if len(tokens) > 2:
        return None
    if len(tokens) == 1:
        return default
    try:
        return max(1, int(tokens[1]))
    except ValueError:
        return None


def format_history_plain(history: list[Any], *, limit: int) -> str:
    """Format recent transcript messages as plain text.

    Args:
        history: Runtime message list.
        limit: Maximum number of trailing messages to include.

    Returns:
        Plain-text history lines in ``role: content`` format.
    """

    recent = history[-max(1, limit) :]
    return "\n".join(f"{message.role.value}: {message.content}" for message in recent)


def format_thread_summaries_plain(
    threads: list[ThreadSummary],
    *,
    active_thread_id: str | None,
) -> str:
    """Format thread summaries for plain-text command output.

    Args:
        threads: Persisted thread summaries.
        active_thread_id: Active thread id, if any.

    Returns:
        One summary per line with the active thread marked by ``*``.
    """

    lines: list[str] = []
    for summary in threads:
        marker = "*" if summary.thread_id == active_thread_id else "-"
        context = "context" if summary.has_context else "no-context"
        lines.append(
            f"{marker} {summary.thread_id}  turns={summary.turn_count}  "
            f"messages={summary.message_count}  {context}"
        )
    return "\n".join(lines)


def format_transcript_entries_plain(
    entries: list[dict[str, Any]],
    *,
    uppercase_roles: bool = False,
) -> str:
    """Format transcript entry mappings as plain text.

    Args:
        entries: Transcript entry mappings with ``role`` and ``content`` keys.
        uppercase_roles: Whether to uppercase rendered role labels.

    Returns:
        Plain-text transcript lines in ``role: content`` format.
    """

    lines: list[str] = []
    for entry in entries:
        role = str(entry.get("role", "unknown"))
        if uppercase_roles:
            role = role.upper()
        content = str(entry.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_memory_snapshot_plain(
    snapshot: dict[str, Any],
    *,
    kind: str,
) -> str:
    """Format memory snapshot data for plain-text slash-command output.

    Args:
        snapshot: Memory snapshot payload from the runtime.
        kind: One of ``all``, ``facts``, ``sessions``, or ``rules``.

    Returns:
        Plain-text memory summary.
    """

    lines = [f"owner: {snapshot['owner_id']}"]
    semantic = snapshot.get("semantic", [])
    episodic = snapshot.get("episodic", [])
    procedural = snapshot.get("procedural")

    if kind in {"all", "facts"}:
        lines.append("facts:")
        if semantic:
            for index, record in enumerate(semantic, start=1):
                target = record.get("object", {})
                if isinstance(target, dict):
                    target_text = target.get("identifier", "?")
                else:
                    target_text = str(target)
                lines.append(
                    f"  {index}. {record.get('predicate', '?')}: {target_text}"
                )
        else:
            lines.append("  none")

    if kind in {"all", "sessions"}:
        lines.append("sessions:")
        if episodic:
            for index, record in enumerate(episodic, start=1):
                lines.append(
                    f"  {index}. {record.get('session_id', '?')}: {record.get('summary', '?')}"
                )
        else:
            lines.append("  none")

    if kind in {"all", "rules"}:
        lines.append("rules:")
        rules = procedural.get("rules", []) if isinstance(procedural, dict) else []
        if rules:
            for index, rule in enumerate(rules, start=1):
                if isinstance(rule, dict):
                    lines.append(f"  {index}. {rule.get('rule', '?')}")
                else:
                    lines.append(f"  {index}. {rule}")
        else:
            lines.append("  none")

    return "\n".join(lines)


def format_memory_status_plain(snapshot: dict[str, Any]) -> str:
    """Format memory namespace counts for plain-text status output.

    Args:
        snapshot: Memory snapshot payload from the runtime.

    Returns:
        Plain-text memory status summary.
    """

    semantic = snapshot.get("semantic", [])
    episodic = snapshot.get("episodic", [])
    procedural = snapshot.get("procedural")
    rule_count = 0
    if isinstance(procedural, dict):
        rules = procedural.get("rules", [])
        rule_count = len(rules) if isinstance(rules, list) else 0
    return "\n".join(
        [
            f"owner: {snapshot['owner_id']}",
            f"facts: {len(semantic)}",
            f"sessions: {len(episodic)}",
            f"rules: {rule_count}",
        ]
    )


def snippet_around_match(text: str, query: str, *, max_len: int = 96) -> str:
    """Return a compact snippet centered around the first query match."""

    normalized = text.strip()
    if len(normalized) <= max_len:
        return normalized
    lower_text = normalized.lower()
    lower_query = query.lower()
    match_index = lower_text.find(lower_query)
    if match_index < 0:
        return normalized[: max_len - 1].rstrip() + "…"

    half_window = max_len // 2
    start = max(0, match_index - half_window)
    end = min(len(normalized), start + max_len)
    start = max(0, end - max_len)
    snippet = normalized[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(normalized):
        snippet = snippet + "…"
    return snippet


def first_matching_text(query: str, *candidates: str) -> str | None:
    """Return the first candidate text containing the case-insensitive query."""

    lower_query = query.lower()
    for candidate in candidates:
        normalized = candidate.strip()
        if normalized and lower_query in normalized.lower():
            return normalized
    return None


def format_entity_identifier(entity: object) -> str:
    """Return the serialized entity identifier when available."""

    if isinstance(entity, dict):
        identifier = entity.get("identifier")
        if identifier:
            return str(identifier)
    return "?"


def search_history_messages(
    messages: Sequence[Any],
    *,
    query: str,
) -> list[tuple[str, str]]:
    """Search visible transcript messages and return labeled snippets."""

    results: list[tuple[str, str]] = []
    for message in messages:
        content = str(getattr(message, "content", "")).strip()
        role = getattr(getattr(message, "role", None), "value", None) or str(
            getattr(message, "role", "unknown")
        )
        if not content or query.lower() not in content.lower():
            continue
        snippet = snippet_around_match(content, query)
        results.append(("history", f"{role}: {snippet}"))
    return results


def search_memory_snapshot(
    snapshot: dict[str, Any],
    *,
    query: str,
) -> list[tuple[str, str]]:
    """Search snapshot-style semantic/episodic/procedural memory content."""

    results: list[tuple[str, str]] = []

    semantic = snapshot.get("semantic", [])
    for index, record in enumerate(semantic, start=1):
        if not isinstance(record, dict):
            continue
        object_identifier = format_entity_identifier(record.get("object"))
        predicate = str(record.get("predicate", "") or "").strip()
        evidence_quote = str(record.get("evidence_quote", "") or "").strip()
        semantic_summary = " ".join(
            part for part in (predicate, object_identifier) if part and part != "?"
        ).strip()
        matched_text = first_matching_text(query, evidence_quote, semantic_summary)
        if matched_text is None:
            continue
        results.append(
            ("memory/fact", f"#{index}: {snippet_around_match(matched_text, query)}")
        )

    episodic = snapshot.get("episodic", [])
    for index, record in enumerate(episodic, start=1):
        if not isinstance(record, dict):
            continue
        summary = str(record.get("summary", "") or "").strip()
        themes_value = record.get("primary_themes")
        themes = (
            ", ".join(str(theme) for theme in themes_value)
            if isinstance(themes_value, list)
            else ""
        )
        matched_text = first_matching_text(query, summary, themes)
        if matched_text is None:
            continue
        results.append(
            ("memory/session", f"#{index}: {snippet_around_match(matched_text, query)}")
        )

    procedural = snapshot.get("procedural")
    rules = procedural.get("rules", []) if isinstance(procedural, dict) else []
    for index, rule in enumerate(rules, start=1):
        if isinstance(rule, dict):
            rule_text = str(rule.get("rule", "") or "").strip()
            evidence_values = rule.get("evidence", [])
            evidence_text = (
                " ".join(str(evidence).strip() for evidence in evidence_values)
                if isinstance(evidence_values, list)
                else ""
            )
        else:
            rule_text = str(rule).strip()
            evidence_text = ""
        matched_text = first_matching_text(query, rule_text, evidence_text)
        if matched_text is None:
            continue
        results.append(
            ("memory/rule", f"#{index}: {snippet_around_match(matched_text, query)}")
        )

    return results
