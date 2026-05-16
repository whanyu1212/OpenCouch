#!/usr/bin/env python3
"""Inspect legacy SQLite memory produced by the LiveKit voice path.

Usage:
    python scripts/inspect_voice_memory.py --owner hy
    python scripts/inspect_voice_memory.py --thread voice-abc123
    python scripts/inspect_voice_memory.py --owner hy --namespace semantic
    python scripts/inspect_voice_memory.py --all-owners
    python scripts/inspect_voice_memory.py --owner hy --raw
    python scripts/inspect_voice_memory.py --sqlite-path apps/backend/.store/memory.sqlite3 --all-owners

This script is more opinionated than ``scripts/inspect_memory.py``:

- semantic records are treated as voice-derived when they were written
  by the explicit ``save_insight`` tool or when their
  ``source_session_id`` / ``thread_id`` looks like a voice session
- episodic records are treated as voice-derived when ``session_id``
  looks like a voice session
- procedural rules are owner-scoped and do not currently store
  per-session provenance, so owner-level inspection can show them, but
  thread-level inspection cannot attribute them safely

Run from the repo root or ``apps/backend/``. This is a legacy SQLite
inspector; Postgres-first environments should query Postgres directly.
The script auto-detects ``.store/memory.sqlite3`` unless ``--sqlite-path``
is provided.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def find_db(sqlite_path: str | None = None) -> Path:
    """Locate the legacy memory SQLite database.

    Returns:
        Path: Path to ``memory.sqlite3``.
    """

    if sqlite_path is not None:
        path = Path(sqlite_path).expanduser()
        if path.exists():
            return path
        print(f"Could not find memory SQLite database at {path}", file=sys.stderr)
        sys.exit(1)

    candidates = [
        Path("apps/backend/.store/memory.sqlite3"),
        Path(".store/memory.sqlite3"),
    ]
    for path in candidates:
        if path.exists():
            return path

    print("Could not find memory.sqlite3 in .store/", file=sys.stderr)
    sys.exit(1)


def fetch_records(
    db: Path,
    *,
    owner_id: str | None = None,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch memory rows, optionally filtered by owner and namespace.

    Args:
        db (Path): SQLite database path.
        owner_id (str | None): Optional owner filter.
        namespace (str | None): Optional namespace filter.

    Returns:
        list[dict[str, Any]]: Raw database rows.
    """

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM memory_records WHERE 1=1"
    params: list[str] = []
    if owner_id is not None:
        query += " AND owner_id = ?"
        params.append(owner_id)
    if namespace is not None:
        query += " AND namespace_kind = ?"
        params.append(namespace)

    query += " ORDER BY insertion_order"
    cursor.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def is_voice_thread_id(value: object) -> bool:
    """Return whether a value looks like a voice thread/session id.

    Args:
        value (object): Candidate value.

    Returns:
        bool: ``True`` when the value starts with ``voice-``.
    """

    return isinstance(value, str) and value.startswith("voice-")


def is_voice_semantic_value(value: dict[str, Any]) -> bool:
    """Return whether a semantic record was written from voice.

    Args:
        value (dict[str, Any]): Decoded semantic value.

    Returns:
        bool: ``True`` when the semantic record has voice provenance.
    """

    return (
        value.get("source") == "voice_tool"
        or is_voice_thread_id(value.get("source_session_id"))
        or is_voice_thread_id(value.get("thread_id"))
    )


def is_voice_episodic_value(value: dict[str, Any]) -> bool:
    """Return whether an episodic arc came from a voice session.

    Args:
        value (dict[str, Any]): Decoded episodic value.

    Returns:
        bool: ``True`` when the episodic record has voice provenance.
    """

    return is_voice_thread_id(value.get("session_id"))


def matches_voice_thread(
    namespace_kind: str,
    value: dict[str, Any],
    *,
    thread_id: str,
) -> bool:
    """Return whether a record is tied to a specific voice thread.

    Args:
        namespace_kind (str): Memory namespace kind.
        value (dict[str, Any]): Decoded record value.
        thread_id (str): Voice thread to match.

    Returns:
        bool: ``True`` when the record can be attributed to the thread.
    """

    if namespace_kind == "semantic":
        return value.get("source_session_id") == thread_id or value.get("thread_id") == thread_id
    if namespace_kind == "episodic":
        return value.get("session_id") == thread_id
    return False


def decode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Decode the JSON payload for each database row.

    Args:
        rows (list[dict[str, Any]]): Raw database rows.

    Returns:
        list[dict[str, Any]]: Rows with ``parsed_value`` attached.
    """

    decoded: list[dict[str, Any]] = []
    for row in rows:
        try:
            parsed = json.loads(row["value"])
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        decoded.append({**row, "parsed_value": parsed})
    return decoded


def list_voice_owners(db: Path) -> list[dict[str, Any]]:
    """Summarize owners with voice-derived memory.

    Args:
        db (Path): SQLite database path.

    Returns:
        list[dict[str, Any]]: Owner summary rows.
    """

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "voice_semantic": 0,
            "voice_episodic": 0,
            "procedural_total": 0,
        }
    )

    for row in decode_rows(fetch_records(db)):
        owner_id = str(row["owner_id"])
        namespace_kind = str(row["namespace_kind"])
        value = row["parsed_value"]

        if namespace_kind == "semantic" and is_voice_semantic_value(value):
            counts[owner_id]["voice_semantic"] += 1
        elif namespace_kind == "episodic" and is_voice_episodic_value(value):
            counts[owner_id]["voice_episodic"] += 1
        elif namespace_kind == "procedural":
            rule_count = len(value.get("rules", [])) if isinstance(value, dict) else 0
            counts[owner_id]["procedural_total"] = max(
                counts[owner_id]["procedural_total"],
                rule_count,
            )

    return [
        {"owner": owner_id, **owner_counts}
        for owner_id, owner_counts in sorted(counts.items())
        if owner_counts["voice_semantic"] > 0
        or owner_counts["voice_episodic"] > 0
        or owner_id.startswith("voice-")
    ]


def filter_voice_rows(
    rows: list[dict[str, Any]],
    *,
    thread_id: str | None = None,
) -> list[dict[str, Any]]:
    """Filter decoded rows down to voice-derived records.

    Args:
        rows (list[dict[str, Any]]): Decoded rows.
        thread_id (str | None): Optional exact voice thread filter.

    Returns:
        list[dict[str, Any]]: Voice-relevant rows.
    """

    filtered: list[dict[str, Any]] = []
    for row in rows:
        namespace_kind = str(row["namespace_kind"])
        value = row["parsed_value"]

        if thread_id is not None:
            if matches_voice_thread(namespace_kind, value, thread_id=thread_id):
                filtered.append(row)
            continue

        if namespace_kind == "semantic" and is_voice_semantic_value(value):
            filtered.append(row)
        elif namespace_kind == "episodic" and is_voice_episodic_value(value):
            filtered.append(row)
        elif namespace_kind == "procedural":
            filtered.append(row)

    return filtered


def format_semantic(value: dict[str, Any]) -> str:
    """Format a voice-derived semantic fact for display.

    Args:
        value (dict[str, Any]): Semantic fact payload.

    Returns:
        str: Human-readable semantic summary.
    """

    subject = value.get("subject", {})
    obj = value.get("object", {})
    subject_id = subject.get("identifier", "?") if isinstance(subject, dict) else str(subject)
    object_id = obj.get("identifier", "?") if isinstance(obj, dict) else str(obj)
    predicate = str(value.get("predicate", "")).replace("_", " ")
    source = value.get("source", "extractor")
    source_session_id = value.get("source_session_id") or value.get("thread_id") or "?"

    return (
        f"  [{value.get('category', '?')}] {subject_id} {predicate} {object_id}\n"
        f"  Quote: \"{value.get('evidence_quote', '')}\"\n"
        f"  Provenance: source={source} session={source_session_id} turn={value.get('source_turn_index', '?')}\n"
        f"  Confidence: {value.get('confidence', '?')}"
    )


def format_episodic(value: dict[str, Any]) -> str:
    """Format a voice-derived episodic session arc for display.

    Args:
        value (dict[str, Any]): Episodic arc payload.

    Returns:
        str: Human-readable episodic summary.
    """

    themes = ", ".join(value.get("primary_themes", [])) or "none"
    mood_arc = value.get("mood_arc", {})
    opened = mood_arc.get("opened", "?") if isinstance(mood_arc, dict) else "?"
    closed = mood_arc.get("closed", "?") if isinstance(mood_arc, dict) else "?"

    lines = [
        f"  Voice session: {value.get('session_id', '?')}",
        f"  Turns: {value.get('turn_count', '?')} | Mood: {opened} → {closed}",
        f"  Themes: {themes}",
        f"  Summary: {value.get('summary', '?')}",
    ]
    if value.get("approach_used"):
        lines.append(f"  Approach: {value.get('approach_used')}")
    return "\n".join(lines)


def format_procedural(value: dict[str, Any]) -> str:
    """Format owner-scoped procedural rules for display.

    Args:
        value (dict[str, Any]): Procedural profile payload.

    Returns:
        str: Human-readable procedural summary.
    """

    rules = value.get("rules", [])
    lines = [
        "  Procedural rules are owner-scoped. Per-thread voice provenance is not stored.",
        f"  Recall: {'on' if value.get('proactive_recall_enabled') else 'off'}",
        f"  Rules ({len(rules)}):",
    ]
    for rule in rules:
        rule_value = rule if isinstance(rule, dict) else {"rule": str(rule)}
        evidence = rule_value.get("evidence") or []
        evidence_preview = ""
        if evidence:
            evidence_preview = f" | evidence: {evidence[0][:80]}"
        lines.append(
            f"    - {rule_value.get('rule', '?')} "
            f"(confidence: {rule_value.get('confidence', '?')}, source: {rule_value.get('source', '?')})"
            f"{evidence_preview}"
        )
    return "\n".join(lines)


def render_records(
    records: list[dict[str, Any]],
    *,
    raw: bool,
) -> None:
    """Render voice-relevant records to stdout.

    Args:
        records (list[dict[str, Any]]): Decoded rows to render.
        raw (bool): Whether to print raw JSON payloads.

    Returns:
        None
    """

    for row in records:
        namespace_kind = str(row["namespace_kind"])
        value = row["parsed_value"]
        record_id = row.get("id", "?")
        if raw:
            print(f"--- [{namespace_kind}] owner={row['owner_id']} id={record_id} ---")
            print(json.dumps(value, indent=2, ensure_ascii=False))
            print()
            continue

        print(
            f"[{namespace_kind}] owner={row['owner_id']} id={record_id} "
            f"{'─' * 36}"
        )
        if namespace_kind == "semantic":
            print(format_semantic(value))
        elif namespace_kind == "episodic":
            print(format_episodic(value))
        elif namespace_kind == "procedural":
            print(format_procedural(value))
        else:
            print(json.dumps(value, indent=2, ensure_ascii=False))
        print()


def main() -> None:
    """Parse CLI args and render voice-relevant memory rows.

    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Inspect voice-derived OpenCouch memory")
    parser.add_argument("--owner", "-o", help="Owner/user id to inspect")
    parser.add_argument("--thread", "-t", help="Specific voice thread/session id to inspect")
    parser.add_argument(
        "--namespace",
        "-n",
        choices=["semantic", "episodic", "procedural"],
        help="Limit output to one namespace",
    )
    parser.add_argument(
        "--all-owners",
        action="store_true",
        help="List owners that currently have voice-derived memory",
    )
    parser.add_argument("--raw", action="store_true", help="Print raw JSON payloads")
    parser.add_argument(
        "--sqlite-path",
        help="Explicit legacy SQLite memory database path.",
    )
    args = parser.parse_args()

    db = find_db(args.sqlite_path)

    if args.all_owners:
        owners = list_voice_owners(db)
        if not owners:
            print("No voice-derived memory records found.")
            return
        print(
            f"{'Owner':<30} {'voice_semantic':>14} {'voice_episodic':>14} {'procedural_total':>18}"
        )
        print("-" * 80)
        for row in owners:
            print(
                f"{row['owner']:<30} "
                f"{row['voice_semantic']:>14} "
                f"{row['voice_episodic']:>14} "
                f"{row['procedural_total']:>18}"
            )
        print()
        print("procedural_total is owner-scoped and not attributable to a specific voice session.")
        return

    if not args.owner and not args.thread:
        parser.error("--owner or --thread is required (or use --all-owners)")

    rows = decode_rows(fetch_records(db, owner_id=args.owner, namespace=args.namespace))
    records = filter_voice_rows(rows, thread_id=args.thread)

    if not records:
        target = f"thread '{args.thread}'" if args.thread else f"owner '{args.owner}'"
        suffix = f" in namespace '{args.namespace}'" if args.namespace else ""
        print(f"No voice-derived memory records found for {target}{suffix}.")
        if args.thread:
            print(
                "Note: procedural rules are owner-scoped and do not store per-thread voice provenance."
            )
        return

    label = f"thread '{args.thread}'" if args.thread else f"owner '{args.owner}'"
    print(f"Found {len(records)} voice-relevant record(s) for {label}")
    if args.thread:
        print(
            "Note: procedural rules are excluded from thread mode because their per-thread provenance is not stored."
        )
    elif any(str(row["namespace_kind"]) == "procedural" for row in records):
        print(
            "Note: procedural rules are shown owner-wide; they are not attributable to a single voice session."
        )
    print()
    render_records(records, raw=args.raw)


if __name__ == "__main__":
    main()
