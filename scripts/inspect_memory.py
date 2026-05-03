#!/usr/bin/env python3
"""Inspect the legacy SQLite OpenCouch memory store for a given user.

Usage:
    python scripts/inspect_memory.py --user hy
    python scripts/inspect_memory.py --user hy --namespace semantic
    python scripts/inspect_memory.py --user hy --namespace episodic
    python scripts/inspect_memory.py --user hy --namespace procedural
    python scripts/inspect_memory.py --all-users
    python scripts/inspect_memory.py --user hy --raw
    python scripts/inspect_memory.py --sqlite-path apps/backend/.store/memory.sqlite3 --all-users

Run from the repo root or apps/backend/. This is a legacy SQLite
inspector; Postgres-first environments should query Postgres directly.
The script auto-detects the .store/ directory location unless
``--sqlite-path`` is provided.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def find_db(sqlite_path: str | None = None) -> Path:
    """Locate the legacy memory SQLite database."""

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
    for p in candidates:
        if p.exists():
            return p
    print("Could not find memory.sqlite3 in .store/", file=sys.stderr)
    sys.exit(1)


def fetch_records(
    db: Path,
    *,
    owner_id: str | None = None,
    namespace: str | None = None,
) -> list[dict]:
    """Fetch memory records, optionally filtered by owner and namespace."""

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM memory_records WHERE 1=1"
    params: list[str] = []

    if owner_id:
        query += " AND owner_id = ?"
        params.append(owner_id)
    if namespace:
        query += " AND namespace_kind = ?"
        params.append(namespace)

    query += " ORDER BY insertion_order"
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def list_users(db: Path) -> list[dict]:
    """List all users with record counts per namespace."""

    conn = sqlite3.connect(str(db))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT owner_id, namespace_kind, COUNT(*) as cnt "
        "FROM memory_records GROUP BY owner_id, namespace_kind "
        "ORDER BY owner_id, namespace_kind"
    )
    rows = cursor.fetchall()
    conn.close()

    users: dict[str, dict[str, int]] = {}
    for owner, ns, cnt in rows:
        users.setdefault(owner, {})[ns] = cnt
    return [{"user": u, **counts} for u, counts in users.items()]


def format_semantic(value: dict) -> str:
    """Format a semantic fact for display."""

    subj = value.get("subject", {})
    obj = value.get("object", {})
    pred = value.get("predicate", "")
    quote = value.get("evidence_quote", "")
    cat = value.get("category", "")
    confidence = value.get("confidence", "")

    subj_id = subj.get("identifier", "?") if isinstance(subj, dict) else str(subj)
    obj_id = obj.get("identifier", "?") if isinstance(obj, dict) else str(obj)
    pred_label = pred.replace("_", " ")

    return (
        f"  [{cat}] {subj_id} {pred_label} {obj_id}\n"
        f"  Quote: \"{quote}\"\n"
        f"  Confidence: {confidence} | Session: {value.get('source_session_id', '?')} turn {value.get('source_turn_index', '?')}"
    )


def format_episodic(value: dict) -> str:
    """Format an episodic session arc for display."""

    themes = ", ".join(value.get("primary_themes", []))
    mood = value.get("mood_arc", {})
    opened = mood.get("opened", "?") if isinstance(mood, dict) else "?"
    closed = mood.get("closed", "?") if isinstance(mood, dict) else "?"
    open_loops = value.get("open_loops", [])
    resolved = value.get("resolved_threads", [])
    approach = value.get("approach_used")
    ctx = value.get("approach_context")

    lines = [
        f"  Session: {value.get('session_id', '?')} | {value.get('turn_count', '?')} turns | {value.get('duration_seconds', 0) // 60}min",
        f"  Themes: {themes or 'none'}",
        f"  Mood: {opened} → {closed}",
        f"  Summary: {value.get('summary', '?')}",
    ]
    if open_loops:
        lines.append(f"  Open loops: {'; '.join(open_loops)}")
    if resolved:
        lines.append(f"  Resolved: {'; '.join(resolved)}")
    if approach and approach != "none":
        lines.append(f"  Approach: {approach}")
    if ctx:
        ctx_parts = []
        for k, v in ctx.items():
            if k == "modality" or k == "therapeutic_approach" or v is None:
                continue
            if isinstance(v, list):
                v = ", ".join(str(i) for i in v) if v else None
            if v:
                ctx_parts.append(f"{k}={v}")
        if ctx_parts:
            lines.append(f"  Context: {'; '.join(ctx_parts)}")
    return "\n".join(lines)


def format_procedural(value: dict) -> str:
    """Format a procedural profile for display."""

    rules = value.get("rules", [])
    recall = value.get("proactive_recall_enabled", False)
    lines = [f"  Recall: {'on' if recall else 'off'}", f"  Rules ({len(rules)}):"]
    for rule in rules:
        r = rule if isinstance(rule, dict) else {"rule": str(rule)}
        lines.append(f"    - {r.get('rule', '?')} (confidence: {r.get('confidence', '?')})")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OpenCouch memory store")
    parser.add_argument("--user", "-u", help="Filter by user/owner ID")
    parser.add_argument(
        "--namespace", "-n",
        choices=["semantic", "episodic", "procedural"],
        help="Filter by memory namespace",
    )
    parser.add_argument("--all-users", action="store_true", help="List all users and record counts")
    parser.add_argument("--raw", action="store_true", help="Output raw JSON instead of formatted")
    parser.add_argument(
        "--sqlite-path",
        help="Explicit legacy SQLite memory database path.",
    )
    args = parser.parse_args()

    db = find_db(args.sqlite_path)

    if args.all_users:
        users = list_users(db)
        if not users:
            print("No records found.")
            return
        print(f"{'User':<30} {'semantic':>8} {'episodic':>8} {'procedural':>10}")
        print("-" * 60)
        for u in users:
            print(
                f"{u['user']:<30} "
                f"{u.get('semantic', 0):>8} "
                f"{u.get('episodic', 0):>8} "
                f"{u.get('procedural', 0):>10}"
            )
        return

    if not args.user:
        parser.error("--user is required (or use --all-users)")

    records = fetch_records(db, owner_id=args.user, namespace=args.namespace)

    if not records:
        print(f"No records found for user '{args.user}'"
              + (f" in namespace '{args.namespace}'" if args.namespace else ""))
        return

    print(f"Found {len(records)} record(s) for user '{args.user}'")
    print()

    for record in records:
        ns = record["namespace_kind"]
        value = json.loads(record["value"])

        if args.raw:
            print(f"--- [{ns}] ---")
            print(json.dumps(value, indent=2, ensure_ascii=False))
            print()
            continue

        print(f"[{ns}] {'─' * 50}")
        if ns == "semantic":
            print(format_semantic(value))
        elif ns == "episodic":
            print(format_episodic(value))
        elif ns == "procedural":
            print(format_procedural(value))
        else:
            print(f"  {json.dumps(value, indent=2, ensure_ascii=False)[:300]}")
        print()


if __name__ == "__main__":
    main()
