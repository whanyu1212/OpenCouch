#!/usr/bin/env python3
"""Inspect the OpenCouch memory store for a given user.

Usage:
    python scripts/inspect_memory.py --user hy
    python scripts/inspect_memory.py --user hy --namespace semantic
    python scripts/inspect_memory.py --all-users
    python scripts/inspect_memory.py --user hy --raw
    python scripts/inspect_memory.py --backend sqlite --sqlite-path apps/backend/.store/memory.sqlite3 --all-users
    python scripts/inspect_memory.py --backend postgres --database-url postgresql://opencouch:opencouch@localhost:5432/opencouch --all-users

Run from the repo root or apps/backend/. The script supports both the
SQLite fallback memory store and the Postgres-first memory store. Use
``scripts/inspect_memory.sh`` when you want local Docker Postgres started
before inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

Backend = Literal["sqlite", "postgres"]


def resolve_backend(
    *,
    requested_backend: str,
    database_url: str | None,
) -> Backend:
    """Resolve the effective memory backend for inspection."""

    if requested_backend in {"sqlite", "postgres"}:
        return requested_backend  # type: ignore[return-value]
    if database_url:
        return "postgres"
    configured = os.getenv("OPENCOUCH_PERSISTENCE_BACKEND", "").strip().lower()
    if configured == "postgres":
        return "postgres"
    return "sqlite"


def find_sqlite_db(sqlite_path: str | None = None) -> Path:
    """Locate the local memory SQLite database."""

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


def resolve_database_url(database_url: str | None = None) -> str:
    """Resolve a Postgres memory database URL."""

    value = database_url or os.getenv("OPENCOUCH_MEMORY_DATABASE_URL")
    if value:
        return value
    print(
        "Postgres inspection requires --database-url or "
        "OPENCOUCH_MEMORY_DATABASE_URL.",
        file=sys.stderr,
    )
    sys.exit(1)


def _decode_record_value(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    if isinstance(value, dict):
        return value
    return {"value": value}


def fetch_sqlite_records(
    db: Path,
    *,
    owner_id: str | None = None,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch SQLite memory records, optionally filtered by owner and namespace."""

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
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
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def fetch_postgres_records(
    database_url: str,
    *,
    owner_id: str | None = None,
    namespace: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch Postgres memory records, optionally filtered by owner and namespace."""

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        print(
            "Postgres inspection requires psycopg. Run through apps/backend/.venv "
            "or use scripts/inspect_memory.sh.",
            file=sys.stderr,
        )
        sys.exit(1)

    query = "SELECT * FROM memory_records WHERE TRUE"
    params: list[str] = []

    if owner_id:
        query += " AND owner_id = %s"
        params.append(owner_id)
    if namespace:
        query += " AND namespace_kind = %s"
        params.append(namespace)

    query += " ORDER BY insertion_order"

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall())


def list_sqlite_users(db: Path) -> list[dict[str, Any]]:
    """List all SQLite users with record counts per namespace."""

    conn = sqlite3.connect(str(db))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT owner_id, namespace_kind, COUNT(*) as cnt "
            "FROM memory_records GROUP BY owner_id, namespace_kind "
            "ORDER BY owner_id, namespace_kind"
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return _group_user_counts(rows)


def list_postgres_users(database_url: str) -> list[dict[str, Any]]:
    """List all Postgres users with record counts per namespace."""

    try:
        import psycopg
    except ImportError:
        print(
            "Postgres inspection requires psycopg. Run through apps/backend/.venv "
            "or use scripts/inspect_memory.sh.",
            file=sys.stderr,
        )
        sys.exit(1)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT owner_id, namespace_kind, COUNT(*) as cnt "
                "FROM memory_records GROUP BY owner_id, namespace_kind "
                "ORDER BY owner_id, namespace_kind"
            )
            return _group_user_counts(cursor.fetchall())


def _group_user_counts(rows: Sequence[Sequence[Any]]) -> list[dict[str, Any]]:
    users: dict[str, dict[str, int]] = {}
    for owner, namespace, count in rows:
        users.setdefault(str(owner), {})[str(namespace)] = int(count)
    return [{"user": user, **counts} for user, counts in users.items()]


def format_semantic(value: dict[str, Any]) -> str:
    """Format a semantic fact for display."""

    subj = value.get("subject", {})
    obj = value.get("object", {})
    pred = value.get("predicate", "")
    quote = value.get("evidence_quote", "")
    cat = value.get("category", "")
    confidence = value.get("confidence", "")

    subj_id = subj.get("identifier", "?") if isinstance(subj, dict) else str(subj)
    obj_id = obj.get("identifier", "?") if isinstance(obj, dict) else str(obj)
    pred_label = str(pred).replace("_", " ")

    return (
        f"  [{cat}] {subj_id} {pred_label} {obj_id}\n"
        f'  Quote: "{quote}"\n'
        "  Confidence: "
        f"{confidence} | Session: {value.get('source_session_id', '?')} "
        f"turn {value.get('source_turn_index', '?')}"
    )


def format_episodic(value: dict[str, Any]) -> str:
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
        "  Session: "
        f"{value.get('session_id', '?')} | {value.get('turn_count', '?')} turns | "
        f"{value.get('duration_seconds', 0) // 60}min",
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
    if isinstance(ctx, dict):
        ctx_parts = []
        for key, raw_value in ctx.items():
            if key in {"modality", "therapeutic_approach"} or raw_value is None:
                continue
            value_text = raw_value
            if isinstance(raw_value, list):
                value_text = ", ".join(str(item) for item in raw_value) or None
            if value_text:
                ctx_parts.append(f"{key}={value_text}")
        if ctx_parts:
            lines.append(f"  Context: {'; '.join(ctx_parts)}")
    return "\n".join(lines)


def format_procedural(value: dict[str, Any]) -> str:
    """Format a procedural profile for display."""

    rules = value.get("rules", [])
    recall = value.get("proactive_recall_enabled", False)
    lines = [f"  Recall: {'on' if recall else 'off'}", f"  Rules ({len(rules)}):"]
    for rule in rules:
        rule_obj = rule if isinstance(rule, dict) else {"rule": str(rule)}
        lines.append(
            f"    - {rule_obj.get('rule', '?')} "
            f"(confidence: {rule_obj.get('confidence', '?')})"
        )
    return "\n".join(lines)


def _print_user_counts(users: list[dict[str, Any]], *, backend: Backend) -> None:
    if not users:
        print(f"No records found in {backend} memory store.")
        return
    print(f"Backend: {backend}")
    print(f"{'User':<30} {'semantic':>8} {'episodic':>8} {'procedural':>10}")
    print("-" * 60)
    for user in users:
        print(
            f"{user['user']:<30} "
            f"{user.get('semantic', 0):>8} "
            f"{user.get('episodic', 0):>8} "
            f"{user.get('procedural', 0):>10}"
        )


def _print_records(
    records: list[dict[str, Any]],
    *,
    backend: Backend,
    owner_id: str,
    namespace: str | None,
    raw: bool,
) -> None:
    if not records:
        print(
            f"No records found for user '{owner_id}'"
            + (f" in namespace '{namespace}'" if namespace else "")
            + f" in {backend} memory store."
        )
        return

    print(f"Backend: {backend}")
    print(f"Found {len(records)} record(s) for user '{owner_id}'")
    print()

    for record in records:
        ns = record["namespace_kind"]
        value = _decode_record_value(record["value"])

        if raw:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OpenCouch memory store")
    parser.add_argument("--user", "-u", help="Filter by user/owner ID")
    parser.add_argument(
        "--namespace",
        "-n",
        choices=["semantic", "episodic", "procedural"],
        help="Filter by memory namespace",
    )
    parser.add_argument(
        "--all-users",
        action="store_true",
        help="List all users and record counts",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Output raw JSON instead of formatted",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sqlite", "postgres"],
        default="auto",
        help=(
            "Memory backend to inspect. 'auto' uses --database-url or "
            "OPENCOUCH_PERSISTENCE_BACKEND, falling back to SQLite."
        ),
    )
    parser.add_argument(
        "--sqlite-path",
        help="Explicit SQLite memory database path.",
    )
    parser.add_argument(
        "--database-url",
        help="Postgres memory database URL. Defaults to OPENCOUCH_MEMORY_DATABASE_URL.",
    )
    args = parser.parse_args()

    backend = resolve_backend(
        requested_backend=args.backend,
        database_url=args.database_url,
    )

    sqlite_db: Path | None = None
    database_url: str | None = None
    if backend == "sqlite":
        sqlite_db = find_sqlite_db(args.sqlite_path)
    else:
        database_url = resolve_database_url(args.database_url)

    if args.all_users:
        users = (
            list_sqlite_users(sqlite_db)
            if backend == "sqlite" and sqlite_db is not None
            else list_postgres_users(database_url or "")
        )
        _print_user_counts(users, backend=backend)
        return

    if not args.user:
        parser.error("--user is required (or use --all-users)")

    records = (
        fetch_sqlite_records(
            sqlite_db,
            owner_id=args.user,
            namespace=args.namespace,
        )
        if backend == "sqlite" and sqlite_db is not None
        else fetch_postgres_records(
            database_url or "",
            owner_id=args.user,
            namespace=args.namespace,
        )
    )
    _print_records(
        records,
        backend=backend,
        owner_id=args.user,
        namespace=args.namespace,
        raw=args.raw,
    )


if __name__ == "__main__":
    main()
