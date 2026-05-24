#!/usr/bin/env python3
"""Clear OpenCouch memory records for a user or all users.

Usage:
    python scripts/clear_memory.py --backend sqlite --sqlite-path apps/backend/.store/memory.sqlite3 --user hy --force
    python scripts/clear_memory.py --backend sqlite --sqlite-path apps/backend/.store/memory.sqlite3 --all-users --force
    python scripts/clear_memory.py --backend postgres --database-url postgresql://opencouch:opencouch@localhost:5432/opencouch --all-users --force

Use ``scripts/clear_memory.sh`` when you want local Docker Postgres started
before clearing. This script deletes rows from ``memory_records`` only; it
preserves the database schema.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Literal

Backend = Literal["sqlite", "postgres"]


def resolve_backend(
    *,
    requested_backend: str,
    database_url: str | None,
) -> Backend:
    """Resolve the effective memory backend for clearing."""

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
        "Postgres clearing requires --database-url or "
        "OPENCOUCH_MEMORY_DATABASE_URL.",
        file=sys.stderr,
    )
    sys.exit(1)


def _confirm_or_exit(
    *,
    backend: Backend,
    owner_id: str | None,
    force: bool,
) -> None:
    if force:
        return

    target = f"user '{owner_id}'" if owner_id else "ALL users"
    response = input(
        f"This will permanently delete memory records for {target} "
        f"from the {backend} store. Proceed? Type 'yes' to continue: "
    )
    if response.strip().lower() != "yes":
        print("Cancelled.")
        sys.exit(1)


def clear_sqlite_records(
    db: Path,
    *,
    owner_id: str | None = None,
) -> int:
    """Delete SQLite memory records and return the number of deleted rows."""

    conn = sqlite3.connect(str(db))
    try:
        cursor = conn.cursor()
        if owner_id:
            cursor.execute("DELETE FROM memory_records WHERE owner_id = ?", (owner_id,))
        else:
            cursor.execute("DELETE FROM memory_records")
        deleted = cursor.rowcount if cursor.rowcount is not None else 0
        conn.commit()
        conn.execute("VACUUM")
        return int(deleted)
    finally:
        conn.close()


def clear_postgres_records(
    database_url: str,
    *,
    owner_id: str | None = None,
) -> int:
    """Delete Postgres memory records and return the number of deleted rows."""

    try:
        import psycopg
    except ImportError:
        print(
            "Postgres clearing requires psycopg. Run through apps/backend/.venv "
            "or use scripts/clear_memory.sh.",
            file=sys.stderr,
        )
        sys.exit(1)

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cursor:
            if owner_id:
                cursor.execute(
                    "DELETE FROM memory_records WHERE owner_id = %s",
                    (owner_id,),
                )
            else:
                cursor.execute("DELETE FROM memory_records")
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
        conn.commit()
    return int(deleted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear OpenCouch memory records")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--user", "-u", help="Clear records for one user/owner ID")
    target.add_argument(
        "--all-users",
        action="store_true",
        help="Clear records for all users",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "sqlite", "postgres"],
        default="auto",
        help=(
            "Memory backend to clear. 'auto' uses --database-url or "
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip interactive confirmation.",
    )
    args = parser.parse_args()

    backend = resolve_backend(
        requested_backend=args.backend,
        database_url=args.database_url,
    )
    owner_id = args.user if not args.all_users else None

    _confirm_or_exit(backend=backend, owner_id=owner_id, force=args.force)

    if backend == "sqlite":
        deleted = clear_sqlite_records(
            find_sqlite_db(args.sqlite_path),
            owner_id=owner_id,
        )
    else:
        deleted = clear_postgres_records(
            resolve_database_url(args.database_url),
            owner_id=owner_id,
        )

    target_label = f"user '{owner_id}'" if owner_id else "all users"
    print(f"Deleted {deleted} memory record(s) for {target_label} from {backend}.")


if __name__ == "__main__":
    main()
