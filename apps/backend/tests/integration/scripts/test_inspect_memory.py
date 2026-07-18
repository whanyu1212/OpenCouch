"""Tests for the memory inspection helper script."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT = REPO_ROOT / "scripts" / "inspect_memory.py"


def _create_memory_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE memory_records (
                insertion_order INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL,
                owner_id TEXT NOT NULL,
                namespace_kind TEXT NOT NULL,
                category TEXT,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_referenced_at TEXT NOT NULL,
                dormant_at TEXT,
                user_visible INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        conn.execute(
            """
            INSERT INTO memory_records (
                id,
                owner_id,
                namespace_kind,
                category,
                value,
                created_at,
                last_referenced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "memory-1",
                "user-1",
                "semantic",
                "trigger",
                json.dumps(
                    {
                        "category": "trigger",
                        "subject": {"identifier": "user-1"},
                        "predicate": "WORRIES_ABOUT",
                        "object": {"identifier": "presentations"},
                        "evidence_quote": "Presentations make me anxious.",
                        "confidence": "high",
                        "source_session_id": "session-1",
                        "source_turn_index": 0,
                    }
                ),
                "2026-05-23T00:00:00Z",
                "2026-05-23T00:00:00Z",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_inspect_memory_lists_sqlite_users(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    _create_memory_db(db_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "sqlite",
            "--sqlite-path",
            str(db_path),
            "--all-users",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Backend: sqlite" in result.stdout
    assert "user-1" in result.stdout
    assert "semantic" in result.stdout


def test_inspect_memory_reads_sqlite_user_records(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    _create_memory_db(db_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "sqlite",
            "--sqlite-path",
            str(db_path),
            "--user",
            "user-1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Found 1 record(s) for user 'user-1'" in result.stdout
    assert "[trigger] user-1 WORRIES ABOUT presentations" in result.stdout
    assert 'Quote: "Presentations make me anxious."' in result.stdout


def test_inspect_memory_auto_always_selects_postgres() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("inspect_memory", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.resolve_backend(requested_backend="auto") == "postgres"
    assert module.resolve_backend(requested_backend="postgres") == "postgres"
    assert module.resolve_backend(requested_backend="sqlite") == "sqlite"


def test_inspect_memory_auto_requires_postgres_dsn_even_with_legacy_sqlite(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / ".store"
    store_dir.mkdir()
    _create_memory_db(store_dir / "memory.sqlite3")
    env = os.environ.copy()
    env["OPENCOUCH_PERSISTENCE_BACKEND"] = "sqlite"
    env.pop("OPENCOUCH_MEMORY_DATABASE_URL", None)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--all-users"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Postgres inspection requires --database-url" in result.stderr
    assert "Backend: sqlite" not in result.stdout


def test_inspect_memory_sqlite_requires_explicit_path() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "sqlite", "--all-users"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--backend sqlite requires --sqlite-path" in result.stderr


def test_inspect_memory_sqlite_path_requires_explicit_backend(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.sqlite3"
    _create_memory_db(db_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sqlite-path",
            str(db_path),
            "--all-users",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--sqlite-path requires --backend sqlite" in result.stderr


def test_inspect_memory_help_marks_sqlite_as_migration_only() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    help_text = " ".join(result.stdout.split())
    assert "'auto' always selects Postgres" in help_text
    assert "SQLite is migration-only" in help_text
