"""Tests for the memory clearing helper script."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT = REPO_ROOT / "scripts" / "clear_memory.py"


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
        for index, owner_id in enumerate(("user-1", "user-2"), start=1):
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
                    f"memory-{index}",
                    owner_id,
                    "semantic",
                    "trigger",
                    json.dumps({"evidence_quote": f"memory for {owner_id}"}),
                    "2026-05-23T00:00:00Z",
                    "2026-05-23T00:00:00Z",
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _record_count(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(conn.execute("SELECT COUNT(*) FROM memory_records").fetchone()[0])
    finally:
        conn.close()


def _owner_count(path: Path, owner_id: str) -> int:
    conn = sqlite3.connect(path)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM memory_records WHERE owner_id = ?",
                (owner_id,),
            ).fetchone()[0]
        )
    finally:
        conn.close()


def test_clear_memory_deletes_one_sqlite_user_with_force(tmp_path: Path) -> None:
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
            "--force",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Deleted 1 memory record(s) for user 'user-1' from sqlite." in result.stdout
    assert _owner_count(db_path, "user-1") == 0
    assert _owner_count(db_path, "user-2") == 1


def test_clear_memory_deletes_all_sqlite_users_with_force(tmp_path: Path) -> None:
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
            "--force",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Deleted 2 memory record(s) for all users from sqlite." in result.stdout
    assert _record_count(db_path) == 0


def test_clear_memory_requires_confirmation_without_force(tmp_path: Path) -> None:
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
        input="no\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Cancelled." in result.stdout
    assert _record_count(db_path) == 2


def test_clear_memory_requires_explicit_target() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "sqlite", "--force"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "one of the arguments --user/-u --all-users is required" in result.stderr


def test_clear_memory_auto_uses_database_url_for_postgres(monkeypatch) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("clear_memory", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.delenv("OPENCOUCH_PERSISTENCE_BACKEND", raising=False)

    assert (
        module.resolve_backend(
            requested_backend="auto",
            database_url="postgresql://example",
        )
        == "postgres"
    )
    assert (
        module.resolve_backend(requested_backend="auto", database_url=None) == "sqlite"
    )
