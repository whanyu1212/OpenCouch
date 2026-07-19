"""Tests for the Postgres-only crisis-ledger operator script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT = REPO_ROOT / "scripts" / "audit_crisis_ledger.py"


def test_help_exposes_only_postgres_connection_option() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "summary", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--database-url" in result.stdout
    assert "--backend" not in result.stdout
    assert "--sqlite-path" not in result.stdout


def test_command_requires_postgres_database_url() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "summary",
            "--date",
            "2099-01-01",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )

    assert result.returncode != 0
    assert "OPENCOUCH_CRISIS_LOG_DATABASE_URL is required" in result.stderr
