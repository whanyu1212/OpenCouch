"""Tests for the memory inspection helper script."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT = REPO_ROOT / "scripts" / "inspect_memory.py"
WRAPPER = REPO_ROOT / "scripts" / "inspect_memory.sh"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inspect_memory", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inspect_memory_auto_always_selects_postgres() -> None:
    module = _load_script()

    assert module.resolve_backend(requested_backend="auto") == "postgres"
    assert module.resolve_backend(requested_backend="postgres") == "postgres"
    with pytest.raises(ValueError, match="Unsupported memory backend: sqlite"):
        module.resolve_backend(requested_backend="sqlite")


def test_inspect_memory_rejects_sqlite_backend() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--backend", "sqlite", "--all-users"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid choice: 'sqlite'" in result.stderr


def test_inspect_memory_rejects_sqlite_path() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sqlite-path",
            "memory.sqlite3",
            "--all-users",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "unrecognized arguments: --sqlite-path memory.sqlite3" in result.stderr


@pytest.mark.parametrize(
    ("removed_args", "error_message"),
    [
        (("--backend", "sqlite"), "Unsupported memory backend: sqlite"),
        (("--backend=sqlite",), "Unsupported memory backend: sqlite"),
        (("--sqlite-path", "memory.sqlite3"), "SQLite memory tooling has been removed"),
        (("--sqlite-path=memory.sqlite3",), "SQLite memory tooling has been removed"),
    ],
)
def test_inspect_memory_wrapper_rejects_sqlite_before_starting_docker(
    tmp_path: Path,
    removed_args: tuple[str, ...],
    error_message: str,
) -> None:
    docker_marker = tmp_path / "docker-called"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        '#!/usr/bin/env bash\ntouch "$DOCKER_MARKER"\n', encoding="utf-8"
    )
    fake_docker.chmod(0o755)
    env = os.environ.copy()
    env["DOCKER_MARKER"] = str(docker_marker)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        ["bash", str(WRAPPER), *removed_args, "--all-users"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert error_message in result.stderr
    assert not docker_marker.exists()


def test_inspect_memory_accepts_postgres_arguments_and_formats_records(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    calls: list[tuple[str, str | None, str | None]] = []

    def fetch_records(
        database_url: str,
        *,
        owner_id: str | None = None,
        namespace: str | None = None,
    ) -> list[dict[str, object]]:
        calls.append((database_url, owner_id, namespace))
        return [
            {
                "namespace_kind": "semantic",
                "value": {
                    "category": "trigger",
                    "subject": {"identifier": "user-1"},
                    "predicate": "WORRIES_ABOUT",
                    "object": {"identifier": "presentations"},
                    "evidence_quote": "Presentations make me anxious.",
                    "confidence": "high",
                    "source_session_id": "session-1",
                    "source_turn_index": 0,
                },
            }
        ]

    monkeypatch.setattr(module, "fetch_postgres_records", fetch_records)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--backend",
            "postgres",
            "--database-url",
            "postgresql://example/opencouch",
            "--user",
            "user-1",
            "--namespace",
            "semantic",
        ],
    )

    module.main()

    assert calls == [("postgresql://example/opencouch", "user-1", "semantic")]
    output = capsys.readouterr().out
    assert "Backend: postgres" in output
    assert "Found 1 record(s) for user 'user-1'" in output
    assert "[trigger] user-1 WORRIES ABOUT presentations" in output
    assert 'Quote: "Presentations make me anxious."' in output


def test_inspect_memory_formats_postgres_user_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    monkeypatch.setenv(
        "OPENCOUCH_MEMORY_DATABASE_URL", "postgresql://example/opencouch"
    )
    monkeypatch.setattr(
        module,
        "list_postgres_users",
        lambda database_url: [{"user": "user-1", "semantic": 2, "episodic": 1}],
    )
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--all-users"])

    module.main()

    output = capsys.readouterr().out
    assert "Backend: postgres" in output
    assert "User                           semantic episodic procedural" in output
    assert "user-1                                2        1          0" in output


def test_inspect_memory_requires_postgres_dsn() -> None:
    env = os.environ.copy()
    env.pop("OPENCOUCH_MEMORY_DATABASE_URL", None)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--all-users"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Postgres inspection requires --database-url" in result.stderr
