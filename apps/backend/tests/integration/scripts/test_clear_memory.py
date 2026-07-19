"""Tests for the memory clearing helper script."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT = REPO_ROOT / "scripts" / "clear_memory.py"
WRAPPER = REPO_ROOT / "scripts" / "clear_memory.sh"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("clear_memory", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_clear_memory_auto_always_selects_postgres() -> None:
    module = _load_script()

    assert module.resolve_backend(requested_backend="auto") == "postgres"
    assert module.resolve_backend(requested_backend="postgres") == "postgres"
    with pytest.raises(ValueError, match="Unsupported memory backend: sqlite"):
        module.resolve_backend(requested_backend="sqlite")


def test_clear_memory_rejects_sqlite_backend() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backend",
            "sqlite",
            "--all-users",
            "--force",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid choice: 'sqlite'" in result.stderr


def test_clear_memory_rejects_sqlite_path() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sqlite-path",
            "memory.sqlite3",
            "--all-users",
            "--force",
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
def test_clear_memory_wrapper_rejects_sqlite_before_starting_docker(
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
        ["bash", str(WRAPPER), *removed_args, "--all-users", "--force"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert error_message in result.stderr
    assert not docker_marker.exists()


def test_clear_memory_accepts_postgres_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    calls: list[tuple[str, str | None]] = []

    def clear_records(database_url: str, *, owner_id: str | None = None) -> int:
        calls.append((database_url, owner_id))
        return 3

    monkeypatch.setattr(module, "clear_postgres_records", clear_records)
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
            "--force",
        ],
    )

    module.main()

    assert calls == [("postgresql://example/opencouch", "user-1")]
    assert (
        capsys.readouterr().out
        == "Deleted 3 memory record(s) for user 'user-1' from postgres.\n"
    )


def test_clear_memory_requires_confirmation_without_force() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--database-url",
            "postgresql://example/opencouch",
            "--all-users",
        ],
        input="no\n",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "from the postgres store" in result.stdout
    assert "Cancelled." in result.stdout


def test_clear_memory_requires_explicit_target() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--force"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "one of the arguments --user/-u --all-users is required" in result.stderr


def test_clear_memory_requires_postgres_dsn() -> None:
    env = os.environ.copy()
    env.pop("OPENCOUCH_MEMORY_DATABASE_URL", None)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--all-users", "--force"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Postgres clearing requires --database-url" in result.stderr
