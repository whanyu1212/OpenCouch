"""Contracts for the grouped persistent runtime constructor."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from agent.runtime import PersistentAgentRuntime

_REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "apps" / "backend").is_dir() and (parent / "eval").is_dir()
)
_SCAN_ROOTS = (Path("apps/backend"), Path("eval"), Path("scripts"))
_IGNORED_DIRECTORY_NAMES = {".git", ".venv", "__pycache__", "node_modules"}
_LEGACY_KEYWORDS = {
    "auto_finalize_excluded",
    "crisis_log_backend",
    "crisis_log_database_url",
    "crisis_log_persistence_backend",
    "crisis_log_sqlite_path",
    "default_llm_client",
    "embedding_provider",
    "feedback_sqlite_path",
    "finalize_active_sessions_on_close",
    "memory_backend",
    "memory_database_url",
    "memory_mode",
    "memory_sqlite_path",
    "memory_store",
    "session_feedback_backend",
    "session_feedback_database_url",
    "session_feedback_persistence_backend",
    "session_sweep_interval_seconds",
    "session_timeout",
    "speculative_memory_prefetch",
    "sqlite_path",
    "text_session_backend",
    "text_session_create_tables",
    "text_session_database_url",
    "text_session_history_limit",
    "text_session_sqlite_path",
    "thread_database_url",
    "thread_persistence_backend",
}


def _is_runtime_constructor(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Name) and call.func.id == "PersistentAgentRuntime"
    ) or (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "PersistentAgentRuntime"
    )


def test_runtime_constructor_exposes_only_grouped_keyword_configuration() -> None:
    parameters = inspect.signature(PersistentAgentRuntime).parameters

    assert list(parameters) == [
        "storage_paths",
        "persistence_config",
        "dependencies",
        "behavior_config",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    assert all(parameter.default is None for parameter in parameters.values())


def test_repository_runtime_callers_use_grouped_configuration() -> None:
    violations: list[str] = []
    current_test = Path(__file__).resolve()

    for scan_root in _SCAN_ROOTS:
        root = _REPO_ROOT / scan_root
        assert root.is_dir(), f"Runtime caller scan root does not exist: {root}"
        for path in root.rglob("*.py"):
            if path.resolve() == current_test:
                continue
            relative_path = path.relative_to(_REPO_ROOT)
            if path.name.startswith("._") or any(
                part in _IGNORED_DIRECTORY_NAMES for part in relative_path.parts
            ):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call) or not _is_runtime_constructor(call):
                    continue
                legacy = sorted(
                    keyword.arg
                    for keyword in call.keywords
                    if keyword.arg in _LEGACY_KEYWORDS
                )
                if call.args or legacy:
                    violations.append(
                        f"{relative_path.as_posix()}:{call.lineno}: "
                        f"positional={len(call.args)}, legacy={legacy}"
                    )

    assert not violations, "Use grouped runtime configuration:\n" + "\n".join(
        violations
    )


def test_runtime_constructor_rejects_positional_configuration() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        PersistentAgentRuntime(":memory:")  # type: ignore[misc]


@pytest.mark.parametrize(
    "legacy_keyword",
    [
        "sqlite_path",
        "memory_sqlite_path",
        "memory_mode",
        "memory_store",
        "text_session_backend",
        "session_timeout",
    ],
)
def test_runtime_constructor_rejects_flat_configuration(
    legacy_keyword: str,
) -> None:
    with pytest.raises(TypeError, match=legacy_keyword):
        PersistentAgentRuntime(**{legacy_keyword: None})  # type: ignore[arg-type]
