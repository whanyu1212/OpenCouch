"""Guard internal callers against legacy runtime constructor arguments."""

from __future__ import annotations

import ast
from pathlib import Path


def _find_repo_root() -> Path:
    """Find the repository root from stable project directory markers."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "apps" / "backend").is_dir() and (parent / "eval").is_dir():
            return parent
    raise RuntimeError("Could not locate the OpenCouch repository root")


_REPO_ROOT = _find_repo_root()
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


def test_internal_runtime_callers_use_grouped_configuration() -> None:
    """New internal callers must use grouped configuration objects."""

    violations: list[str] = []
    for scan_root in _SCAN_ROOTS:
        root = _REPO_ROOT / scan_root
        assert root.is_dir(), f"Runtime caller scan root does not exist: {root}"
        for path in root.rglob("*.py"):
            relative_path = path.relative_to(_REPO_ROOT)
            if path.name.startswith("._") or any(
                part in _IGNORED_DIRECTORY_NAMES for part in relative_path.parts
            ):
                continue
            relative_path_text = relative_path.as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for call in ast.walk(function):
                    if not isinstance(call, ast.Call) or not _is_runtime_constructor(
                        call
                    ):
                        continue
                    legacy = {
                        keyword.arg
                        for keyword in call.keywords
                        if keyword.arg in _LEGACY_KEYWORDS
                    }
                    if not call.args and not legacy:
                        continue
                    violations.append(
                        f"{relative_path_text}:{call.lineno} ({function.name}): "
                        f"positional={len(call.args)}, legacy={sorted(legacy)}"
                    )
    assert not violations, "Use grouped runtime configuration:\n" + "\n".join(
        violations
    )
