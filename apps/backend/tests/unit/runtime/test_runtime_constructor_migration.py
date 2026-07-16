"""Guard internal callers against legacy runtime constructor arguments."""

from __future__ import annotations

import ast
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_SCAN_ROOTS = ("agent", "api", "opencouch_tui", "tests")
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
_ALLOWED_FUNCTION = (
    "test_partial_grouped_persistence_config_preserves_legacy_thread_backend"
)
_ALLOWED_KEYWORDS = {"thread_database_url", "thread_persistence_backend"}


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
    allowed_call_seen = False
    for root_name in _SCAN_ROOTS:
        for path in (_BACKEND_ROOT / root_name).rglob("*.py"):
            if path.name.startswith("._"):
                continue
            relative_path = path.relative_to(_BACKEND_ROOT).as_posix()
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
                    is_allowed = (
                        relative_path.endswith("test_persistence_backend_selection.py")
                        and function.name == _ALLOWED_FUNCTION
                        and not call.args
                        and legacy == _ALLOWED_KEYWORDS
                    )
                    if is_allowed:
                        allowed_call_seen = True
                    else:
                        violations.append(
                            f"{relative_path}:{call.lineno} ({function.name}): "
                            f"positional={len(call.args)}, legacy={sorted(legacy)}"
                        )
    assert allowed_call_seen, "Remove the stale runtime migration allowlist"
    assert not violations, "Use grouped runtime configuration:\n" + "\n".join(
        violations
    )
