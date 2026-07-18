"""Configuration models and resolution for the persistent agent runtime."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

from agent.audit.crisis_log import CrisisLogBackend
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.memory.modes import MemoryMode
from agent.memory.providers.embeddings import EmbeddingProvider
from agent.memory.store import MemoryStore
from agent.runtime.session_store import TextSessionBackend
from llm.base import BaseLLMClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
_STORE_DIR = BACKEND_ROOT / ".store"
DEFAULT_THREAD_DB_PATH = _STORE_DIR / "threads.sqlite3"
DEFAULT_TEXT_SESSION_DB_PATH = _STORE_DIR / "text_sessions.sqlite3"
DEFAULT_CRISIS_LOG_DB_PATH = _STORE_DIR / "crisis.sqlite3"
DEFAULT_FEEDBACK_DB_PATH = _STORE_DIR / "session_feedback.sqlite3"
SESSION_TIMEOUT = timedelta(minutes=20)
_UNSET = object()
_LEGACY_STORAGE_PATH_WARNING = (
    "PersistentAgentRuntime direct SQLite path arguments are deprecated."
)
_LEGACY_SQLITE_DURABLE_MESSAGE = (
    "Durable SQLite SDK-session persistence is legacy and disabled unless "
    "explicitly allowed. Use Postgres for durable runtime persistence, "
    "or set allow_legacy_sqlite=True for temporary migration-only usage."
)
_REMOVED_MEMORY_SQLITE_MESSAGE = (
    "SQLite memory persistence has been removed. Use memory_backend='postgres' "
    "with a Postgres database URL, or inject a MemoryStore for ephemeral tests."
)
_REMOVED_THREAD_SQLITE_MESSAGE = (
    "SQLite runtime-state and active-session persistence has been removed. "
    "Use thread_persistence_backend='postgres' with a Postgres database URL."
)
_REMOVED_AUDIT_FEEDBACK_SQLITE_MESSAGE = (
    "SQLite crisis-audit and session-feedback persistence has been removed. "
    "Use Postgres for durable application persistence; incognito mode and "
    "explicit in-memory test backends remain available."
)


def _is_non_default_sqlite_path(
    sqlite_path: object,
    *,
    default_path: str | Path,
) -> bool:
    """Return whether a supplied SQLite path is a concrete non-default path."""

    if sqlite_path is _UNSET or sqlite_path is None:
        return False
    return Path(str(sqlite_path)).expanduser().resolve(strict=False) != Path(
        default_path
    ).expanduser().resolve(strict=False)


@dataclass(slots=True)
class RuntimeStoragePaths:
    """Grouped compatibility paths; only SDK-session SQLite remains effective."""

    sqlite_path: str | Path | object = _UNSET
    crisis_log_sqlite_path: str | Path | object = _UNSET
    feedback_sqlite_path: str | Path | object = _UNSET
    text_session_sqlite_path: str | Path | None | object = _UNSET


@dataclass(slots=True)
class _ResolvedRuntimeStoragePaths:
    """Constructor-ready SQLite path values for runtime-owned storage."""

    sqlite_path: str | Path
    crisis_log_sqlite_path: str | Path
    feedback_sqlite_path: str | Path
    text_session_sqlite_path: str | Path | None
    sqlite_path_configured: bool
    crisis_log_sqlite_path_configured: bool
    feedback_sqlite_path_configured: bool
    text_session_sqlite_path_configured: bool


def _resolve_runtime_storage_paths(
    *,
    sqlite_path: str | Path | object,
    storage_paths: RuntimeStoragePaths | None,
    memory_sqlite_path: str | Path | object,
    crisis_log_sqlite_path: str | Path | object,
    feedback_sqlite_path: str | Path | object,
    text_session_sqlite_path: str | Path | None | object,
) -> _ResolvedRuntimeStoragePaths:
    """Resolve default, legacy, and grouped SQLite path arguments."""

    legacy_path_args = (
        ("sqlite_path", sqlite_path),
        ("memory_sqlite_path", memory_sqlite_path),
        ("crisis_log_sqlite_path", crisis_log_sqlite_path),
        ("feedback_sqlite_path", feedback_sqlite_path),
        ("text_session_sqlite_path", text_session_sqlite_path),
    )
    supplied_legacy_path_args = [
        name for name, value in legacy_path_args if value is not _UNSET
    ]
    if supplied_legacy_path_args:
        guidance: list[str] = []
        if memory_sqlite_path is not _UNSET:
            guidance.append(
                "memory_sqlite_path is ignored because SQLite memory persistence "
                "has been removed."
            )
        if any(name != "memory_sqlite_path" for name in supplied_legacy_path_args):
            guidance.append(
                "Use storage_paths=RuntimeStoragePaths(...) for supported path "
                "overrides."
            )
        warnings.warn(
            f"{_LEGACY_STORAGE_PATH_WARNING} {' '.join(guidance)} Legacy args: "
            f"{', '.join(supplied_legacy_path_args)}.",
            DeprecationWarning,
            stacklevel=3,
        )

    sqlite_path_configured = _is_non_default_sqlite_path(
        sqlite_path,
        default_path=DEFAULT_THREAD_DB_PATH,
    )
    crisis_log_sqlite_path_configured = _is_non_default_sqlite_path(
        crisis_log_sqlite_path,
        default_path=DEFAULT_CRISIS_LOG_DB_PATH,
    )
    feedback_sqlite_path_configured = _is_non_default_sqlite_path(
        feedback_sqlite_path,
        default_path=DEFAULT_FEEDBACK_DB_PATH,
    )
    text_session_sqlite_path_configured = _is_non_default_sqlite_path(
        text_session_sqlite_path,
        default_path=DEFAULT_TEXT_SESSION_DB_PATH,
    )

    resolved_sqlite_path = (
        DEFAULT_THREAD_DB_PATH
        if sqlite_path is _UNSET
        else cast(str | Path, sqlite_path)
    )
    resolved_crisis_log_sqlite_path = (
        DEFAULT_CRISIS_LOG_DB_PATH
        if crisis_log_sqlite_path is _UNSET
        else cast(str | Path, crisis_log_sqlite_path)
    )
    resolved_feedback_sqlite_path = (
        DEFAULT_FEEDBACK_DB_PATH
        if feedback_sqlite_path is _UNSET
        else cast(str | Path, feedback_sqlite_path)
    )
    resolved_text_session_sqlite_path = (
        None
        if text_session_sqlite_path is _UNSET
        else cast(str | Path | None, text_session_sqlite_path)
    )

    if storage_paths is not None:
        if storage_paths.sqlite_path is not _UNSET:
            resolved_sqlite_path = cast(str | Path, storage_paths.sqlite_path)
            sqlite_path_configured = _is_non_default_sqlite_path(
                storage_paths.sqlite_path,
                default_path=DEFAULT_THREAD_DB_PATH,
            )
        if storage_paths.crisis_log_sqlite_path is not _UNSET:
            resolved_crisis_log_sqlite_path = cast(
                str | Path,
                storage_paths.crisis_log_sqlite_path,
            )
            crisis_log_sqlite_path_configured = _is_non_default_sqlite_path(
                storage_paths.crisis_log_sqlite_path,
                default_path=DEFAULT_CRISIS_LOG_DB_PATH,
            )
        if storage_paths.feedback_sqlite_path is not _UNSET:
            resolved_feedback_sqlite_path = cast(
                str | Path,
                storage_paths.feedback_sqlite_path,
            )
            feedback_sqlite_path_configured = _is_non_default_sqlite_path(
                storage_paths.feedback_sqlite_path,
                default_path=DEFAULT_FEEDBACK_DB_PATH,
            )
        if storage_paths.text_session_sqlite_path is not _UNSET:
            resolved_text_session_sqlite_path = cast(
                str | Path | None,
                storage_paths.text_session_sqlite_path,
            )
            text_session_sqlite_path_configured = _is_non_default_sqlite_path(
                storage_paths.text_session_sqlite_path,
                default_path=DEFAULT_TEXT_SESSION_DB_PATH,
            )

    return _ResolvedRuntimeStoragePaths(
        sqlite_path=resolved_sqlite_path,
        crisis_log_sqlite_path=resolved_crisis_log_sqlite_path,
        feedback_sqlite_path=resolved_feedback_sqlite_path,
        text_session_sqlite_path=resolved_text_session_sqlite_path,
        sqlite_path_configured=sqlite_path_configured,
        crisis_log_sqlite_path_configured=crisis_log_sqlite_path_configured,
        feedback_sqlite_path_configured=feedback_sqlite_path_configured,
        text_session_sqlite_path_configured=text_session_sqlite_path_configured,
    )


@dataclass(slots=True)
class RuntimePersistenceConfig:
    """Grouped backend and database URL configuration for the runtime."""

    memory_mode: MemoryMode | object = _UNSET
    memory_backend: Literal["postgres"] | object = _UNSET
    memory_database_url: str | None | object = _UNSET
    thread_persistence_backend: Literal["memory", "sqlite", "postgres"] | object = (
        _UNSET
    )
    thread_database_url: str | None | object = _UNSET
    crisis_log_persistence_backend: Literal["sqlite", "postgres"] | object = _UNSET
    crisis_log_database_url: str | None | object = _UNSET
    session_feedback_persistence_backend: Literal["sqlite", "postgres"] | object = (
        _UNSET
    )
    session_feedback_database_url: str | None | object = _UNSET
    text_session_backend: TextSessionBackend | object = _UNSET
    text_session_database_url: str | None | object = _UNSET
    allow_legacy_sqlite: bool | object = _UNSET

    @classmethod
    def for_shared_backend(
        cls,
        *,
        memory_mode: MemoryMode,
        persistence_backend: Literal["postgres"],
        database_url: str | None,
        text_session_backend: TextSessionBackend = "auto",
        text_session_database_url: str | None = None,
        allow_legacy_sqlite: bool = False,
    ) -> RuntimePersistenceConfig:
        """Build config when all durable stores share one backend/DSN."""

        if persistence_backend != "postgres":
            raise ValueError(
                f"Unsupported shared persistence backend: {persistence_backend}. "
                "Use 'postgres'."
            )

        resolved_text_session_database_url = text_session_database_url or database_url
        return cls(
            memory_mode=memory_mode,
            memory_backend=persistence_backend,
            memory_database_url=database_url,
            thread_persistence_backend=persistence_backend,
            thread_database_url=database_url,
            crisis_log_persistence_backend=persistence_backend,
            crisis_log_database_url=database_url,
            session_feedback_persistence_backend=persistence_backend,
            session_feedback_database_url=database_url,
            text_session_backend=text_session_backend,
            text_session_database_url=resolved_text_session_database_url,
            allow_legacy_sqlite=allow_legacy_sqlite,
        )


def _is_in_memory_sqlite_path(path: str | Path | None) -> bool:
    """Return whether a SQLite path is explicitly in-memory."""

    return path is not None and str(path) == ":memory:"


def _validate_legacy_sqlite_durable_allowed(
    *,
    memory_mode: MemoryMode,
    memory_backend: Literal["sqlite", "postgres"],
    thread_persistence_backend: Literal["memory", "sqlite", "postgres"],
    sqlite_path: str | Path,
    crisis_log_persistence_backend: Literal["sqlite", "postgres"],
    crisis_log_backend: CrisisLogBackend | None,
    session_feedback_persistence_backend: Literal["sqlite", "postgres"],
    session_feedback_backend: SessionFeedbackBackend | None,
    text_session_backend: TextSessionBackend,
    text_session_database_url: str | None,
    text_session_sqlite_path: str | Path | None,
    allow_legacy_sqlite: bool,
) -> None:
    """Reject removed stores and unapproved remaining durable SQLite."""

    if memory_backend == "sqlite":
        raise ValueError(_REMOVED_MEMORY_SQLITE_MESSAGE)

    if memory_mode != MemoryMode.INCOGNITO and thread_persistence_backend == "sqlite":
        raise ValueError(_REMOVED_THREAD_SQLITE_MESSAGE)

    removed_backends: list[str] = []
    if (
        memory_mode != MemoryMode.INCOGNITO
        and crisis_log_backend is None
        and crisis_log_persistence_backend == "sqlite"
    ):
        removed_backends.append("crisis_log_persistence_backend")
    if (
        memory_mode != MemoryMode.INCOGNITO
        and session_feedback_backend is None
        and session_feedback_persistence_backend == "sqlite"
    ):
        removed_backends.append("session_feedback_persistence_backend")
    if removed_backends:
        raise ValueError(
            f"{_REMOVED_AUDIT_FEEDBACK_SQLITE_MESSAGE} Removed fields: "
            f"{', '.join(removed_backends)}."
        )

    if memory_mode == MemoryMode.INCOGNITO or allow_legacy_sqlite:
        return

    sqlite_backends: list[str] = []
    text_session_uses_sqlite = text_session_backend == "sqlite" or (
        text_session_backend == "auto" and not text_session_database_url
    )
    text_session_uses_disk = (
        not _is_in_memory_sqlite_path(sqlite_path)
        if text_session_sqlite_path is None
        else not _is_in_memory_sqlite_path(text_session_sqlite_path)
    )
    if text_session_uses_sqlite and text_session_uses_disk:
        sqlite_backends.append("text_session_backend")

    if sqlite_backends:
        raise ValueError(
            f"{_LEGACY_SQLITE_DURABLE_MESSAGE} SQLite fields: "
            f"{', '.join(sqlite_backends)}."
        )


@dataclass(slots=True)
class _ResolvedRuntimePersistenceConfig:
    """Constructor-ready backend and database URL settings."""

    memory_mode: MemoryMode
    memory_backend: Literal["postgres"]
    memory_database_url: str | None
    thread_persistence_backend: Literal["memory", "sqlite", "postgres"]
    thread_database_url: str | None
    crisis_log_persistence_backend: Literal["sqlite", "postgres"]
    crisis_log_database_url: str | None
    session_feedback_persistence_backend: Literal["sqlite", "postgres"]
    session_feedback_database_url: str | None
    text_session_backend: TextSessionBackend
    text_session_database_url: str | None


def _resolve_runtime_persistence_config(
    *,
    persistence_config: RuntimePersistenceConfig | None,
    memory_mode: MemoryMode,
    memory_backend: Literal["sqlite", "postgres"],
    memory_database_url: str | None,
    thread_persistence_backend: Literal["memory", "sqlite", "postgres"],
    thread_database_url: str | None,
    sqlite_path: str | Path,
    sqlite_path_configured: bool,
    crisis_log_persistence_backend: Literal["sqlite", "postgres"],
    crisis_log_database_url: str | None,
    crisis_log_backend: CrisisLogBackend | None,
    session_feedback_persistence_backend: Literal["sqlite", "postgres"],
    session_feedback_database_url: str | None,
    session_feedback_backend: SessionFeedbackBackend | None,
    text_session_backend: TextSessionBackend,
    text_session_database_url: str | None,
    text_session_sqlite_path: str | Path | None,
    text_session_sqlite_path_configured: bool,
) -> _ResolvedRuntimePersistenceConfig:
    """Resolve legacy and grouped backend/database-url runtime settings."""

    allow_legacy_sqlite = False

    if persistence_config is not None:
        if persistence_config.memory_mode is not _UNSET:
            memory_mode = cast(MemoryMode, persistence_config.memory_mode)
        if persistence_config.memory_backend is not _UNSET:
            memory_backend = cast(
                Literal["sqlite", "postgres"],
                persistence_config.memory_backend,
            )
        if persistence_config.memory_database_url is not _UNSET:
            memory_database_url = cast(
                str | None,
                persistence_config.memory_database_url,
            )
        if persistence_config.thread_persistence_backend is not _UNSET:
            thread_persistence_backend = cast(
                Literal["memory", "sqlite", "postgres"],
                persistence_config.thread_persistence_backend,
            )
        if persistence_config.thread_database_url is not _UNSET:
            thread_database_url = cast(
                str | None,
                persistence_config.thread_database_url,
            )
        if persistence_config.crisis_log_persistence_backend is not _UNSET:
            crisis_log_persistence_backend = cast(
                Literal["sqlite", "postgres"],
                persistence_config.crisis_log_persistence_backend,
            )
        if persistence_config.crisis_log_database_url is not _UNSET:
            crisis_log_database_url = cast(
                str | None,
                persistence_config.crisis_log_database_url,
            )
        if persistence_config.session_feedback_persistence_backend is not _UNSET:
            session_feedback_persistence_backend = cast(
                Literal["sqlite", "postgres"],
                persistence_config.session_feedback_persistence_backend,
            )
        if persistence_config.session_feedback_database_url is not _UNSET:
            session_feedback_database_url = cast(
                str | None,
                persistence_config.session_feedback_database_url,
            )
        if persistence_config.text_session_backend is not _UNSET:
            text_session_backend = cast(
                TextSessionBackend,
                persistence_config.text_session_backend,
            )
        if persistence_config.text_session_database_url is not _UNSET:
            text_session_database_url = cast(
                str | None,
                persistence_config.text_session_database_url,
            )
        if persistence_config.allow_legacy_sqlite is not _UNSET:
            allow_legacy_sqlite = cast(bool, persistence_config.allow_legacy_sqlite)

    if thread_persistence_backend == "sqlite" and _is_in_memory_sqlite_path(
        sqlite_path
    ):
        thread_persistence_backend = "memory"

    _validate_legacy_sqlite_durable_allowed(
        memory_mode=memory_mode,
        memory_backend=memory_backend,
        thread_persistence_backend=thread_persistence_backend,
        sqlite_path=sqlite_path,
        crisis_log_persistence_backend=crisis_log_persistence_backend,
        crisis_log_backend=crisis_log_backend,
        session_feedback_persistence_backend=session_feedback_persistence_backend,
        session_feedback_backend=session_feedback_backend,
        text_session_backend=text_session_backend,
        text_session_database_url=text_session_database_url,
        text_session_sqlite_path=text_session_sqlite_path,
        allow_legacy_sqlite=allow_legacy_sqlite,
    )

    return _ResolvedRuntimePersistenceConfig(
        memory_mode=memory_mode,
        memory_backend=cast(Literal["postgres"], memory_backend),
        memory_database_url=memory_database_url,
        thread_persistence_backend=thread_persistence_backend,
        thread_database_url=thread_database_url,
        crisis_log_persistence_backend=crisis_log_persistence_backend,
        crisis_log_database_url=crisis_log_database_url,
        session_feedback_persistence_backend=session_feedback_persistence_backend,
        session_feedback_database_url=session_feedback_database_url,
        text_session_backend=text_session_backend,
        text_session_database_url=text_session_database_url,
    )


@dataclass(slots=True)
class RuntimeDependencies:
    """Grouped dependency injection hooks for runtime construction."""

    memory_store: MemoryStore | None | object = _UNSET
    crisis_log_backend: CrisisLogBackend | None | object = _UNSET
    session_feedback_backend: SessionFeedbackBackend | None | object = _UNSET
    embedding_provider: EmbeddingProvider | None | object = _UNSET
    default_llm_client: BaseLLMClient | None | object = _UNSET
    auto_finalize_excluded: Callable[[str], bool] | None | object = _UNSET


@dataclass(slots=True)
class _ResolvedRuntimeDependencies:
    """Constructor-ready runtime dependency overrides."""

    memory_store: MemoryStore | None
    crisis_log_backend: CrisisLogBackend | None
    session_feedback_backend: SessionFeedbackBackend | None
    embedding_provider: EmbeddingProvider | None
    default_llm_client: BaseLLMClient | None
    auto_finalize_excluded: Callable[[str], bool] | None


def _resolve_runtime_dependencies(
    *,
    dependencies: RuntimeDependencies | None,
    memory_store: MemoryStore | None,
    crisis_log_backend: CrisisLogBackend | None,
    session_feedback_backend: SessionFeedbackBackend | None,
    embedding_provider: EmbeddingProvider | None,
    default_llm_client: BaseLLMClient | None,
    auto_finalize_excluded: Callable[[str], bool] | None,
) -> _ResolvedRuntimeDependencies:
    """Resolve legacy and grouped runtime dependency overrides."""

    if dependencies is not None:
        if dependencies.memory_store is not _UNSET:
            memory_store = cast(MemoryStore | None, dependencies.memory_store)
        if dependencies.crisis_log_backend is not _UNSET:
            crisis_log_backend = cast(
                CrisisLogBackend | None,
                dependencies.crisis_log_backend,
            )
        if dependencies.session_feedback_backend is not _UNSET:
            session_feedback_backend = cast(
                SessionFeedbackBackend | None,
                dependencies.session_feedback_backend,
            )
        if dependencies.embedding_provider is not _UNSET:
            embedding_provider = cast(
                EmbeddingProvider | None,
                dependencies.embedding_provider,
            )
        if dependencies.default_llm_client is not _UNSET:
            default_llm_client = cast(
                BaseLLMClient | None,
                dependencies.default_llm_client,
            )
        if dependencies.auto_finalize_excluded is not _UNSET:
            auto_finalize_excluded = cast(
                Callable[[str], bool] | None,
                dependencies.auto_finalize_excluded,
            )

    return _ResolvedRuntimeDependencies(
        memory_store=memory_store,
        crisis_log_backend=crisis_log_backend,
        session_feedback_backend=session_feedback_backend,
        embedding_provider=embedding_provider,
        default_llm_client=default_llm_client,
        auto_finalize_excluded=auto_finalize_excluded,
    )


@dataclass(slots=True)
class RuntimeBehaviorConfig:
    """Grouped operational behavior settings for the runtime."""

    text_session_create_tables: bool | object = _UNSET
    text_session_history_limit: int | None | object = _UNSET
    session_timeout: timedelta | object = _UNSET
    session_sweep_interval_seconds: float | object = _UNSET
    finalize_active_sessions_on_close: bool | object = _UNSET
    speculative_memory_prefetch: bool | object = _UNSET


@dataclass(slots=True)
class _ResolvedRuntimeBehaviorConfig:
    """Constructor-ready runtime behavior settings."""

    text_session_create_tables: bool
    text_session_history_limit: int | None
    session_timeout: timedelta
    session_sweep_interval_seconds: float
    finalize_active_sessions_on_close: bool
    speculative_memory_prefetch: bool


def _resolve_runtime_behavior_config(
    *,
    behavior_config: RuntimeBehaviorConfig | None,
    text_session_create_tables: bool,
    text_session_history_limit: int | None,
    session_timeout: timedelta,
    session_sweep_interval_seconds: float,
    finalize_active_sessions_on_close: bool,
    speculative_memory_prefetch: bool,
) -> _ResolvedRuntimeBehaviorConfig:
    """Resolve legacy and grouped runtime behavior settings."""

    if behavior_config is not None:
        if behavior_config.text_session_create_tables is not _UNSET:
            text_session_create_tables = cast(
                bool,
                behavior_config.text_session_create_tables,
            )
        if behavior_config.text_session_history_limit is not _UNSET:
            text_session_history_limit = cast(
                int | None,
                behavior_config.text_session_history_limit,
            )
        if behavior_config.session_timeout is not _UNSET:
            session_timeout = cast(timedelta, behavior_config.session_timeout)
        if behavior_config.session_sweep_interval_seconds is not _UNSET:
            session_sweep_interval_seconds = cast(
                float,
                behavior_config.session_sweep_interval_seconds,
            )
        if behavior_config.finalize_active_sessions_on_close is not _UNSET:
            finalize_active_sessions_on_close = cast(
                bool,
                behavior_config.finalize_active_sessions_on_close,
            )
        if behavior_config.speculative_memory_prefetch is not _UNSET:
            speculative_memory_prefetch = cast(
                bool,
                behavior_config.speculative_memory_prefetch,
            )

    return _ResolvedRuntimeBehaviorConfig(
        text_session_create_tables=text_session_create_tables,
        text_session_history_limit=text_session_history_limit,
        session_timeout=session_timeout,
        session_sweep_interval_seconds=session_sweep_interval_seconds,
        finalize_active_sessions_on_close=finalize_active_sessions_on_close,
        speculative_memory_prefetch=speculative_memory_prefetch,
    )
