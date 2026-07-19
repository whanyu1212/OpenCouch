"""Configuration models and validation for the persistent agent runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal

from agent.audit.crisis_log import CrisisLogBackend
from agent.feedback.session_feedback import SessionFeedbackBackend
from agent.memory.modes import MemoryMode
from agent.memory.providers.embeddings import EmbeddingProvider
from agent.memory.store import MemoryStore
from agent.runtime.session_store import TextSessionBackend
from llm.base import BaseLLMClient

BACKEND_ROOT = Path(__file__).resolve().parents[2]
_STORE_DIR = BACKEND_ROOT / ".store"
DEFAULT_TEXT_SESSION_DB_PATH = _STORE_DIR / "text_sessions.sqlite3"
SESSION_TIMEOUT = timedelta(minutes=20)
_LEGACY_SQLITE_DURABLE_MESSAGE = (
    "Durable SQLite SDK-session persistence is legacy and disabled unless "
    "explicitly allowed. Use Postgres for durable runtime persistence, "
    "or set allow_legacy_sqlite=True for SDK text-session compatibility."
)


@dataclass(slots=True)
class RuntimeStoragePaths:
    """Filesystem paths for runtime storage that remains SQLite-backed."""

    text_session_sqlite_path: str | Path | None = None


@dataclass(slots=True)
class RuntimePersistenceConfig:
    """Backend and database URL configuration for the runtime."""

    memory_mode: MemoryMode = MemoryMode.LOCAL
    memory_backend: Literal["postgres"] = "postgres"
    memory_database_url: str | None = None
    thread_persistence_backend: Literal["memory", "postgres"] = "postgres"
    thread_database_url: str | None = None
    crisis_log_persistence_backend: Literal["postgres"] = "postgres"
    crisis_log_database_url: str | None = None
    session_feedback_persistence_backend: Literal["postgres"] = "postgres"
    session_feedback_database_url: str | None = None
    text_session_backend: TextSessionBackend = "auto"
    text_session_database_url: str | None = None
    allow_legacy_sqlite: bool = False

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
        """Build config when all durable stores share one backend and DSN."""

        if persistence_backend != "postgres":
            raise ValueError(
                f"Unsupported shared persistence backend: {persistence_backend}. "
                "Use 'postgres'."
            )

        return cls(
            memory_mode=memory_mode,
            memory_backend="postgres",
            memory_database_url=database_url,
            thread_persistence_backend="postgres",
            thread_database_url=database_url,
            crisis_log_persistence_backend="postgres",
            crisis_log_database_url=database_url,
            session_feedback_persistence_backend="postgres",
            session_feedback_database_url=database_url,
            text_session_backend=text_session_backend,
            text_session_database_url=text_session_database_url or database_url,
            allow_legacy_sqlite=allow_legacy_sqlite,
        )


@dataclass(slots=True)
class RuntimeDependencies:
    """Dependency injection hooks for runtime construction."""

    memory_store: MemoryStore | None = None
    crisis_log_backend: CrisisLogBackend | None = None
    session_feedback_backend: SessionFeedbackBackend | None = None
    embedding_provider: EmbeddingProvider | None = None
    default_llm_client: BaseLLMClient | None = None
    auto_finalize_excluded: Callable[[str], bool] | None = None


@dataclass(slots=True)
class RuntimeBehaviorConfig:
    """Operational behavior settings for the runtime."""

    text_session_create_tables: bool = True
    text_session_history_limit: int | None = None
    session_timeout: timedelta = SESSION_TIMEOUT
    session_sweep_interval_seconds: float = 30.0
    finalize_active_sessions_on_close: bool = True
    speculative_memory_prefetch: bool = True


def validate_runtime_configuration(
    *,
    persistence: RuntimePersistenceConfig,
    storage_paths: RuntimeStoragePaths,
) -> None:
    """Validate the grouped runtime contract before resources are created."""

    if persistence.memory_backend != "postgres":
        raise ValueError(
            "SQLite memory persistence has been removed. Use memory_backend='postgres'."
        )
    if persistence.thread_persistence_backend not in {"memory", "postgres"}:
        raise ValueError(
            "SQLite runtime-state and active-session persistence has been removed. "
            "Use thread_persistence_backend='postgres' or 'memory'."
        )
    if persistence.crisis_log_persistence_backend != "postgres":
        raise ValueError(
            "SQLite crisis-audit persistence has been removed. Use Postgres."
        )
    if persistence.session_feedback_persistence_backend != "postgres":
        raise ValueError(
            "SQLite session-feedback persistence has been removed. Use Postgres."
        )

    if (
        persistence.memory_mode == MemoryMode.INCOGNITO
        or persistence.allow_legacy_sqlite
    ):
        return

    text_session_uses_path_sqlite = persistence.text_session_backend == "sqlite" or (
        persistence.text_session_backend == "auto"
        and not persistence.text_session_database_url
    )
    text_session_path = storage_paths.text_session_sqlite_path
    path_sqlite_uses_disk = (
        text_session_path is None or str(text_session_path) != ":memory:"
    )
    database_url = persistence.text_session_database_url or ""
    sqlalchemy_sqlite_uses_disk = persistence.text_session_backend in {
        "auto",
        "sqlalchemy",
    } and _sqlalchemy_sqlite_url_uses_disk(database_url)
    if (
        text_session_uses_path_sqlite and path_sqlite_uses_disk
    ) or sqlalchemy_sqlite_uses_disk:
        raise ValueError(
            f"{_LEGACY_SQLITE_DURABLE_MESSAGE} SQLite fields: text_session_backend."
        )


def _sqlalchemy_sqlite_url_uses_disk(database_url: str) -> bool:
    """Return whether a SQLAlchemy SQLite URL resolves to a disk database."""

    normalized_url = database_url.strip()
    if not normalized_url.startswith("sqlite"):
        return False

    from sqlalchemy.engine import make_url

    url = make_url(normalized_url)
    database = url.database
    if not database or database == ":memory:":
        return False

    query = {str(key): str(value) for key, value in url.query.items()}
    uri_enabled = query.get("uri", "").lower() in {"1", "true", "yes", "on"}
    uri_memory_database = query.get("mode") == "memory" or database in {
        "file::memory:",
        "file:memory",
    }
    return not (uri_enabled and uri_memory_database)
