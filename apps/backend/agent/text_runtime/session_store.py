"""OpenAI Agents SDK session storage helpers for the text runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agents.memory import SQLiteSession, SessionSettings

from agent.memory.modes import MemoryMode
from agent.models import Message, MessageRole

TextSessionBackend = Literal["disabled", "sqlite", "sqlalchemy"]


@dataclass(frozen=True, slots=True)
class TextSessionStoreConfig:
    """Configuration for OpenAI Agents SDK conversation sessions."""

    backend: TextSessionBackend = "disabled"
    sqlite_path: str | Path = ":memory:"
    database_url: str | None = None
    create_tables: bool = True
    history_limit: int | None = None


class TextSessionStore:
    """Factory/cache for SDK sessions keyed by OpenCouch thread id."""

    def __init__(self, config: TextSessionStoreConfig) -> None:
        if config.backend == "disabled":
            raise ValueError("TextSessionStore cannot be constructed as disabled.")
        if config.backend == "sqlalchemy" and not config.database_url:
            raise ValueError(
                "text_session_backend='sqlalchemy' requires text_session_database_url."
            )

        self._config = config
        self._sessions: dict[str, Any] = {}
        self._engine: Any | None = None

        if config.backend == "sqlalchemy":
            from sqlalchemy.ext.asyncio import create_async_engine

            assert config.database_url is not None
            self._engine = create_async_engine(
                normalize_sqlalchemy_async_url(config.database_url)
            )

    @property
    def backend(self) -> TextSessionBackend:
        """Return the configured SDK session backend."""

        return self._config.backend

    def session_for_thread(self, thread_id: str) -> Any:
        """Return the SDK session object for a thread."""

        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty.")

        existing = self._sessions.get(normalized_thread_id)
        if existing is not None:
            return existing

        settings = SessionSettings(limit=self._config.history_limit)
        if self._config.backend == "sqlite":
            session = SQLiteSession(
                normalized_thread_id,
                db_path=self._config.sqlite_path,
                session_settings=settings,
            )
        elif self._config.backend == "sqlalchemy":
            from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

            if self._engine is None:
                raise RuntimeError("SQLAlchemy text session engine is not initialized.")
            session = SQLAlchemySession(
                normalized_thread_id,
                engine=self._engine,
                create_tables=self._config.create_tables,
                session_settings=settings,
            )
        else:
            raise ValueError(
                f"Unsupported text session backend {self._config.backend!r}."
            )

        self._sessions[normalized_thread_id] = session
        return session

    async def get_history(
        self,
        thread_id: str,
        *,
        limit: int | None = None,
    ) -> list[Message]:
        """Materialize SDK session items as public transcript messages."""

        session = self.session_for_thread(thread_id)
        items = await session.get_items(limit=limit)
        return messages_from_sdk_session_items(items)

    async def clear_thread(self, thread_id: str) -> None:
        """Clear the SDK session for a thread."""

        session = self.session_for_thread(thread_id)
        await session.clear_session()

    async def aclose(self) -> None:
        """Close cached SDK session resources."""

        for session in self._sessions.values():
            close = getattr(session, "close", None)
            if close is not None:
                close()
        self._sessions.clear()

        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None


def create_text_session_store(
    *,
    memory_mode: MemoryMode,
    backend: TextSessionBackend = "disabled",
    sqlite_path: str | Path = ":memory:",
    database_url: str | None = None,
    create_tables: bool = True,
    history_limit: int | None = None,
) -> TextSessionStore | None:
    """Create an SDK session store, preserving incognito's no-disk contract."""

    if backend == "disabled":
        return None
    if memory_mode == MemoryMode.INCOGNITO:
        return TextSessionStore(
            TextSessionStoreConfig(
                backend="sqlite",
                sqlite_path=":memory:",
                create_tables=create_tables,
                history_limit=history_limit,
            )
        )
    return TextSessionStore(
        TextSessionStoreConfig(
            backend=backend,
            sqlite_path=sqlite_path,
            database_url=database_url,
            create_tables=create_tables,
            history_limit=history_limit,
        )
    )


def normalize_sqlalchemy_async_url(url: str) -> str:
    """Return an async-driver SQLAlchemy URL for SDK SQLAlchemy sessions."""

    normalized = url.strip()
    if normalized.startswith("postgresql+asyncpg://"):
        return normalized
    if normalized.startswith("postgresql://"):
        return normalized.replace("postgresql://", "postgresql+asyncpg://", 1)
    if normalized.startswith("postgres://"):
        return normalized.replace("postgres://", "postgresql+asyncpg://", 1)
    if normalized.startswith("sqlite+aiosqlite://"):
        return normalized
    if normalized.startswith("sqlite://"):
        return normalized.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return normalized


def messages_from_sdk_session_items(items: list[Any]) -> list[Message]:
    """Convert SDK session items into public conversation messages."""

    messages: list[Message] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "")
        if role not in {"system", "user", "assistant"}:
            continue
        content = _content_to_text(item.get("content")).strip()
        if not content:
            continue
        messages.append(
            Message(
                role=MessageRole(role),
                content=content,
                response_style=None,
            )
        )
    return messages


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, Mapping):
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part.strip() for part in parts if part.strip())
