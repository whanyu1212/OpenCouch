"""OpenAI Agents SDK session storage helpers for the text runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import isawaitable
import logging
from pathlib import Path
from typing import Any, Literal

from agents.memory import SQLiteSession, SessionSettings

from agent.memory.modes import MemoryMode
from agent.models import Message, MessageRole
from agent.runtime.session.history import (
    content_to_text,
    messages_from_sdk_session_items,
)

logger = logging.getLogger(__name__)

TextSessionBackend = Literal["auto", "disabled", "sqlite", "sqlalchemy"]
ActiveTextSessionBackend = Literal["sqlite", "sqlalchemy"]


@dataclass(frozen=True, slots=True)
class TextSessionStoreConfig:
    """Configuration for OpenAI Agents SDK conversation sessions."""

    backend: ActiveTextSessionBackend = "sqlite"
    sqlite_path: str | Path = ":memory:"
    database_url: str | None = None
    create_tables: bool = True
    history_limit: int | None = None


class TextSessionStore:
    """Factory/cache for SDK sessions keyed by OpenCouch thread id."""

    def __init__(self, config: TextSessionStoreConfig) -> None:
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
    def backend(self) -> ActiveTextSessionBackend:
        """Return the configured SDK session backend."""

        return self._config.backend

    def session_for_thread(self, thread_id: str) -> Any:
        """Return the SDK session object for a thread."""

        normalized_thread_id = self._normalize_thread_id(thread_id)

        existing = self._sessions.get(normalized_thread_id)
        if existing is not None:
            return existing

        session = self._create_session(normalized_thread_id)
        self._sessions[normalized_thread_id] = session
        return session

    def _create_session(self, normalized_thread_id: str) -> Any:
        """Construct one SDK session without adding it to the cache."""

        settings = SessionSettings(limit=self._config.history_limit)
        if self._config.backend == "sqlite":
            return SQLiteSession(
                normalized_thread_id,
                db_path=self._config.sqlite_path,
                session_settings=settings,
            )
        if self._config.backend == "sqlalchemy":
            from agents.extensions.memory.sqlalchemy_session import SQLAlchemySession

            if self._engine is None:
                raise RuntimeError("SQLAlchemy text session engine is not initialized.")
            return SQLAlchemySession(
                normalized_thread_id,
                engine=self._engine,
                create_tables=self._config.create_tables,
                session_settings=settings,
            )
        raise ValueError(f"Unsupported text session backend {self._config.backend!r}.")

    async def evict_thread(self, thread_id: str) -> None:
        """Evict one cached SDK session without deleting its history."""

        normalized_thread_id = self._normalize_thread_id(thread_id)
        session = self._sessions.pop(normalized_thread_id, None)
        if session is None:
            return

        await self._close_session(session)

    def turn_session_for_thread(
        self,
        thread_id: str,
        *,
        current_user_message: str,
    ) -> "TextSessionTurn":
        """Return a per-turn SDK session that stores raw conversation turns."""

        return TextSessionTurn(
            self.session_for_thread(thread_id),
            current_user_message=current_user_message,
        )

    async def seed_thread_from_messages(
        self,
        thread_id: str,
        messages: list[Message],
    ) -> bool:
        """Seed an empty SDK session from app transcript recovery messages."""

        session = self.session_for_thread(thread_id)
        if await session.get_items(limit=1):
            return False

        items = [
            {"role": message.role.value, "content": message.content}
            for message in messages
            if message.role in {MessageRole.USER, MessageRole.ASSISTANT}
            and message.content.strip()
        ]
        if not items:
            return False

        await session.add_items(items)
        return True

    async def get_history(
        self,
        thread_id: str,
        *,
        limit: int | None = None,
        cache: bool = True,
    ) -> list[Message]:
        """Materialize SDK session items as public transcript messages."""

        normalized_thread_id = self._normalize_thread_id(thread_id)
        session = self._sessions.get(normalized_thread_id)
        close_after_read = False
        if session is None:
            if cache:
                session = self.session_for_thread(normalized_thread_id)
            else:
                session = self._create_session(normalized_thread_id)
                close_after_read = True
        try:
            items = await session.get_items(limit=limit)
        finally:
            if close_after_read:
                try:
                    await self._close_session(session)
                except Exception:
                    logger.warning(
                        "TextSessionStore: transient history session close failed; "
                        "preserving the history outcome.",
                        exc_info=True,
                    )
        return messages_from_sdk_session_items(items)

    async def clear_thread(self, thread_id: str) -> None:
        """Clear and evict the SDK session for a thread."""

        session = self.session_for_thread(thread_id)
        try:
            await session.clear_session()
        finally:
            await self.evict_thread(thread_id)

    async def ensure_turn_recorded(
        self,
        thread_id: str,
        *,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Ensure one raw user/assistant turn exists in SDK history."""

        user_text = user_message.strip()
        assistant_text = assistant_message.strip()
        if not user_text and not assistant_text:
            return

        history = await self.get_history(thread_id)
        if _history_ends_with_turn(
            history,
            user_message=user_text,
            assistant_message=assistant_text,
        ):
            return

        items: list[dict[str, str]] = []
        if user_text and not _history_ends_with_message(
            history,
            role=MessageRole.USER,
            content=user_text,
        ):
            items.append({"role": "user", "content": user_text})
        if assistant_text:
            items.append({"role": "assistant", "content": assistant_text})
        if items:
            await self.session_for_thread(thread_id).add_items(items)

    async def aclose(self) -> None:
        """Close cached SDK session resources."""

        first_failure: BaseException | None = None
        for thread_id in tuple(self._sessions):
            try:
                await self.evict_thread(thread_id)
            except BaseException as exc:
                if first_failure is None:
                    first_failure = exc

        if self._engine is not None:
            try:
                await self._engine.dispose()
            except BaseException as exc:
                if first_failure is None:
                    first_failure = exc
            finally:
                self._engine = None

        if first_failure is not None:
            raise first_failure

    @staticmethod
    async def _close_session(session: Any) -> None:
        close = getattr(session, "close", None)
        if callable(close):
            close_result = close()
            if isawaitable(close_result):
                await close_result

    @staticmethod
    def _normalize_thread_id(thread_id: str) -> str:
        normalized_thread_id = thread_id.strip()
        if not normalized_thread_id:
            raise ValueError("thread_id must not be empty.")
        return normalized_thread_id


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
    resolved_backend = _resolve_backend(backend, database_url=database_url)
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
            backend=resolved_backend,
            sqlite_path=sqlite_path,
            database_url=database_url,
            create_tables=create_tables,
            history_limit=history_limit,
        )
    )


class TextSessionTurn:
    """Per-turn session facade that stores raw chat turns in the SDK store.

    The model receives runtime prompts as the current input, but public history
    and future SDK session history should contain only the user's raw message
    and assistant-facing reply. This facade delegates reads to the underlying
    SDK session and normalizes writes for one runtime turn.
    """

    def __init__(self, session: Any, *, current_user_message: str) -> None:
        self._session = session
        self._current_user_message = current_user_message.strip()
        self._stored_current_user = False
        self._stored_assistant_texts: set[str] = set()
        self.session_id = getattr(session, "session_id")
        self.session_settings = getattr(session, "session_settings", None)

    async def get_items(self, limit: int | None = None) -> list[Any]:
        """Return prior raw conversation history."""

        return await self._session.get_items(limit=limit)

    async def add_items(self, items: list[Any]) -> None:
        """Persist only raw user/assistant conversation messages."""

        normalized = self._conversation_items_for_storage(items)
        if normalized:
            await self._session.add_items(normalized)

    async def pop_item(self) -> Any | None:
        """Remove and return the latest item from the underlying session."""

        return await self._session.pop_item()

    async def clear_session(self) -> None:
        """Clear the underlying session."""

        await self._session.clear_session()

    def _conversation_items_for_storage(self, items: list[Any]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "")
            if role == "user":
                if self._stored_current_user:
                    continue
                content = (
                    self._current_user_message
                    or content_to_text(item.get("content")).strip()
                )
                if content:
                    normalized.append({"role": "user", "content": content})
                    self._stored_current_user = True
                continue
            if role != "assistant":
                continue

            content = content_to_text(item.get("content")).strip()
            if not content or content in self._stored_assistant_texts:
                continue
            normalized.append({"role": "assistant", "content": content})
            self._stored_assistant_texts.add(content)
        return normalized


def _resolve_backend(
    backend: TextSessionBackend,
    *,
    database_url: str | None,
) -> ActiveTextSessionBackend:
    if backend == "auto":
        return "sqlalchemy" if database_url else "sqlite"
    if backend in {"sqlite", "sqlalchemy"}:
        return backend
    raise ValueError(f"Unsupported active text session backend {backend!r}.")


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


def _history_ends_with_turn(
    history: list[Message],
    *,
    user_message: str,
    assistant_message: str,
) -> bool:
    if len(history) < 2:
        return False
    user, assistant = history[-2], history[-1]
    return (
        user.role == MessageRole.USER
        and user.content == user_message
        and assistant.role == MessageRole.ASSISTANT
        and assistant.content == assistant_message
    )


def _history_ends_with_message(
    history: list[Message],
    *,
    role: MessageRole,
    content: str,
) -> bool:
    if not history:
        return False
    latest = history[-1]
    return latest.role == role and latest.content == content
