"""Tests for the standalone Telegram channel gateway."""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from uuid import uuid4
from types import SimpleNamespace
from typing import Any

import pytest

from agent.memory.modes import MemoryMode
from agent.models import (
    AgentOutput,
    Channel,
    CrisisAssessment,
    DoneEvent,
    ResponseCategory,
    ResponseReadyEvent,
)
from channels.gateway import (
    GatewayLockError,
    TelegramGatewayLock,
    resolve_telegram_memory_mode,
    run_telegram_gateway,
    telegram_gateway_lock_path,
)
from channels.registry.sqlite_fallback import SqliteTelegramSessionRegistry
from channels.telegram import (
    TELEGRAM_MAINTENANCE_MESSAGE,
    TELEGRAM_SESSION_CLOSED_MESSAGE,
    TELEGRAM_START_MESSAGE,
    TelegramChannel,
    TelegramConfigurationError,
    TelegramGatewayConfig,
    TelegramSessionRegistry,
    build_telegram_session_registry,
    render_telegram_markdown,
    split_telegram_markdown_html,
    split_telegram_text,
    telegram_thread_id,
)
from channels.registry.postgres import PostgresTelegramSessionRegistry
from agent.persistence import PersistentAgentRuntime, SessionLeaseExpired, SessionStatus
from llm.base import BaseLLMClient, StructuredResponseT

_POSTGRES_TEST_URL_ENV = "OPENCOUCH_TEST_POSTGRES_URL"


def _postgres_database_url() -> str | None:
    """Return the opt-in Postgres DSN for Telegram registry integration tests.

    Returns:
        str | None: Configured Postgres DSN, or ``None`` when unavailable.
    """

    return os.getenv(_POSTGRES_TEST_URL_ENV) or os.getenv(
        "OPENCOUCH_MEMORY_DATABASE_URL"
    )


class _FakeLLM(BaseLLMClient):
    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return "fake"

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield "fake"

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
    ) -> StructuredResponseT:
        raise AssertionError("structured generation should not be used in this test")


class _FakeMessage:
    def __init__(self, text: str | None = "hello") -> None:
        self.text = text
        self.replies: list[str] = []
        self.reply_kwargs: list[dict[str, Any]] = []

    async def reply_text(self, text: str, **kwargs: Any) -> None:
        self.replies.append(text)
        self.reply_kwargs.append(kwargs)


class _FakeRuntime:
    def __init__(self, events: list[Any] | None = None) -> None:
        self.events = events or []
        self.turn_calls: list[dict[str, Any]] = []
        self.end_session_calls: list[tuple[str, BaseLLMClient | None]] = []
        self.reset_thread_calls: list[str] = []
        self.session_statuses: dict[str, SessionStatus] = {}
        self.rotate_after_turn = False
        self.turn_errors: list[Exception] = []
        self.end_session_result: object | None = None
        self.end_session_error: Exception | None = None
        self.reset_thread_error: Exception | None = None

    async def run_turn_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.turn_calls.append(kwargs)
        thread_id = str(kwargs["thread_id"])
        if self.turn_errors:
            raise self.turn_errors.pop(0)
        self.session_statuses[thread_id] = SessionStatus.ACTIVE
        for event in self.events:
            yield event
        if self.rotate_after_turn:
            self.session_statuses[thread_id] = SessionStatus.ROTATION_REQUIRED

    async def end_session(
        self,
        thread_id: str,
        *,
        llm_client: BaseLLMClient | None = None,
    ) -> object | None:
        self.end_session_calls.append((thread_id, llm_client))
        if self.end_session_error is not None:
            raise self.end_session_error
        self.session_statuses[thread_id] = SessionStatus.ABSENT
        return self.end_session_result

    async def session_status(self, thread_id: str) -> SessionStatus:
        return self.session_statuses.get(thread_id, SessionStatus.ABSENT)

    async def reset_thread(self, thread_id: str) -> None:
        self.reset_thread_calls.append(thread_id)
        if self.reset_thread_error is not None:
            raise self.reset_thread_error


def _output(text: str) -> AgentOutput:
    return AgentOutput(
        response_text=text,
        response_type=ResponseCategory.THERAPEUTIC,
        crisis=CrisisAssessment(),
    )


def _update(
    *,
    text: str | None = "hello",
    chat_id: int = 123,
    sender_id: int = 456,
    chat_type: str = "private",
) -> SimpleNamespace:
    message = _FakeMessage(text)
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=sender_id),
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_message=message,
        message=message,
    )


def _config(
    *,
    allowed_user_ids: frozenset[int] = frozenset({456}),
    thread_rotation_enabled: bool = False,
    session_registry_sqlite_path: Path | None = None,
    session_transcript_soft_limit: int = 240,
    reclaim_grace_seconds: float = 3600.0,
) -> TelegramGatewayConfig:
    return TelegramGatewayConfig(
        bot_token="token",
        allowed_user_ids=allowed_user_ids,
        owner_id="hanyu",
        thread_rotation_enabled=thread_rotation_enabled,
        session_registry_sqlite_path=(session_registry_sqlite_path or Path(":memory:")),
        session_transcript_soft_limit=session_transcript_soft_limit,
        reclaim_grace_seconds=reclaim_grace_seconds,
    )


def _registry_for_config(
    config: TelegramGatewayConfig,
) -> TelegramSessionRegistry | None:
    """Return the default test registry for a Telegram config.

    Args:
        config (TelegramGatewayConfig): Telegram channel config.

    Returns:
        TelegramSessionRegistry | None: Postgres registry when configured, otherwise
            ``None`` so production fallback behavior is still covered.
    """

    database_url = _postgres_database_url()
    if config.thread_rotation_enabled and database_url is not None:
        return PostgresTelegramSessionRegistry(database_url)
    return None


def _channel(
    *,
    runtime: _FakeRuntime | None = None,
    config: TelegramGatewayConfig | None = None,
    llm: _FakeLLM | None = None,
    response_llm: _FakeLLM | None = None,
    session_registry: TelegramSessionRegistry | None = None,
) -> tuple[TelegramChannel, _FakeRuntime, _FakeLLM, _FakeLLM]:
    fake_runtime = runtime or _FakeRuntime()
    fake_llm = llm or _FakeLLM()
    fake_response_llm = response_llm or _FakeLLM()
    resolved_config = config or _config()
    channel = TelegramChannel(
        config=resolved_config,
        runtime=fake_runtime,  # type: ignore[arg-type]
        llm_client=fake_llm,
        response_llm_client=fake_response_llm,
        session_registry=session_registry or _registry_for_config(resolved_config),
    )
    return channel, fake_runtime, fake_llm, fake_response_llm


def test_config_from_env_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCOUCH_TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "456")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_OWNER_ID", "hanyu")

    with pytest.raises(TelegramConfigurationError, match="BOT_TOKEN"):
        TelegramGatewayConfig.from_env()


def test_config_from_env_requires_owner_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "456")
    monkeypatch.delenv("OPENCOUCH_TELEGRAM_OWNER_ID", raising=False)

    with pytest.raises(TelegramConfigurationError, match="OWNER_ID"):
        TelegramGatewayConfig.from_env()


def test_config_from_env_requires_non_empty_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_OWNER_ID", "hanyu")

    with pytest.raises(TelegramConfigurationError, match="ALLOW_FROM"):
        TelegramGatewayConfig.from_env()


def test_config_from_env_parses_allowlist_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "456, 789")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_OWNER_ID", "hanyu")
    monkeypatch.delenv("OPENCOUCH_TELEGRAM_DROP_PENDING_UPDATES", raising=False)
    monkeypatch.delenv("OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER", raising=False)
    monkeypatch.delenv("OPENCOUCH_TELEGRAM_THREAD_ROTATION_ENABLED", raising=False)
    monkeypatch.delenv("OPENCOUCH_TELEGRAM_SESSION_DB_PATH", raising=False)
    monkeypatch.delenv(
        "OPENCOUCH_TELEGRAM_SESSION_TRANSCRIPT_SOFT_LIMIT",
        raising=False,
    )
    monkeypatch.delenv(
        "OPENCOUCH_TELEGRAM_ROTATION_SWEEP_INTERVAL_SECONDS",
        raising=False,
    )
    monkeypatch.delenv("OPENCOUCH_TELEGRAM_RECLAIM_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("OPENCOUCH_TELEGRAM_RECLAIM_GRACE_SECONDS", raising=False)

    config = TelegramGatewayConfig.from_env()

    assert config.allowed_user_ids == frozenset({456, 789})
    assert config.drop_pending_updates is True
    assert config.response_model_tier == "fast"
    assert config.thread_rotation_enabled is False


def test_config_from_env_parses_thread_rotation_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "telegram_sessions.sqlite3"
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "456")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_OWNER_ID", "hanyu")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_THREAD_ROTATION_ENABLED", "true")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_SESSION_DB_PATH", str(registry_path))
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_SESSION_TRANSCRIPT_SOFT_LIMIT", "12")

    config = TelegramGatewayConfig.from_env()

    assert config.thread_rotation_enabled is True
    assert config.session_registry_sqlite_path == registry_path
    assert config.session_transcript_soft_limit == 12


def test_config_from_env_rejects_non_numeric_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "456,hanyu")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_OWNER_ID", "hanyu")

    with pytest.raises(TelegramConfigurationError, match="numeric"):
        TelegramGatewayConfig.from_env()


def test_config_from_env_parses_drop_pending_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "456")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_OWNER_ID", "hanyu")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_DROP_PENDING_UPDATES", "false")

    assert TelegramGatewayConfig.from_env().drop_pending_updates is False


def test_config_from_env_parses_response_model_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_ALLOW_FROM", "456")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_OWNER_ID", "hanyu")
    monkeypatch.setenv("OPENCOUCH_TELEGRAM_RESPONSE_MODEL_TIER", "quality")

    assert TelegramGatewayConfig.from_env().response_model_tier == "quality"


def test_telegram_thread_id_uses_dm_prefix() -> None:
    assert telegram_thread_id(123) == "telegram:dm:123"


@pytest.mark.asyncio
async def test_start_and_help_are_static_without_runtime_calls() -> None:
    channel, runtime, _, _ = _channel()
    start_update = _update()
    help_update = _update()

    await channel.handle_start(start_update, None)  # type: ignore[arg-type]
    await channel.handle_help(help_update, None)  # type: ignore[arg-type]

    assert start_update.effective_message.replies == [TELEGRAM_START_MESSAGE]
    assert help_update.effective_message.replies == [TELEGRAM_START_MESSAGE]
    assert runtime.turn_calls == []
    assert runtime.end_session_calls == []


@pytest.mark.asyncio
async def test_denied_sender_logs_numeric_id_and_skips_runtime(
    caplog: pytest.LogCaptureFixture,
) -> None:
    channel, runtime, _, _ = _channel(config=_config(allowed_user_ids=frozenset({111})))
    update = _update(sender_id=456)

    with caplog.at_level(logging.WARNING):
        await channel.handle_text(update, None)  # type: ignore[arg-type]

    assert "sender_id=456" in caplog.text
    assert runtime.turn_calls == []
    assert update.effective_message.replies == [
        "This Telegram account is not allowed to use this OpenCouch bot."
    ]


@pytest.mark.asyncio
async def test_group_message_is_rejected_without_runtime_call() -> None:
    channel, runtime, _, _ = _channel()
    update = _update(chat_type="group")

    await channel.handle_text(update, None)  # type: ignore[arg-type]

    assert runtime.turn_calls == []
    assert update.effective_message.replies == [
        "OpenCouch Telegram MVP only supports direct messages."
    ]


@pytest.mark.asyncio
async def test_end_calls_runtime_end_session_with_thread_and_llm() -> None:
    channel, runtime, llm, _ = _channel()
    update = _update(chat_id=321)

    await channel.handle_end(update, None)  # type: ignore[arg-type]

    assert runtime.end_session_calls == [("telegram:dm:321", llm)]
    assert update.effective_message.replies == [TELEGRAM_SESSION_CLOSED_MESSAGE]


@pytest.mark.asyncio
async def test_end_after_timeout_still_returns_confirmation() -> None:
    channel, runtime, _, _ = _channel()
    runtime.end_session_result = None
    update = _update()

    await channel.handle_end(update, None)  # type: ignore[arg-type]

    assert update.effective_message.replies == [TELEGRAM_SESSION_CLOSED_MESSAGE]


@pytest.mark.asyncio
async def test_normal_message_calls_runtime_with_telegram_metadata() -> None:
    runtime = _FakeRuntime(
        events=[
            ResponseReadyEvent(output=_output("ready")),
            DoneEvent(output=_output("done")),
        ]
    )
    channel, _, llm, response_llm = _channel(runtime=runtime)
    update = _update(text="hi", chat_id=321)

    await channel.handle_text(update, None)  # type: ignore[arg-type]

    assert update.effective_message.replies == ["ready"]
    assert len(runtime.turn_calls) == 1
    assert runtime.turn_calls[0] == {
        "thread_id": "telegram:dm:321",
        "message": "hi",
        "channel": Channel.TELEGRAM,
        "user_id": "hanyu",
        "installed_skills": [],
        "llm_client": llm,
        "response_llm_client": response_llm,
    }


@pytest.mark.asyncio
async def test_rotation_creates_and_reuses_session_thread(tmp_path: Path) -> None:
    runtime = _FakeRuntime(
        events=[
            ResponseReadyEvent(output=_output("ready")),
            DoneEvent(output=_output("done")),
        ]
    )
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
        session_transcript_soft_limit=12,
    )
    channel, _, llm, response_llm = _channel(runtime=runtime, config=config)

    try:
        first_update = _update(text="first", chat_id=321)
        second_update = _update(text="second", chat_id=321)

        await channel.handle_text(first_update, None)  # type: ignore[arg-type]
        await channel.handle_text(second_update, None)  # type: ignore[arg-type]

        assert first_update.effective_message.replies == ["ready"]
        assert second_update.effective_message.replies == ["ready"]
        assert len(runtime.turn_calls) == 2
        first_thread_id = runtime.turn_calls[0]["thread_id"]
        assert first_thread_id.startswith("telegram:dm:321:session:")
        assert runtime.turn_calls[0] == {
            "thread_id": first_thread_id,
            "message": "first",
            "channel": Channel.TELEGRAM,
            "user_id": "hanyu",
            "installed_skills": [],
            "llm_client": llm,
            "response_llm_client": response_llm,
            "expected_liveness": "absent",
            "session_transcript_soft_limit": 12,
        }
        assert runtime.turn_calls[1]["thread_id"] == first_thread_id
        assert runtime.turn_calls[1]["expected_liveness"] == "active"
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_rotation_end_closes_active_pointer_and_next_message_rotates(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(events=[ResponseReadyEvent(output=_output("ready"))])
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
    )
    channel, _, llm, _ = _channel(runtime=runtime, config=config)

    try:
        first_update = _update(text="first", chat_id=321)
        end_update = _update(text="/end", chat_id=321)
        second_update = _update(text="second", chat_id=321)

        await channel.handle_text(first_update, None)  # type: ignore[arg-type]
        first_thread_id = runtime.turn_calls[0]["thread_id"]
        await channel.handle_end(end_update, None)  # type: ignore[arg-type]
        await channel.handle_text(second_update, None)  # type: ignore[arg-type]

        assert runtime.end_session_calls == [(first_thread_id, llm)]
        assert end_update.effective_message.replies == [TELEGRAM_SESSION_CLOSED_MESSAGE]
        assert len(runtime.turn_calls) == 2
        assert runtime.turn_calls[1]["thread_id"] != first_thread_id
        assert runtime.turn_calls[1]["thread_id"].startswith("telegram:dm:321:session:")
        assert runtime.turn_calls[1]["expected_liveness"] == "absent"
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_rotation_required_finalizes_after_successful_reply(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(events=[ResponseReadyEvent(output=_output("ready"))])
    runtime.rotate_after_turn = True
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
    )
    channel, _, llm, _ = _channel(runtime=runtime, config=config)

    try:
        update = _update(text="hi", chat_id=321)

        await channel.handle_text(update, None)  # type: ignore[arg-type]

        thread_id = runtime.turn_calls[0]["thread_id"]
        assert update.effective_message.replies == ["ready"]
        assert runtime.end_session_calls == [(thread_id, llm)]
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_rotation_pending_end_failure_retries_before_next_message(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(events=[ResponseReadyEvent(output=_output("ready"))])
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
    )
    channel, _, llm, _ = _channel(runtime=runtime, config=config)

    try:
        await channel.handle_text(_update(text="first", chat_id=321), None)  # type: ignore[arg-type]
        first_thread_id = runtime.turn_calls[0]["thread_id"]
        runtime.end_session_error = RuntimeError("temporary close failure")
        end_update = _update(text="/end", chat_id=321)

        await channel.handle_end(end_update, None)  # type: ignore[arg-type]

        assert end_update.effective_message.replies == [TELEGRAM_MAINTENANCE_MESSAGE]
        assert runtime.end_session_calls == [(first_thread_id, llm)]

        runtime.end_session_error = None
        next_update = _update(text="second", chat_id=321)
        await channel.handle_text(next_update, None)  # type: ignore[arg-type]

        assert runtime.end_session_calls == [
            (first_thread_id, llm),
            (first_thread_id, llm),
        ]
        assert next_update.effective_message.replies == ["ready"]
        assert len(runtime.turn_calls) == 2
        assert runtime.turn_calls[1]["thread_id"] != first_thread_id
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_rotation_retries_once_after_session_lease_expired(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(events=[ResponseReadyEvent(output=_output("ready"))])
    runtime.turn_errors = [
        SessionLeaseExpired("stale-thread", SessionStatus.ROTATION_REQUIRED)
    ]
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
    )
    channel, _, _, _ = _channel(runtime=runtime, config=config)

    try:
        update = _update(text="hi", chat_id=321)

        await channel.handle_text(update, None)  # type: ignore[arg-type]

        assert update.effective_message.replies == ["ready"]
        assert len(runtime.turn_calls) == 2
        assert runtime.turn_calls[0]["thread_id"] != runtime.turn_calls[1]["thread_id"]
        assert runtime.turn_calls[1]["expected_liveness"] == "absent"
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_startup_recovery_closes_orphan_unclosed_session(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
    )
    registry = SqliteTelegramSessionRegistry(config.session_registry_sqlite_path)
    channel, _, _, _ = _channel(
        runtime=runtime,
        config=config,
        session_registry=registry,
    )
    thread_id = await registry.create_session(321)
    conn = registry._ensure_conn()  # noqa: SLF001
    await conn.execute(
        """
        UPDATE telegram_chat_active
        SET active_thread_id = NULL, active_started_at = NULL
        WHERE chat_id = ?
        """,
        ("321",),
    )
    await conn.commit()

    try:
        await channel.start()

        async with conn.execute(
            """
            SELECT closed_at, closed_reason
            FROM telegram_chat_sessions
            WHERE thread_id = ?
            """,
            (thread_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] is not None
        assert row[1] == "restart_stale"
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_maintenance_sweep_skips_locked_chat(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
    )
    channel, _, _, _ = _channel(runtime=runtime, config=config)
    registry = channel._require_registry()  # noqa: SLF001
    thread_id = await registry.create_session(321)
    runtime.session_statuses[thread_id] = SessionStatus.EXPIRED_UNFINALIZED
    lock = channel._chat_lock(321)  # noqa: SLF001
    await lock.acquire()

    try:
        await channel._maintenance_sweep_once()  # noqa: SLF001

        assert runtime.end_session_calls == []
        active = await registry.get_active(321)
        assert active is not None
        assert active.active_thread_id == thread_id
    finally:
        lock.release()
        await channel.stop()


@pytest.mark.asyncio
async def test_reclaim_first_failure_remains_retryable(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    runtime.reset_thread_error = RuntimeError("temporary reset failure")
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
        reclaim_grace_seconds=0.0,
    )
    registry = SqliteTelegramSessionRegistry(config.session_registry_sqlite_path)
    channel, _, _, _ = _channel(
        runtime=runtime,
        config=config,
        session_registry=registry,
    )
    thread_id = await registry.create_session(321)
    await registry.close_thread(321, thread_id, "end_command")

    try:
        await channel._reclaim_sweep_once()  # noqa: SLF001

        conn = registry._ensure_conn()  # noqa: SLF001
        async with conn.execute(
            """
            SELECT reclaim_attempts, reclaim_stuck_at, last_reclaim_error
            FROM telegram_chat_sessions
            WHERE thread_id = ?
            """,
            (thread_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row == (1, None, "temporary reset failure")
    finally:
        await channel.stop()


@pytest.mark.asyncio
async def test_rotation_real_runtime_registry_integration(tmp_path: Path) -> None:
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
    )
    async with PersistentAgentRuntime(
        sqlite_path=tmp_path / "threads.sqlite3",
        memory_sqlite_path=tmp_path / "memory.sqlite3",
        crisis_log_sqlite_path=tmp_path / "crisis.sqlite3",
        finalize_active_sessions_on_close=False,
    ) as runtime:
        channel = TelegramChannel(
            config=config,
            runtime=runtime,
            llm_client=None,  # type: ignore[arg-type]
            response_llm_client=None,  # type: ignore[arg-type]
            session_registry=_registry_for_config(config),
        )
        registry = channel._require_registry()  # noqa: SLF001

        try:
            first_update = _update(text="first", chat_id=321)
            end_update = _update(text="/end", chat_id=321)
            second_update = _update(text="second", chat_id=321)

            await channel.handle_text(first_update, None)  # type: ignore[arg-type]
            first_active = await registry.get_active(321)
            assert first_active is not None
            first_thread_id = first_active.active_thread_id
            assert first_thread_id is not None
            assert first_thread_id.startswith("telegram:dm:321:session:")

            await channel.handle_end(end_update, None)  # type: ignore[arg-type]
            after_end = await registry.get_active(321)
            assert after_end is not None
            assert after_end.active_thread_id is None

            await channel.handle_text(second_update, None)  # type: ignore[arg-type]
            second_active = await registry.get_active(321)
            assert second_active is not None
            second_thread_id = second_active.active_thread_id
            assert second_thread_id is not None
            assert second_thread_id != first_thread_id
        finally:
            await channel.stop()


@pytest.mark.asyncio
async def test_done_event_is_fallback_when_response_ready_is_absent() -> None:
    runtime = _FakeRuntime(events=[DoneEvent(output=_output("done"))])
    channel, _, _, _ = _channel(runtime=runtime)
    update = _update(text="hi")

    await channel.handle_text(update, None)  # type: ignore[arg-type]

    assert update.effective_message.replies == ["done"]


@pytest.mark.asyncio
async def test_response_ready_sends_only_one_reply() -> None:
    runtime = _FakeRuntime(
        events=[
            ResponseReadyEvent(output=_output("ready")),
            DoneEvent(output=_output("done")),
        ]
    )
    channel, _, _, _ = _channel(runtime=runtime)
    update = _update(text="hi")

    await channel.handle_text(update, None)  # type: ignore[arg-type]

    assert update.effective_message.replies == ["ready"]


@pytest.mark.asyncio
async def test_long_replies_are_split() -> None:
    runtime = _FakeRuntime(events=[ResponseReadyEvent(output=_output("x" * 5000))])
    channel, _, _, _ = _channel(runtime=runtime)
    update = _update(text="hi")

    await channel.handle_text(update, None)  # type: ignore[arg-type]

    assert len(update.effective_message.replies) == 2
    assert all(len(reply) <= 4000 for reply in update.effective_message.replies)


def test_split_telegram_text_preserves_line_boundaries_when_possible() -> None:
    chunks = split_telegram_text("a\nb\nc", limit=3)

    assert chunks == ["a\nb", "c"]


def test_render_telegram_markdown_formats_bold_and_links() -> None:
    rendered = render_telegram_markdown(
        "A few **types of wearables**. "
        "[pmc.ncbi.nlm.nih.gov](https://pmc.ncbi.nlm.nih.gov/articles/PMC12789550/)"
    )

    assert "<b>types of wearables</b>" in rendered
    assert (
        '<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC12789550/">'
        "pmc.ncbi.nlm.nih.gov</a>"
    ) in rendered
    assert "**" not in rendered


def test_render_telegram_markdown_escapes_html_injection() -> None:
    rendered = render_telegram_markdown("Hello <script>alert(1)</script>")

    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script>" not in rendered


def test_split_telegram_markdown_html_keeps_balanced_tags() -> None:
    chunks = split_telegram_markdown_html(f"**{'long text ' * 300}**", limit=400)

    assert len(chunks) > 1
    assert all(len(chunk) <= 400 for chunk in chunks)
    assert all(chunk.count("<b>") == chunk.count("</b>") for chunk in chunks)


@pytest.mark.asyncio
async def test_reply_sends_rendered_html_parse_mode() -> None:
    channel, _, _, _ = _channel()
    update = _update()

    await channel._reply(  # noqa: SLF001
        update,
        "**bold** [site](https://example.com)",
    )

    assert update.effective_message.replies == [
        '<b>bold</b> <a href="https://example.com">site</a>'
    ]
    assert update.effective_message.reply_kwargs == [{"parse_mode": "HTML"}]


@pytest.mark.asyncio
async def test_registry_upgrades_boolean_legacy_migration_state(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "telegram_sessions.sqlite3"
    with sqlite3.connect(registry_path) as conn:
        conn.execute(
            """
            CREATE TABLE telegram_chat_active (
                chat_id TEXT PRIMARY KEY,
                active_thread_id TEXT,
                active_started_at TEXT,
                migrated_legacy INTEGER NOT NULL DEFAULT 0,
                migration_last_error TEXT,
                migration_updated_at TEXT,
                close_requested_reason TEXT,
                close_requested_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO telegram_chat_active(chat_id, migrated_legacy)
            VALUES('321', 1), ('654', 0)
            """
        )

    registry = SqliteTelegramSessionRegistry(registry_path)
    try:
        finalized = await registry.get_active(321)
        pending = await registry.get_active(654)

        assert finalized is not None
        assert finalized.legacy_migration_state == "finalized"
        assert pending is not None
        assert pending.legacy_migration_state == "pending"
    finally:
        await registry.aclose()


def test_build_telegram_session_registry_selects_sqlite(tmp_path: Path) -> None:
    registry = build_telegram_session_registry(
        backend="sqlite",
        sqlite_path=tmp_path / "telegram_sessions.sqlite3",
        database_url=None,
    )

    assert isinstance(registry, SqliteTelegramSessionRegistry)


def test_build_telegram_session_registry_selects_postgres() -> None:
    registry = build_telegram_session_registry(
        backend="postgres",
        sqlite_path=Path(":memory:"),
        database_url="postgresql://opencouch:opencouch@localhost:5432/opencouch",
    )

    assert isinstance(registry, PostgresTelegramSessionRegistry)


def test_build_telegram_session_registry_prefers_postgres_url_over_sqlite_backend(
    tmp_path: Path,
) -> None:
    registry = build_telegram_session_registry(
        backend="sqlite",
        sqlite_path=tmp_path / "telegram_sessions.sqlite3",
        database_url="postgresql://opencouch:opencouch@localhost:5432/opencouch",
    )

    assert isinstance(registry, PostgresTelegramSessionRegistry)


def test_build_telegram_session_registry_falls_back_without_postgres_url() -> None:
    registry = build_telegram_session_registry(
        backend="postgres",
        sqlite_path=Path(":memory:"),
        database_url=None,
    )

    assert isinstance(registry, SqliteTelegramSessionRegistry)


@pytest.mark.asyncio
async def test_postgres_registry_roundtrips_rotated_session_state() -> None:
    database_url = _postgres_database_url()
    if database_url is None:
        pytest.skip(
            "Postgres integration DSN not configured; set "
            "OPENCOUCH_TEST_POSTGRES_URL or OPENCOUCH_MEMORY_DATABASE_URL"
        )

    registry = PostgresTelegramSessionRegistry(database_url)
    chat_id = f"test-{uuid4()}"
    try:
        thread_id = await registry.create_session(chat_id)
        active = await registry.get_active(chat_id)

        assert active is not None
        assert active.active_thread_id == thread_id
        assert active.legacy_migration_state == "finalized"

        await registry.set_pending_close(chat_id, "end_command")
        pending = await registry.get_active(chat_id)
        assert pending is not None
        assert pending.close_requested_reason == "end_command"

        await registry.close_thread(chat_id, thread_id, "end_command")
        closed = await registry.get_active(chat_id)
        assert closed is not None
        assert closed.active_thread_id is None

        reclaimable = await registry.list_reclaimable(timedelta(seconds=0))
        assert any(session.thread_id == thread_id for session in reclaimable)

        await registry.mark_reclaim_result(thread_id)
        assert all(
            session.thread_id != thread_id
            for session in await registry.list_reclaimable(timedelta(seconds=0))
        )
    finally:
        await registry.aclose()


def test_resolve_telegram_memory_mode_refuses_guest() -> None:
    with pytest.raises(TelegramConfigurationError, match="persistent memory"):
        resolve_telegram_memory_mode("guest")


def test_resolve_telegram_memory_mode_accepts_persistent_aliases() -> None:
    assert resolve_telegram_memory_mode("persistent") is MemoryMode.LOCAL
    assert resolve_telegram_memory_mode("local") is MemoryMode.LOCAL
    assert resolve_telegram_memory_mode("synced") is MemoryMode.SYNCED


def test_gateway_lock_path_is_scoped_to_store_dir(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "threads.sqlite3"

    assert telegram_gateway_lock_path(sqlite_path) == tmp_path / "telegram_gateway.lock"


def test_gateway_lock_refuses_second_holder(tmp_path: Path) -> None:
    lock_path = tmp_path / "telegram_gateway.lock"

    with TelegramGatewayLock(lock_path):
        with pytest.raises(GatewayLockError, match="already holds"):
            with TelegramGatewayLock(lock_path):
                pass


def test_gateway_lock_releases_on_context_exit(tmp_path: Path) -> None:
    lock_path = tmp_path / "telegram_gateway.lock"

    with TelegramGatewayLock(lock_path):
        pass
    with TelegramGatewayLock(lock_path):
        pass


def test_gateway_lock_rewrites_single_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "telegram_gateway.lock"

    with TelegramGatewayLock(lock_path):
        assert lock_path.read_text(encoding="utf-8") == f"{os.getpid()}\n"
    with TelegramGatewayLock(lock_path):
        assert lock_path.read_text(encoding="utf-8") == f"{os.getpid()}\n"


@pytest.mark.asyncio
async def test_stop_event_from_signals_raises_when_signal_wiring_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import channels.gateway as gateway

    class _Loop:
        def add_signal_handler(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("not main thread")

    monkeypatch.setattr(gateway.asyncio, "get_running_loop", lambda: _Loop())

    def _raise_signal_error(*args: Any, **kwargs: Any) -> None:
        raise ValueError("not main thread")

    monkeypatch.setattr(gateway.signal, "signal", _raise_signal_error)

    with pytest.raises(RuntimeError, match="shutdown signal handlers"):
        gateway._stop_event_from_signals()


@pytest.mark.asyncio
async def test_gateway_wiring_uses_runtime_context_and_selected_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import channels.gateway as gateway

    calls: dict[str, Any] = {}
    llm = _FakeLLM()
    response_llm = _FakeLLM()
    config = _config()

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            calls["runtime_kwargs"] = kwargs

        async def __aenter__(self) -> _Runtime:
            calls["entered"] = True
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            calls["exited"] = True

    class _Lock:
        def __init__(self, path: Path) -> None:
            calls["lock_path"] = path

        def __enter__(self) -> _Lock:
            calls["lock_entered"] = True
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            calls["lock_exited"] = True

    async def _run_application(*args: Any, **kwargs: Any) -> None:
        calls["drop_pending_updates"] = kwargs["drop_pending_updates"]

    monkeypatch.setattr(gateway, "load_runtime_env", lambda: None)
    monkeypatch.setattr(gateway, "create_configured_control_llm_client", lambda: llm)
    monkeypatch.setattr(
        gateway,
        "create_configured_response_llm_client",
        lambda tier: response_llm,
    )
    monkeypatch.setattr(gateway, "PersistentAgentRuntime", _Runtime)
    monkeypatch.setattr(gateway, "TelegramGatewayLock", _Lock)
    monkeypatch.setattr(
        gateway, "build_telegram_application", lambda **kwargs: object()
    )
    monkeypatch.setattr(gateway, "run_telegram_application", _run_application)
    monkeypatch.setattr(
        gateway, "resolve_telegram_memory_mode", lambda: MemoryMode.LOCAL
    )

    await run_telegram_gateway(config=config)

    assert calls["lock_entered"] is True
    assert calls["entered"] is True
    assert calls["runtime_kwargs"]["default_llm_client"] is llm
    assert calls["runtime_kwargs"]["memory_mode"] is MemoryMode.LOCAL
    assert calls["runtime_kwargs"]["finalize_active_sessions_on_close"] is True
    assert calls["runtime_kwargs"]["auto_finalize_excluded"] is None
    assert calls["drop_pending_updates"] is True
    assert calls["exited"] is True
    assert calls["lock_exited"] is True


@pytest.mark.asyncio
async def test_gateway_wiring_defers_rotated_session_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import channels.gateway as gateway

    calls: dict[str, Any] = {}
    llm = _FakeLLM()
    response_llm = _FakeLLM()
    config = _config(
        thread_rotation_enabled=True,
        session_registry_sqlite_path=tmp_path / "telegram_sessions.sqlite3",
    )

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            calls["runtime_kwargs"] = kwargs

        async def __aenter__(self) -> _Runtime:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            pass

    class _Channel:
        def __init__(self, **kwargs: Any) -> None:
            calls["channel_kwargs"] = kwargs

        async def start(self) -> None:
            calls["channel_started"] = True

        async def stop(self) -> None:
            calls["channel_stopped"] = True

    class _Lock:
        def __init__(self, path: Path) -> None:
            pass

        def __enter__(self) -> _Lock:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            pass

    async def _run_application(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(gateway, "load_runtime_env", lambda: None)
    monkeypatch.setattr(gateway, "create_configured_control_llm_client", lambda: llm)
    monkeypatch.setattr(
        gateway,
        "create_configured_response_llm_client",
        lambda tier: response_llm,
    )
    monkeypatch.setattr(gateway, "PersistentAgentRuntime", _Runtime)
    monkeypatch.setattr(gateway, "TelegramChannel", _Channel)
    monkeypatch.setattr(gateway, "TelegramGatewayLock", _Lock)
    monkeypatch.setattr(
        gateway, "build_telegram_application", lambda **kwargs: object()
    )
    monkeypatch.setattr(gateway, "run_telegram_application", _run_application)
    monkeypatch.setattr(
        gateway, "resolve_telegram_memory_mode", lambda: MemoryMode.LOCAL
    )

    await run_telegram_gateway(config=config)

    assert calls["runtime_kwargs"]["finalize_active_sessions_on_close"] is False
    predicate = calls["runtime_kwargs"]["auto_finalize_excluded"]
    assert predicate("telegram:dm:321:session:01ABC") is True
    assert predicate("telegram:dm:321") is False
    assert calls["channel_started"] is True
    assert calls["channel_stopped"] is True
