"""Tests for the standalone Telegram channel gateway."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
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
from channels.telegram import (
    TELEGRAM_SESSION_CLOSED_MESSAGE,
    TELEGRAM_START_MESSAGE,
    TelegramChannel,
    TelegramConfigurationError,
    TelegramGatewayConfig,
    split_telegram_text,
    telegram_thread_id,
)
from services.llm.base import BaseLLMClient, StructuredResponseT


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

    async def reply_text(self, text: str, **kwargs: Any) -> None:
        self.replies.append(text)
        assert "parse_mode" not in kwargs


class _FakeRuntime:
    def __init__(self, events: list[Any] | None = None) -> None:
        self.events = events or []
        self.turn_calls: list[dict[str, Any]] = []
        self.end_session_calls: list[tuple[str, BaseLLMClient | None]] = []
        self.end_session_result: object | None = None

    async def run_turn_stream(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.turn_calls.append(kwargs)
        for event in self.events:
            yield event

    async def end_session(
        self,
        thread_id: str,
        *,
        llm_client: BaseLLMClient | None = None,
    ) -> object | None:
        self.end_session_calls.append((thread_id, llm_client))
        return self.end_session_result


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
    *, allowed_user_ids: frozenset[int] = frozenset({456})
) -> TelegramGatewayConfig:
    return TelegramGatewayConfig(
        bot_token="token",
        allowed_user_ids=allowed_user_ids,
        owner_id="hanyu",
    )


def _channel(
    *,
    runtime: _FakeRuntime | None = None,
    config: TelegramGatewayConfig | None = None,
    llm: _FakeLLM | None = None,
    response_llm: _FakeLLM | None = None,
) -> tuple[TelegramChannel, _FakeRuntime, _FakeLLM, _FakeLLM]:
    fake_runtime = runtime or _FakeRuntime()
    fake_llm = llm or _FakeLLM()
    fake_response_llm = response_llm or _FakeLLM()
    channel = TelegramChannel(
        config=config or _config(),
        runtime=fake_runtime,  # type: ignore[arg-type]
        llm_client=fake_llm,
        response_llm_client=fake_response_llm,
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

    config = TelegramGatewayConfig.from_env()

    assert config.allowed_user_ids == frozenset({456, 789})
    assert config.drop_pending_updates is True
    assert config.response_model_tier == "fast"


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
    assert calls["drop_pending_updates"] is True
    assert calls["exited"] is True
    assert calls["lock_exited"] is True
