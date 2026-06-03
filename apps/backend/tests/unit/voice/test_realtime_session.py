from __future__ import annotations

import pytest

from agent.observability.config import TraceConfig
from agent.observability.context import TraceContext, use_trace_context
from agent.observability.events import REALTIME_SESSION_CONFIG_BUILT
from agent.observability.recorder import InMemoryTraceRecorder
from agent.voice.realtime import (
    build_realtime_session_config,
    create_realtime_client_secret,
)


def test_voice_session_config_uses_realtime_model_and_voice_tools() -> None:
    config = build_realtime_session_config(
        thread_id="voice-thread",
        user_id="user-1",
        memory_mode="persistent",
    )

    assert config["type"] == "realtime"
    assert config["model"] == "gpt-realtime-2"
    assert config["reasoning"]["effort"] == "low"
    assert config["audio"]["output"]["voice"]
    assert config["tool_choice"] == "auto"
    assert {tool["name"] for tool in config["tools"]} >= {
        "show_memory_status",
        "show_saved_memory",
        "lookup_crisis_resources",
    }


def test_voice_session_config_emits_privacy_safe_trace_event() -> None:
    recorder = InMemoryTraceRecorder()
    context = TraceContext(trace_id="trace-voice", config=TraceConfig(enabled=True))

    with use_trace_context(context, recorder):
        config = build_realtime_session_config(
            thread_id="voice-thread",
            user_id="user-1",
            memory_mode="persistent",
            assistant_voice="cedar",
        )

    assert config["audio"]["output"]["voice"] == "cedar"
    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.name == REALTIME_SESSION_CONFIG_BUILT
    assert event.attributes == {
        "voice_runtime": "openai_realtime",
        "model": "gpt-realtime-2",
        "voice": "cedar",
        "memory_mode": "persistent",
        "input_transcription_model": "[redacted]",
        "tool_choice": "auto",
    }
    assert "thread_id" not in event.attributes
    assert "user_id" not in event.attributes


def test_voice_session_config_enables_input_audio_transcription() -> None:
    config = build_realtime_session_config(
        thread_id="voice-thread",
        user_id="user-1",
        memory_mode="persistent",
    )

    assert config["audio"]["input"]["transcription"]["model"] == "gpt-4o-transcribe"


def test_voice_session_config_uses_selected_realtime_voice() -> None:
    config = build_realtime_session_config(
        thread_id="voice-thread",
        user_id="user-1",
        memory_mode="persistent",
        assistant_voice="cedar",
    )

    assert config["audio"]["output"]["voice"] == "cedar"


def test_voice_session_lets_realtime_create_responses_after_vad() -> None:
    config = build_realtime_session_config(
        thread_id="voice-thread",
        user_id="user-1",
        memory_mode="persistent",
    )

    assert config["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert config["audio"]["input"]["turn_detection"]["create_response"] is True
    assert config["audio"]["input"]["turn_detection"]["interrupt_response"] is True


def test_incognito_session_instructions_disable_durable_memory() -> None:
    config = build_realtime_session_config(
        thread_id="voice-thread",
        user_id=None,
        memory_mode="incognito",
    )

    assert "incognito" in config["instructions"].lower()
    assert "do not save" in config["instructions"].lower()
    assert "durable memory" in config["instructions"].lower()


def test_voice_session_instructions_include_memory_bootstrap() -> None:
    config = build_realtime_session_config(
        thread_id="voice-thread",
        user_id="user-1",
        memory_mode="persistent",
        memory_context="Saved preferences: use concise replies.",
    )

    assert "Saved preferences: use concise replies." in config["instructions"]


@pytest.mark.asyncio
async def test_create_realtime_client_secret_binds_safety_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClientSecrets:
        async def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return type("SecretResponse", (), {"value": "ek_test_secret"})()

    class _FakeRealtime:
        client_secrets = _FakeClientSecrets()

    class _FakeAsyncOpenAI:
        realtime = _FakeRealtime()

    monkeypatch.setattr("agent.voice.realtime.AsyncOpenAI", _FakeAsyncOpenAI)

    secret = await create_realtime_client_secret(
        session_config={"type": "realtime", "model": "gpt-realtime-2"},
        safety_identifier="safe-user",
    )

    assert secret == "ek_test_secret"
    assert captured["session"] == {"type": "realtime", "model": "gpt-realtime-2"}
    assert captured["extra_headers"] == {"OpenAI-Safety-Identifier": "safe-user"}
