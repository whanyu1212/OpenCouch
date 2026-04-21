"""Tests for the rewritten Realtime voice session."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from agent.memory.modes import MemoryMode
import voice.realtime as realtime


class _FakeMemoryStore:
    async def asearch(self, namespace, query=None, limit=20):
        return []


class _FakeSessionResource:
    def __init__(self) -> None:
        self.updated_sessions: list[dict] = []

    async def update(self, *, session):
        self.updated_sessions.append(session)


class _FakeInputAudioBufferResource:
    def __init__(self) -> None:
        self.append_calls: list[dict] = []

    async def append(self, *, audio: str) -> None:
        self.append_calls.append({"audio": audio})


class _FakeConversationItemResource:
    def __init__(self) -> None:
        self.truncate_calls: list[dict] = []

    async def truncate(
        self,
        *,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        self.truncate_calls.append(
            {
                "item_id": item_id,
                "content_index": content_index,
                "audio_end_ms": audio_end_ms,
            }
        )


class _FakeConnection:
    def __init__(self) -> None:
        self.session = _FakeSessionResource()
        self.input_audio_buffer = _FakeInputAudioBufferResource()
        self.conversation = SimpleNamespace(item=_FakeConversationItemResource())

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _FakeConnectionManager:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _FakeConnection:
        return self._connection

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _EventConnection:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


@pytest.mark.asyncio
async def test_start_configures_ga_realtime_audio_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Session startup uses the documented GA audio shape."""

    connection = _FakeConnection()

    class _FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            self.realtime = SimpleNamespace(connect=self.connect)

        def connect(self, *, model: str):
            assert model == realtime.DEFAULT_REALTIME_MODEL
            return _FakeConnectionManager(connection)

    def _fake_create_task(coro):
        coro.close()
        return None

    monkeypatch.setattr(realtime, "AsyncOpenAI", _FakeOpenAI)
    monkeypatch.setattr(realtime.asyncio, "create_task", _fake_create_task)

    session = realtime.RealtimeVoiceSession(
        openai_api_key="test",
        memory_store=_FakeMemoryStore(),
        memory_response_style=MemoryMode.LOCAL,
        user_id="voice-user",
        thread_id="voice-thread",
    )

    await session.start()

    [payload] = connection.session.updated_sessions
    assert payload["type"] == "realtime"
    assert payload["model"] == realtime.DEFAULT_REALTIME_MODEL
    assert payload["output_modalities"] == ["audio"]
    assert payload["audio"]["input"]["format"] == realtime.PCM16_AUDIO_FORMAT
    assert payload["audio"]["output"]["format"] == realtime.PCM16_AUDIO_FORMAT
    assert payload["audio"]["output"]["voice"] == realtime.DEFAULT_ASSISTANT_VOICE
    assert (
        payload["audio"]["input"]["transcription"]["model"]
        == realtime.DEFAULT_REALTIME_TRANSCRIPTION_MODEL
    )
    assert (
        payload["audio"]["input"]["transcription"]["language"]
        == realtime.DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE
    )
    assert (
        payload["audio"]["input"]["transcription"]["prompt"]
        == realtime.DEFAULT_REALTIME_TRANSCRIPTION_PROMPT
    )
    assert payload["audio"]["input"]["turn_detection"]["type"] == "server_vad"
    assert payload["audio"]["input"]["turn_detection"]["threshold"] == 0.3
    assert payload["audio"]["input"]["turn_detection"]["create_response"] is True
    assert payload["audio"]["input"]["turn_detection"]["interrupt_response"] is True
    assert payload["audio"]["input"]["turn_detection"]["silence_duration_ms"] == 300


@pytest.mark.asyncio
async def test_send_audio_and_truncate_forward_client_events() -> None:
    """Outgoing helper methods map browser actions to Realtime client events."""

    class _FakeAsyncOpenAI:
        def __init__(self, api_key: str) -> None:
            self.realtime = SimpleNamespace()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(realtime, "AsyncOpenAI", _FakeAsyncOpenAI)

    connection = _FakeConnection()
    session = realtime.RealtimeVoiceSession(
        openai_api_key="test",
        memory_store=_FakeMemoryStore(),
        memory_response_style=MemoryMode.LOCAL,
        user_id="voice-user",
        thread_id="voice-thread",
    )
    session._connection = connection

    await session.send_audio(b"\x01\x02\x03\x04")
    await session.truncate_assistant_audio(
        item_id="assist-1",
        content_index=0,
        audio_end_ms=420,
    )

    assert connection.input_audio_buffer.append_calls == [
        {"audio": base64.b64encode(b"\x01\x02\x03\x04").decode("utf-8")}
    ]
    assert connection.conversation.item.truncate_calls == [
        {"item_id": "assist-1", "content_index": 0, "audio_end_ms": 420}
    ]

    monkeypatch.undo()


@pytest.mark.asyncio
async def test_listen_events_forward_user_and_assistant_transcripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caption deltas stay ephemeral while transcript events commit history."""

    user_transcripts: list[tuple[str, str]] = []
    assistant_transcripts: list[tuple[str, str]] = []
    captions: list[tuple[str, str, str, str]] = []

    async def _on_transcript(text: str, item_id: str) -> None:
        user_transcripts.append((text, item_id))

    async def _on_agent_transcript(text: str, item_id: str) -> None:
        assistant_transcripts.append((text, item_id))

    async def _on_caption(
        role: str,
        text: str,
        item_id: str,
        status: str,
    ) -> None:
        captions.append((role, text, item_id, status))

    class _FakeAsyncOpenAI:
        def __init__(self, api_key: str) -> None:
            self.realtime = SimpleNamespace()

    monkeypatch.setattr(realtime, "AsyncOpenAI", _FakeAsyncOpenAI)

    session = realtime.RealtimeVoiceSession(
        openai_api_key="test",
        memory_store=_FakeMemoryStore(),
        memory_response_style=MemoryMode.LOCAL,
        user_id="voice-user",
        thread_id="voice-thread",
        on_transcript=_on_transcript,
        on_agent_transcript=_on_agent_transcript,
        on_caption=_on_caption,
    )
    session._connection = _EventConnection(
        [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta",
                item_id="user-1",
                content_index=0,
                delta="I feel",
            ),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta",
                item_id="user-1",
                content_index=0,
                delta=" overwhelmed",
            ),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                item_id="user-1",
                transcript="I feel overwhelmed",
                content_index=0,
            ),
            SimpleNamespace(
                type="response.output_audio_transcript.delta",
                item_id="assist-1",
                content_index=0,
                delta="Take",
            ),
            SimpleNamespace(
                type="response.output_audio_transcript.delta",
                item_id="assist-1",
                content_index=0,
                delta=" a breath",
            ),
            SimpleNamespace(
                type="response.output_audio_transcript.done",
                item_id="assist-1",
                content_index=0,
                transcript="Take a breath",
            ),
            SimpleNamespace(
                type="response.output_text.done",
                item_id="assist-1",
                content_index=0,
                text="Take a breath",
            ),
        ]
    )
    session._running = True

    await session._listen_events()

    assert user_transcripts == [("I feel overwhelmed", "user-1")]
    assert assistant_transcripts == [("Take a breath", "assist-1")]
    assert captions == [
        ("user", "I feel", "user-1", "partial"),
        ("user", "I feel overwhelmed", "user-1", "partial"),
        ("user", "I feel overwhelmed", "user-1", "final"),
        ("assistant", "Take", "assist-1", "partial"),
        ("assistant", "Take a breath", "assist-1", "partial"),
        ("assistant", "Take a breath", "assist-1", "final"),
    ]


@pytest.mark.asyncio
async def test_listen_events_interrupts_current_audio_and_drops_late_deltas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late audio from an interrupted item is ignored after speech starts."""

    interruptions: list[str] = []
    truncated: list[tuple[str, int, int]] = []
    audio_chunks: list[tuple[bytes, str, int]] = []
    captions: list[tuple[str, str, str, str]] = []

    async def _on_audio_delta(
        audio_bytes: bytes,
        item_id: str,
        content_index: int,
    ) -> None:
        audio_chunks.append((audio_bytes, item_id, content_index))

    async def _on_interrupted() -> None:
        interruptions.append("interrupted")

    async def _on_truncated(
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        truncated.append((item_id, content_index, audio_end_ms))

    async def _on_caption(
        role: str,
        text: str,
        item_id: str,
        status: str,
    ) -> None:
        captions.append((role, text, item_id, status))

    class _FakeAsyncOpenAI:
        def __init__(self, api_key: str) -> None:
            self.realtime = SimpleNamespace()

    monkeypatch.setattr(realtime, "AsyncOpenAI", _FakeAsyncOpenAI)

    session = realtime.RealtimeVoiceSession(
        openai_api_key="test",
        memory_store=_FakeMemoryStore(),
        memory_response_style=MemoryMode.LOCAL,
        user_id="voice-user",
        thread_id="voice-thread",
        on_audio_delta=_on_audio_delta,
        on_interrupted=_on_interrupted,
        on_truncated=_on_truncated,
        on_caption=_on_caption,
    )
    session._connection = _EventConnection(
        [
            SimpleNamespace(
                type="response.output_audio.delta",
                item_id="assist-1",
                content_index=0,
                delta=base64.b64encode(b"\x01\x02").decode("utf-8"),
            ),
            SimpleNamespace(
                type="response.output_audio_transcript.delta",
                item_id="assist-1",
                content_index=0,
                delta="Take a breath",
            ),
            SimpleNamespace(type="input_audio_buffer.speech_started"),
            SimpleNamespace(
                type="response.output_audio.delta",
                item_id="assist-1",
                content_index=0,
                delta=base64.b64encode(b"\x03\x04").decode("utf-8"),
            ),
            SimpleNamespace(
                type="response.output_audio_transcript.done",
                item_id="assist-1",
                content_index=0,
                transcript="Take a breath",
            ),
            SimpleNamespace(
                type="conversation.item.truncated",
                item_id="assist-1",
                content_index=0,
                audio_end_ms=180,
            ),
        ]
    )
    session._running = True

    await session._listen_events()

    assert interruptions == ["interrupted"]
    assert audio_chunks == [(b"\x01\x02", "assist-1", 0)]
    assert captions == [
        ("assistant", "Take a breath", "assist-1", "partial"),
        ("assistant", "", "assist-1", "cleared"),
    ]
    assert truncated == [("assist-1", 0, 180)]


@pytest.mark.asyncio
async def test_listen_events_forwards_realtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realtime server errors surface to the API callback."""

    errors: list[str] = []

    async def _on_error(message: str) -> None:
        errors.append(message)

    class _FakeAsyncOpenAI:
        def __init__(self, api_key: str) -> None:
            self.realtime = SimpleNamespace()

    monkeypatch.setattr(realtime, "AsyncOpenAI", _FakeAsyncOpenAI)

    session = realtime.RealtimeVoiceSession(
        openai_api_key="test",
        memory_store=_FakeMemoryStore(),
        memory_response_style=MemoryMode.LOCAL,
        user_id="voice-user",
        thread_id="voice-thread",
        on_error=_on_error,
    )
    session._connection = _EventConnection(
        [
            SimpleNamespace(
                type="error",
                error=SimpleNamespace(message="Unsupported audio format"),
            )
        ]
    )
    session._running = True

    await session._listen_events()

    assert errors == ["Unsupported audio format"]


@pytest.mark.asyncio
async def test_end_session_forwards_voice_transcript_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disconnect should route the accumulated transcript through the runtime seam."""

    class _FakeAsyncOpenAI:
        def __init__(self, api_key: str) -> None:
            self.realtime = SimpleNamespace()

    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def end_transcript_session(self, **kwargs) -> None:
            self.calls.append(kwargs)

    monkeypatch.setattr(realtime, "AsyncOpenAI", _FakeAsyncOpenAI)

    runtime = _FakeRuntime()
    session = realtime.RealtimeVoiceSession(
        openai_api_key="test",
        memory_store=_FakeMemoryStore(),
        memory_response_style=MemoryMode.LOCAL,
        user_id="voice-user",
        thread_id="voice-thread",
        runtime=runtime,
        llm_client="fake-llm",
    )
    session._started_at = "2026-04-20T10:00:00Z"
    session._transcript = [
        {"role": "user", "content": "I feel overwhelmed"},
        {"role": "assistant", "content": "Take a breath"},
    ]

    await session.end_session()

    assert runtime.calls == [
        {
            "thread_id": "voice-thread",
            "user_id": "voice-user",
            "transcript": [
                {"role": "user", "content": "I feel overwhelmed"},
                {"role": "assistant", "content": "Take a breath"},
            ],
            "llm_client": "fake-llm",
            "started_at": "2026-04-20T10:00:00Z",
        }
    ]
