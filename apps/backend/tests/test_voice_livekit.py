"""Targeted tests for the LiveKit voice backend helpers."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient
from livekit.agents import ChatContext

import main
from agent.memory.modes import MemoryMode
from agent.memory.procedural import (
    aadd_procedural_rule,
    aget_procedural_profile,
    build_procedural_rule,
)
from agent.memory.store import OpenCouchMemoryStore
from agent.therapeutic.guided_exercise import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    EXERCISE_CONTINUUM,
    EXERCISE_LEAVES_ON_STREAM,
    EXERCISE_TINY_ACTION,
    EXERCISE_THOUGHT_RECORD,
    EXERCISE_VALUES_COMPASS,
)
from voice.livekit.agent import (
    CrisisAgent,
    HoldSpaceAgent,
    UnderstandingAgent,
    ReflectiveAgent,
    TechniqueAgent,
    TherapeuticAgent,
    _assess_therapeutic_process_state,
    _build_therapeutic_process_guidance,
    _build_therapeutic_agent,
    _compose_therapeutic_agent_instructions,
    _close_runtime,
    _copy_handoff_chat_ctx,
    _copy_therapeutic_handoff_chat_ctx,
    _resolve_livekit_session_metadata,
    _should_finalize_transcript_on_shutdown,
    _build_realtime_model,
    _therapeutic_agent_kind_for_state,
    _build_turn_handling,
    _serialize_session_history,
    _load_procedural_memory,
    _load_semantic_facts,
    _load_turn_relevant_semantic_facts,
)
import voice.livekit.agent as livekit_agent_module
import voice.livekit.api as livekit_api_module
from voice.livekit.activity import VOICE_ACTIVITY_TOPIC
from voice.livekit.session_data import SessionData, TherapeuticProcessState
from voice.livekit.finalization_status import (
    VoiceFinalizationStatus,
    get_voice_finalization_status,
    set_voice_finalization_status,
)
from voice.livekit.tasks import _resolve_exercise
from voice.livekit.tools import (
    answer_grounded_factual_lookup,
    cancel_memory_deletion,
    confirm_memory_deletion,
    crisis_check,
    prepare_memory_deletion,
    provide_crisis_resources,
    save_insight,
    set_proactive_memory_recall,
    show_memory_status,
    show_saved_memory,
)
from voice.realtime import build_voice_system_prompt


class _FakeMemoryStore:
    def __init__(self, search_records: list[SimpleNamespace] | None = None) -> None:
        self.put_calls: list[dict[str, object]] = []
        self.search_records = list(search_records or [])

    async def aput(self, namespace, key, value, **kwargs) -> None:
        self.put_calls.append(
            {
                "namespace": namespace,
                "key": key,
                "value": value,
            }
        )

    async def asearch(self, namespace, query=None, limit=20):
        return self.search_records[:limit]


class _FakeLookupLLM:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
            }
        )
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeLocalParticipant:
    def __init__(self) -> None:
        self.published_data: list[dict[str, object]] = []

    async def publish_data(
        self,
        payload: str | bytes,
        *,
        reliable: bool = True,
        destination_identities: list[str] | None = None,
        topic: str = "",
    ) -> None:
        self.published_data.append(
            {
                "payload": payload,
                "reliable": reliable,
                "destination_identities": destination_identities or [],
                "topic": topic,
            }
        )


def _fake_voice_activity_session(participant: _FakeLocalParticipant) -> SimpleNamespace:
    """Build the minimal LiveKit session shape needed for activity events.

    Args:
        participant: Fake local participant that records published data.

    Returns:
        Object exposing ``room_io.room.local_participant``.
    """

    return SimpleNamespace(
        room_io=SimpleNamespace(
            room=SimpleNamespace(local_participant=participant),
        ),
    )


def _published_voice_activities(
    participant: _FakeLocalParticipant,
) -> list[dict[str, object]]:
    """Decode voice activity events published by a fake participant.

    Args:
        participant: Fake local participant that recorded published data.

    Returns:
        Decoded JSON payloads for the voice activity topic.
    """

    events: list[dict[str, object]] = []
    for published in participant.published_data:
        if published["topic"] != VOICE_ACTIVITY_TOPIC:
            continue
        payload = published["payload"]
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        events.append(json.loads(payload))
    return events


@pytest.mark.asyncio
async def test_save_insight_disallows_interruptions_before_writing() -> None:
    """Durable writes should disable barge-in interruptions."""

    store = _FakeMemoryStore()
    participant = _FakeLocalParticipant()
    interrupted: list[str] = []
    context = SimpleNamespace(
        userdata=SessionData(
            user_id="user-1",
            thread_id="thread-1",
            memory_store=store,
        ),
        session=_fake_voice_activity_session(participant),
        disallow_interruptions=lambda: interrupted.append("blocked"),
    )

    result = await save_insight(context, fact="I like green tea at night.")

    assert result == (
        "Saved for future conversations. Do not narrate the save unless "
        "the user asked you to."
    )
    assert interrupted == ["blocked"]
    assert len(store.put_calls) == 1
    assert store.put_calls[0]["namespace"] == ("user-1", "semantic")
    assert _published_voice_activities(participant)[0] | {
        "timestamp": "ignored",
    } == {
        "type": "voice_activity",
        "activity": "memory_saved",
        "status": "completed",
        "label": "Memory saved",
        "detail": "Saved for future sessions.",
        "timestamp": "ignored",
    }


@pytest.mark.asyncio
async def test_save_insight_skips_guest_mode_memory_writes() -> None:
    """Guest-mode voice sessions should not write durable memory."""

    store = _FakeMemoryStore()
    interrupted: list[str] = []
    context = SimpleNamespace(
        userdata=SessionData(
            user_id="user-1",
            thread_id="thread-1",
            memory_store=store,
            memory_mode=MemoryMode.INCOGNITO,
        ),
        disallow_interruptions=lambda: interrupted.append("blocked"),
    )

    result = await save_insight(context, fact="I like green tea at night.")

    assert result == "Memory is disabled in guest mode."
    assert interrupted == []
    assert store.put_calls == []


async def _seed_voice_memory(store: OpenCouchMemoryStore) -> None:
    """Seed voice-shaped memory records for memory-control tests.

    Args:
        store: In-memory store to seed.

    Returns:
        None: Mutates the supplied store.
    """

    await store.aput(
        ("user-1", "semantic"),
        "fact-presentations",
        {
            "evidence_quote": "Presentations make me anxious.",
            "source": "voice_tool",
            "thread_id": "thread-1",
        },
    )
    await aadd_procedural_rule(
        store,
        user_id="user-1",
        rule=build_procedural_rule(
            rule_text="You prefer shorter responses.",
            evidence=["Please keep replies short."],
        ),
    )


@pytest.mark.asyncio
async def test_show_saved_memory_lists_voice_memory() -> None:
    """Voice memory listing should reuse the shared memory-control surface."""

    store = OpenCouchMemoryStore()
    await _seed_voice_memory(store)
    context = SimpleNamespace(
        userdata=SessionData(user_id="user-1", memory_store=store),
    )

    result = await show_saved_memory(context)

    assert "Here's what I currently have saved" in result
    assert "Presentations make me anxious." in result
    assert "You prefer shorter responses." in result


@pytest.mark.asyncio
async def test_show_memory_status_reports_counts_and_recall_state() -> None:
    """Voice memory status should report counts and proactive recall."""

    store = OpenCouchMemoryStore()
    await _seed_voice_memory(store)
    context = SimpleNamespace(
        userdata=SessionData(user_id="user-1", memory_store=store),
    )

    result = await show_memory_status(context)

    assert "Saved facts: 1" in result
    assert "Session summaries: 0" in result
    assert "Style preferences: 1" in result
    assert "Proactive recall: off" in result


@pytest.mark.asyncio
async def test_set_proactive_memory_recall_updates_profile_and_session_state() -> None:
    """Voice recall toggles should persist and update live session state."""

    store = OpenCouchMemoryStore()
    interrupted: list[str] = []
    userdata = SessionData(user_id="user-1", memory_store=store)
    context = SimpleNamespace(
        userdata=userdata,
        disallow_interruptions=lambda: interrupted.append("blocked"),
    )

    result = await set_proactive_memory_recall(context, enabled=True)
    profile = await aget_procedural_profile(store, user_id="user-1")

    assert "proactive recall on" in result.lower()
    assert profile.proactive_recall_enabled is True
    assert userdata.proactive_recall_enabled is True
    assert interrupted == ["blocked"]


@pytest.mark.asyncio
async def test_prepare_and_confirm_memory_deletion_requires_confirmation() -> None:
    """Voice deletes should only mutate durable memory after confirmation."""

    store = OpenCouchMemoryStore()
    await _seed_voice_memory(store)
    interrupted: list[str] = []
    userdata = SessionData(user_id="user-1", memory_store=store)
    context = SimpleNamespace(
        userdata=userdata,
        disallow_interruptions=lambda: interrupted.append("blocked"),
    )

    prepare_result = await prepare_memory_deletion(context, query="presentations")

    assert "If you want me to delete it" in prepare_result
    assert userdata.pending_memory_delete is not None
    assert await store.aget(("user-1", "semantic"), "fact-presentations") is not None

    confirm_result = await confirm_memory_deletion(context)

    assert confirm_result == "Deleted that saved fact."
    assert interrupted == ["blocked"]
    assert userdata.pending_memory_delete is None
    assert await store.aget(("user-1", "semantic"), "fact-presentations") is None


@pytest.mark.asyncio
async def test_cancel_memory_deletion_preserves_saved_memory() -> None:
    """Cancelling a pending voice delete should leave memory unchanged."""

    store = OpenCouchMemoryStore()
    await _seed_voice_memory(store)
    userdata = SessionData(user_id="user-1", memory_store=store)
    context = SimpleNamespace(userdata=userdata)

    await prepare_memory_deletion(context, query="presentations")
    result = await cancel_memory_deletion(context)

    assert result == "Cancelled. I didn't change your memory."
    assert userdata.pending_memory_delete is None
    assert await store.aget(("user-1", "semantic"), "fact-presentations") is not None


@pytest.mark.asyncio
async def test_voice_memory_control_tools_noop_in_incognito() -> None:
    """Guest-mode voice memory-control requests should not access durable memory."""

    store = OpenCouchMemoryStore()
    context = SimpleNamespace(
        userdata=SessionData(
            user_id="user-1",
            memory_store=store,
            memory_mode=MemoryMode.INCOGNITO,
        ),
    )

    result = await show_saved_memory(context)

    assert "guest mode" in result
    assert await store.arecord_count() == 0


@pytest.mark.asyncio
async def test_answer_grounded_factual_lookup_uses_search_grounding() -> None:
    """Voice factual lookup should reuse the shared grounded lookup helper."""

    llm = _FakeLookupLLM(
        [
            "Singapore has a national mental health helpline at 1771.\n"
            "Sources: mindline.sg"
        ]
    )
    participant = _FakeLocalParticipant()
    context = SimpleNamespace(
        userdata=SessionData(
            user_id="user-1",
            thread_id="thread-1",
            llm_client=llm,
        ),
        session=_fake_voice_activity_session(participant),
    )

    result = await answer_grounded_factual_lookup(
        context,
        query="Can you verify Singapore mental health helpline numbers?",
    )

    assert "1771" in result
    assert [call["use_search"] for call in llm.calls] == [True]
    assert [
        (event["activity"], event["status"], event["label"])
        for event in _published_voice_activities(participant)
    ] == [
        ("factual_lookup", "started", "Lookup started"),
        ("factual_lookup", "completed", "Lookup used"),
    ]


@pytest.mark.asyncio
async def test_answer_grounded_factual_lookup_fails_closed_without_llm() -> None:
    """Voice factual lookup should not guess when search is unavailable."""

    context = SimpleNamespace(userdata=SessionData(user_id="user-1"))

    result = await answer_grounded_factual_lookup(
        context,
        query="Can you check the latest hotline number?",
    )

    assert "won't guess" in result


@pytest.mark.asyncio
async def test_provide_crisis_resources_uses_stated_location_search() -> None:
    """Voice crisis resources should search only after extracting a location."""

    llm = _FakeLookupLLM(
        [
            "Singapore",
            "Samaritans of Singapore | 1767 | https://www.sos.org.sg\n"
            "IMH Mental Health Helpline | 6389 2222 | https://www.imh.com.sg",
        ]
    )
    participant = _FakeLocalParticipant()
    context = SimpleNamespace(
        userdata=SessionData(
            user_id="user-1",
            thread_id="thread-1",
            llm_client=llm,
        ),
        session=_fake_voice_activity_session(participant),
    )

    result = await provide_crisis_resources(
        context,
        location_context="I'm in Singapore and need a crisis hotline.",
    )

    assert "Singapore" in result
    assert "Samaritans of Singapore" in result
    assert "1767" in result
    assert [call["use_search"] for call in llm.calls] == [False, True]
    assert [
        (event["activity"], event["status"], event["label"])
        for event in _published_voice_activities(participant)
    ] == [
        (
            "crisis_resources_lookup",
            "started",
            "Crisis resources search started",
        ),
        ("crisis_resources_lookup", "completed", "Crisis resources found"),
    ]


@pytest.mark.asyncio
async def test_provide_crisis_resources_asks_for_location_without_guessing() -> None:
    """Voice crisis resources should ask for location when none was disclosed."""

    llm = _FakeLookupLLM([""])
    context = SimpleNamespace(
        userdata=SessionData(
            user_id="user-1",
            thread_id="thread-1",
            llm_client=llm,
        ),
    )

    result = await provide_crisis_resources(
        context,
        location_context="I need a crisis hotline.",
    )

    assert "share your country or city" in result
    assert "988" in result
    assert [call["use_search"] for call in llm.calls] == [False]


@pytest.mark.asyncio
async def test_crisis_check_tracks_peak_level_and_trims_handoff_context() -> None:
    """Crisis handoff should retain conversation turns but not prior instructions."""

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="You are the therapeutic agent.")
    chat_ctx.add_message(
        role="system",
        content="Relevant background from prior sessions.",
        extra={"source": "semantic_memory_injection"},
    )
    chat_ctx.add_message(role="user", content="I want to die.")
    chat_ctx.add_message(role="assistant", content="   ")
    chat_ctx.items.append(SimpleNamespace(type="function_call", name="crisis_check"))
    chat_ctx.items.append(
        SimpleNamespace(
            type="function_call_output",
            name="crisis_check",
            output="Transferring to crisis support",
        )
    )

    userdata = SessionData()
    context = SimpleNamespace(
        userdata=userdata,
        session=SimpleNamespace(
            current_agent=SimpleNamespace(chat_ctx=chat_ctx),
        ),
    )

    next_agent, message = await crisis_check(context, concern="I want to die.")

    assert isinstance(next_agent, CrisisAgent)
    assert message == "Transferring to crisis support"
    assert userdata.crisis_level == 2
    assert userdata.max_crisis_level == 2
    assert [
        getattr(item.role, "value", item.role) for item in next_agent.chat_ctx.items
    ] == ["user"]
    assert next_agent.chat_ctx.items[0].text_content == "I want to die."


def test_serialize_session_history_keeps_only_dialogue_messages() -> None:
    """Persisted transcripts should exclude non-dialogue context items."""

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="Injected memory context.")
    chat_ctx.add_message(role="user", content="I feel overwhelmed today.")
    chat_ctx.add_message(role="assistant", content="That sounds exhausting.")
    chat_ctx.add_message(role="assistant", content="   ")

    transcript = _serialize_session_history(chat_ctx)

    assert transcript == [
        {"role": "user", "content": "I feel overwhelmed today."},
        {"role": "assistant", "content": "That sounds exhausting."},
    ]


def test_livekit_token_accepts_explicit_room_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Token creation should respect an explicitly requested room name."""

    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-api-secret")

    client = TestClient(main.app)
    response = client.post(
        "/api/voice/livekit/token",
        json={
            "user_id": "browser-user",
            "thread_id": "thread-123",
            "room_name": "my-test-room",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["room_name"] == "my-test-room"

    token_parts = payload["participant_token"].split(".")
    token_payload = json.loads(
        base64.urlsafe_b64decode(token_parts[1] + "=" * (-len(token_parts[1]) % 4))
    )
    assert token_payload["video"]["room"] == "my-test-room"


def test_livekit_token_can_disable_agent_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Room tokens for manual connect-mode tests should skip auto-dispatch."""

    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-api-secret")

    client = TestClient(main.app)
    response = client.post(
        "/api/voice/livekit/token",
        json={
            "room_name": "my-test-room",
            "dispatch_agent": False,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    token_parts = payload["participant_token"].split(".")
    token_payload = json.loads(
        base64.urlsafe_b64decode(token_parts[1] + "=" * (-len(token_parts[1]) % 4))
    )
    assert token_payload["video"]["room"] == "my-test-room"
    assert "roomConfig" not in token_payload


def test_livekit_token_preserves_transcription_language_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Browser-selected transcription language should reach the worker metadata."""

    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-api-secret")

    client = TestClient(main.app)
    response = client.post(
        "/api/voice/livekit/token",
        json={
            "user_id": "browser-user",
            "thread_id": "thread-123",
            "transcription_language": "es",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    token_parts = payload["participant_token"].split(".")
    token_payload = json.loads(
        base64.urlsafe_b64decode(token_parts[1] + "=" * (-len(token_parts[1]) % 4))
    )
    assert json.loads(token_payload["metadata"])["transcription_language"] == "es"


def test_livekit_token_preserves_session_memory_mode_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Room metadata should carry per-session memory mode to the worker."""

    monkeypatch.setenv("LIVEKIT_URL", "wss://example.livekit.cloud")
    monkeypatch.setenv("LIVEKIT_API_KEY", "test-api-key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "test-api-secret")

    client = TestClient(main.app)
    response = client.post(
        "/api/voice/livekit/token",
        json={
            "user_id": "browser-user",
            "thread_id": "thread-123",
            "memory_mode": "incognito",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["memory_mode"] == "incognito"

    token_parts = payload["participant_token"].split(".")
    token_payload = json.loads(
        base64.urlsafe_b64decode(token_parts[1] + "=" * (-len(token_parts[1]) % 4))
    )
    assert json.loads(token_payload["metadata"])["memory_mode"] == "incognito"


@pytest.mark.asyncio
async def test_livekit_semantic_memory_load_filters_inactive_records() -> None:
    """Voice memory injection should skip dormant, superseded, and hidden facts."""

    store = _FakeMemoryStore(
        search_records=[
            SimpleNamespace(
                key="active",
                value={"evidence_quote": "I like short evening check-ins."},
            ),
            SimpleNamespace(
                key="hidden",
                value={
                    "evidence_quote": "Hidden preference.",
                    "user_visible": False,
                },
            ),
            SimpleNamespace(
                key="dormant",
                value={
                    "evidence_quote": "Dormant preference.",
                    "dormant_at": "2026-04-23T10:00:00Z",
                },
            ),
            SimpleNamespace(
                key="superseded",
                value={
                    "evidence_quote": "Old preference.",
                    "superseded_by": "active",
                },
            ),
        ]
    )

    startup_facts = await _load_semantic_facts(store, "user-1", MemoryMode.LOCAL)
    turn_facts = await _load_turn_relevant_semantic_facts(
        store,
        user_id="user-1",
        mode=MemoryMode.LOCAL,
        query="evening check-in",
    )

    assert startup_facts == ["Previously noted: I like short evening check-ins."]
    assert turn_facts == [
        ("active", "Previously noted: I like short evening check-ins.")
    ]


@pytest.mark.asyncio
async def test_livekit_procedural_memory_load_returns_recall_toggle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LiveKit startup should carry the user's proactive-recall setting."""

    async def _fake_profile(store, *, user_id: str):
        assert user_id == "user-1"
        return SimpleNamespace(
            rules=[SimpleNamespace(rule="Use shorter replies.")],
            proactive_recall_enabled=True,
        )

    monkeypatch.setattr(livekit_agent_module, "aget_procedural_profile", _fake_profile)

    rules, proactive_recall_enabled = await _load_procedural_memory(
        _FakeMemoryStore(),
        "user-1",
        MemoryMode.LOCAL,
    )

    assert rules == ["Use shorter replies."]
    assert proactive_recall_enabled is True


@pytest.mark.asyncio
async def test_voice_finalization_status_roundtrips(tmp_path) -> None:
    """Disconnect finalization status should persist and update per thread."""

    sqlite_path = tmp_path / "voice-status.sqlite3"

    assert (
        await get_voice_finalization_status("thread-123", sqlite_path=sqlite_path)
        is None
    )

    pending = await set_voice_finalization_status(
        "thread-123",
        status="in_progress",
        detail="Saving session memory.",
        sqlite_path=sqlite_path,
    )

    assert pending.thread_id == "thread-123"
    assert pending.status == "in_progress"
    assert pending.detail == "Saving session memory."

    completed = await set_voice_finalization_status(
        "thread-123",
        status="completed",
        detail="Session memory saved.",
        sqlite_path=sqlite_path,
    )

    stored = await get_voice_finalization_status("thread-123", sqlite_path=sqlite_path)

    assert stored is not None
    assert stored.thread_id == "thread-123"
    assert stored.status == "completed"
    assert stored.detail == "Session memory saved."
    assert stored.updated_at == completed.updated_at


def test_livekit_finalization_status_endpoint_returns_404_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The status endpoint should 404 when no disconnect status exists yet."""

    async def _missing(thread_id: str) -> None:
        assert thread_id == "thread-123"
        return None

    monkeypatch.setattr(livekit_api_module, "get_voice_finalization_status", _missing)

    client = TestClient(main.app)
    response = client.get("/api/voice/livekit/finalization-status/thread-123")

    assert response.status_code == 404


def test_livekit_finalization_status_endpoint_returns_persisted_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The status endpoint should surface the worker-written disconnect state."""

    async def _present(thread_id: str) -> VoiceFinalizationStatus:
        assert thread_id == "thread-123"
        return VoiceFinalizationStatus(
            thread_id=thread_id,
            status="completed",
            detail="Session memory saved.",
            updated_at="2026-04-23T12:00:00Z",
        )

    monkeypatch.setattr(livekit_api_module, "get_voice_finalization_status", _present)

    client = TestClient(main.app)
    response = client.get("/api/voice/livekit/finalization-status/thread-123")

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "detail": "Session memory saved.",
        "updated_at": "2026-04-23T12:00:00Z",
    }


def test_resolve_livekit_session_metadata_uses_env_for_local_dev_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local console/dev runs should be able to override the fallback owner/thread."""

    monkeypatch.setenv("OPENCOUCH_VOICE_USER_ID", "hy")
    monkeypatch.setenv("OPENCOUCH_VOICE_THREAD_ID", "voice-dogfood")

    user_id, thread_id, transcription_language, memory_mode = (
        _resolve_livekit_session_metadata(
            job_metadata=None,
            participant_metadata=None,
        )
    )

    assert user_id == "hy"
    assert thread_id == "voice-dogfood"
    assert transcription_language == "en"
    assert memory_mode == MemoryMode.LOCAL


def test_resolve_livekit_session_metadata_prefers_metadata_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real LiveKit metadata should override local env fallbacks."""

    monkeypatch.setenv("OPENCOUCH_VOICE_USER_ID", "hy")
    monkeypatch.setenv("OPENCOUCH_VOICE_THREAD_ID", "voice-dogfood")

    user_id, thread_id, transcription_language, memory_mode = (
        _resolve_livekit_session_metadata(
            job_metadata=json.dumps(
                {
                    "user_id": "browser-user",
                    "thread_id": "thread-from-job",
                    "transcription_language": "es",
                    "memory_mode": "persistent",
                }
            ),
            participant_metadata=json.dumps(
                {
                    "user_id": "participant-user",
                    "thread_id": "thread-from-participant",
                    "transcription_language": "",
                    "memory_mode": "guest",
                }
            ),
        )
    )

    assert user_id == "participant-user"
    assert thread_id == "thread-from-participant"
    assert transcription_language is None
    assert memory_mode == MemoryMode.INCOGNITO


def test_resolve_livekit_session_metadata_ignores_blank_env_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blank env overrides should fall back to the normal default identities."""

    monkeypatch.setenv("OPENCOUCH_VOICE_USER_ID", "   ")
    monkeypatch.setenv("OPENCOUCH_VOICE_THREAD_ID", "")

    user_id, thread_id, transcription_language, memory_mode = (
        _resolve_livekit_session_metadata(
            job_metadata=None,
            participant_metadata=None,
        )
    )

    assert user_id == "voice-user"
    assert thread_id.startswith("voice-")
    assert transcription_language == "en"
    assert memory_mode == MemoryMode.LOCAL


def test_resolve_exercise_rotates_generic_requests_away_from_recent_repeats() -> None:
    """Generic requests should diversify instead of reusing the last default."""

    exercise_type, _ = _resolve_exercise(
        "can you guide me through something grounding",
        recent_exercise_types=(EXERCISE_5_4_3_2_1,),
    )

    assert exercise_type != EXERCISE_5_4_3_2_1


def test_resolve_exercise_keeps_explicit_user_request_even_if_recent() -> None:
    """Explicit requests like breathing should still select the named exercise."""

    exercise_type, _ = _resolve_exercise(
        "let's do box breathing again",
        recent_exercise_types=(EXERCISE_BOX_BREATHING,),
    )

    assert exercise_type == EXERCISE_BOX_BREATHING


def test_resolve_exercise_allows_text_turns_to_use_full_registry() -> None:
    """Typed turns should be able to reach non-voice exercises directly."""

    exercise_type, _ = _resolve_exercise(
        "let's do a thought record",
        input_modality="text",
    )

    assert exercise_type == EXERCISE_THOUGHT_RECORD


def test_resolve_exercise_keeps_voice_turns_on_voice_safe_subset() -> None:
    """Spoken turns should still fall back away from non-voice exercises."""

    exercise_type, _ = _resolve_exercise(
        "let's do a thought record",
        input_modality="voice",
    )

    assert exercise_type == EXERCISE_5_4_3_2_1


@pytest.mark.parametrize(
    ("exercise_request", "expected_exercise"),
    [
        ("can we pick one tiny action I can take today?", EXERCISE_TINY_ACTION),
        ("can we do a values compass exercise?", EXERCISE_VALUES_COMPASS),
        ("can we use a continuum for this all-or-nothing thought?", EXERCISE_CONTINUUM),
    ],
)
def test_resolve_exercise_allows_low_visual_load_voice_exercises(
    exercise_request: str,
    expected_exercise: str,
) -> None:
    """Voice mode should allow verbal exercises that do not require the screen."""

    exercise_type, _ = _resolve_exercise(exercise_request, input_modality="voice")

    assert exercise_type == expected_exercise


def test_resolve_exercise_keeps_visual_imagery_exercise_out_of_voice_mode() -> None:
    """Voice expansion should not add highly visual imagery by default."""

    exercise_type, _ = _resolve_exercise(
        "can we try the leaves on a stream exercise?",
        input_modality="voice",
    )

    assert exercise_type == EXERCISE_5_4_3_2_1
    assert exercise_type != EXERCISE_LEAVES_ON_STREAM


def test_livekit_realtime_session_uses_openai_turn_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realtime sessions should use agent-owned turn completion."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = _build_realtime_model()
    turn_handling = _build_turn_handling()

    assert model._opts.turn_detection is None
    assert model._opts.input_audio_noise_reduction.type == "near_field"
    assert model._opts.input_audio_transcription.model == "gpt-4o-transcribe"
    assert model._opts.input_audio_transcription.language == "en"
    assert "real-time spoken support conversation" in (
        model._opts.input_audio_transcription.prompt
    )

    assert turn_handling["turn_detection"] == "vad"


def test_livekit_realtime_session_can_auto_detect_transcription_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit auto-detect should clear the default language hint."""

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = _build_realtime_model(transcription_language=None)

    assert model._opts.input_audio_transcription.language is None


def test_build_voice_system_prompt_is_active_without_premature_exercises() -> None:
    """Voice prompt should support forward movement without exercise rushing."""

    prompt = build_voice_system_prompt()

    assert "Prioritize verbal brevity" in prompt
    assert "simple presence, warmth, or a short practical response" in prompt
    assert "Do not rush to organize the conversation" in prompt
    assert (
        "Do not introduce grounding, breathing, or other structured exercises" in prompt
    )
    assert (
        "If you are unsure whether to keep talking or start a structured exercise"
        in prompt
    )


def test_build_voice_system_prompt_respects_recall_toggle() -> None:
    """Voice prompt should mirror text recall ON/OFF discipline."""

    off_prompt = build_voice_system_prompt(
        semantic_facts=["Previously noted: The user likes evening check-ins."],
        proactive_recall_enabled=False,
    )
    on_prompt = build_voice_system_prompt(
        semantic_facts=["Previously noted: The user likes evening check-ins."],
        proactive_recall_enabled=True,
    )

    assert "proactive recall is off" in off_prompt
    assert "Do not explicitly mention past sessions or past statements" in off_prompt
    assert "proactive recall is on" in on_prompt
    assert "you may briefly reference it" in on_prompt


def test_therapeutic_process_state_holds_space_for_generic_distress() -> None:
    """Generic distress should default to hold-space rather than guidance."""

    state = _assess_therapeutic_process_state(
        turn_ctx=ChatContext(),
        user_text="I feel really overwhelmed and everything feels like too much.",
        previous=TherapeuticProcessState(),
    )

    assert state.session_intent in {"vent", "regulate"}
    assert state.guidance_permission == "not_yet"
    assert state.process_stage == "hold"
    guidance = _build_therapeutic_process_guidance(state)
    assert "stay close without becoming passive" in guidance.lower()
    assert "has not yet invited directive guidance" in guidance


def test_therapeutic_process_state_orients_when_vent_has_enough_context() -> None:
    """Detailed venting should not remain pure passive holding forever."""

    state = _assess_therapeutic_process_state(
        turn_ctx=ChatContext(),
        user_text=(
            "I've been pushing through work for months because everyone depends "
            "on me, and now even small requests from my boss make me feel trapped."
        ),
        previous=TherapeuticProcessState(),
    )

    assert state.session_intent == "vent"
    assert state.guidance_permission == "not_yet"
    assert state.process_stage in {"orient", "identify"}
    guidance = _build_therapeutic_process_guidance(state)
    assert "choose one place to start" in guidance or "let them correct it" in guidance


def test_therapeutic_process_state_persists_guidance_permission() -> None:
    """Once invited, active guidance should carry across follow-up detail."""

    previous = TherapeuticProcessState(guidance_permission="granted")

    state = _assess_therapeutic_process_state(
        turn_ctx=ChatContext(),
        user_text="It started after I took on two extra projects and stopped sleeping well.",
        previous=previous,
    )

    assert state.guidance_permission == "granted"
    assert state.session_intent == "work"
    assert state.process_stage == "orient"


def test_therapeutic_process_state_respects_guidance_withdrawal() -> None:
    """A user can return the voice agent to listening mode."""

    previous = TherapeuticProcessState(guidance_permission="granted")

    state = _assess_therapeutic_process_state(
        turn_ctx=ChatContext(),
        user_text="I don't want advice right now, can you just listen?",
        previous=previous,
    )

    assert state.guidance_permission == "not_yet"
    assert state.session_intent == "vent"
    assert state.process_stage == "hold"


def test_therapeutic_process_state_moves_into_examine_when_guidance_is_requested() -> (
    None
):
    """Explicit guidance on a hot thought should move into active work."""

    turn_ctx = ChatContext()
    turn_ctx.add_message(
        role="assistant",
        content="We can look at that thought together if you want.",
    )

    state = _assess_therapeutic_process_state(
        turn_ctx=turn_ctx,
        user_text="Yes, please help me figure out why I keep thinking I'm going to ruin everything.",
        previous=TherapeuticProcessState(),
    )

    assert state.guidance_permission == "granted"
    assert state.session_intent in {"understand", "work", "reflect"}
    assert state.process_stage in {"identify", "examine"}
    assert state.formulation.hot_thought
    guidance = _build_therapeutic_process_guidance(state)
    assert "The user has given permission for more active guidance." in guidance


def test_therapeutic_agent_specialization_blocks_discourage_passivity() -> None:
    """Voice specialization prompts should permit useful forward movement."""

    hold_prompt = _compose_therapeutic_agent_instructions(
        base_instructions="base",
        agent_kind="hold_space",
    )
    technique_prompt = _compose_therapeutic_agent_instructions(
        base_instructions="base",
        agent_kind="technique",
    )

    assert "do not simply echo the user" in hold_prompt
    assert "one gentle direction or question, not a plan" in hold_prompt
    assert "conversational micro-steps before formal exercises" in technique_prompt


def test_therapeutic_agent_kind_router_matches_process_stage() -> None:
    """Controller stages should map to the intended specialized agent."""

    hold_state = TherapeuticProcessState(process_stage="hold")
    reflect_state = TherapeuticProcessState(process_stage="identify")
    understand_state = TherapeuticProcessState(process_stage="examine")
    technique_state = TherapeuticProcessState(
        process_stage="ground",
        session_intent="work",
        guidance_permission="granted",
    )

    assert _therapeutic_agent_kind_for_state(hold_state) == "hold_space"
    assert _therapeutic_agent_kind_for_state(reflect_state) == "reflective"
    assert _therapeutic_agent_kind_for_state(understand_state) == "understanding"
    assert _therapeutic_agent_kind_for_state(technique_state) == "technique"


def test_copy_therapeutic_handoff_chat_ctx_preserves_whitelisted_system_context() -> (
    None
):
    """Therapeutic handoffs should keep dialogue and controller/memory notes only."""

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="Base therapeutic instructions.")
    chat_ctx.add_message(role="user", content="I keep doing this.")
    chat_ctx.add_message(role="assistant", content="")
    chat_ctx.add_message(
        role="system",
        content="Therapeutic controller state for this turn:",
        extra={"source": "therapeutic_process_controller"},
    )
    chat_ctx.add_message(
        role="system",
        content="Relevant background from prior sessions.",
        extra={"source": "semantic_memory_injection"},
    )
    chat_ctx.items.append(SimpleNamespace(type="function_call", name="show_memory"))
    chat_ctx.items.append(
        SimpleNamespace(
            type="function_call_output",
            name="show_memory",
            output="Saved facts...",
        )
    )

    copied = _copy_therapeutic_handoff_chat_ctx(chat_ctx)

    assert copied is not None
    contents = [
        (getattr(item.role, "value", item.role), item.text_content, item.extra)
        for item in copied.items
        if item.type == "message"
    ]
    assert ("user", "I keep doing this.", {}) in contents
    assert any(
        role == "system"
        and text == "Therapeutic controller state for this turn:"
        and extra.get("source") == "therapeutic_process_controller"
        for role, text, extra in contents
    )
    assert any(
        role == "system"
        and text == "Relevant background from prior sessions."
        and extra.get("source") == "semantic_memory_injection"
        for role, text, extra in contents
    )
    assert not any(text == "Base therapeutic instructions." for _, text, _ in contents)
    assert not any(text == "" for _, text, _ in contents)
    assert not any(item.type != "message" for item in copied.items)


def test_copy_handoff_chat_ctx_carries_dialogue_only() -> None:
    """Crisis and task handoffs should not inherit system or tool artifacts."""

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="Base therapeutic instructions.")
    chat_ctx.add_message(
        role="system",
        content="Therapeutic controller state for this turn:",
        extra={"source": "therapeutic_process_controller"},
    )
    chat_ctx.add_message(role="user", content="I need a minute.")
    chat_ctx.add_message(role="assistant", content="I'm here with you.")
    chat_ctx.add_message(role="assistant", content="   ")
    chat_ctx.items.append(SimpleNamespace(type="function_call", name="tool"))
    chat_ctx.items.append(
        SimpleNamespace(type="function_call_output", name="tool", output="output")
    )

    copied = _copy_handoff_chat_ctx(chat_ctx)

    assert copied is not None
    assert [
        (getattr(item.role, "value", item.role), item.text_content)
        for item in copied.items
    ] == [
        ("user", "I need a minute."),
        ("assistant", "I'm here with you."),
    ]


def test_build_therapeutic_agent_returns_specialized_agent_class() -> None:
    """Specialized agent builder should pick the matching subclass."""

    assert isinstance(
        _build_therapeutic_agent(agent_kind="hold_space", instructions="base"),
        HoldSpaceAgent,
    )
    assert isinstance(
        _build_therapeutic_agent(agent_kind="reflective", instructions="base"),
        ReflectiveAgent,
    )
    assert isinstance(
        _build_therapeutic_agent(agent_kind="understanding", instructions="base"),
        UnderstandingAgent,
    )
    assert isinstance(
        _build_therapeutic_agent(agent_kind="technique", instructions="base"),
        TechniqueAgent,
    )


@pytest.mark.asyncio
async def test_crisis_de_escalation_drops_crisis_system_and_tool_artifacts() -> None:
    """Returning from crisis should preserve dialogue without crisis-only noise."""

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="Crisis-only instructions.")
    chat_ctx.add_message(role="user", content="I am safe now.")
    chat_ctx.add_message(role="assistant", content="I'm glad you told me.")
    chat_ctx.items.append(
        SimpleNamespace(type="function_call", name="provide_crisis_resources")
    )
    chat_ctx.items.append(
        SimpleNamespace(
            type="function_call_output",
            name="provide_crisis_resources",
            output="Crisis resources...",
        )
    )
    crisis_agent = CrisisAgent(chat_ctx=chat_ctx)
    userdata = SessionData(
        crisis_level=2,
        therapeutic_instructions="base therapeutic instructions",
    )
    context = SimpleNamespace(userdata=userdata)

    next_agent, message = await crisis_agent.de_escalate(context)

    assert isinstance(next_agent, HoldSpaceAgent)
    assert message == (
        "The user has de-escalated. Transitioning back to supportive conversation."
    )
    assert userdata.crisis_level == 0
    assert [
        (getattr(item.role, "value", item.role), item.text_content)
        for item in next_agent.chat_ctx.items
    ] == [
        ("user", "I am safe now."),
        ("assistant", "I'm glad you told me."),
    ]


def test_build_therapeutic_agent_preserves_greeting_delay() -> None:
    """Room-start greetings should be able to wait for remote audio attach."""

    agent = _build_therapeutic_agent(
        agent_kind="hold_space",
        instructions="base",
        greet_on_enter=True,
        greet_delay_seconds=0.6,
    )

    assert isinstance(agent, HoldSpaceAgent)
    assert agent._greet_on_enter is True
    assert agent._greet_delay_seconds == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_therapeutic_agent_on_enter_skips_greeting_when_disabled() -> None:
    """Room-mode agents should stay silent until the user speaks first."""

    generated: list[str] = []

    async def _generate_reply(**kwargs) -> None:
        generated.append(kwargs["instructions"])

    agent = HoldSpaceAgent(instructions="base instructions", greet_on_enter=False)
    agent._activity = SimpleNamespace(
        session=SimpleNamespace(generate_reply=_generate_reply)
    )

    await agent.on_enter()

    assert generated == []


@pytest.mark.asyncio
async def test_therapeutic_agent_on_enter_generates_greeting_when_enabled() -> None:
    """Console-mode agents should keep the existing startup greeting."""

    generated: list[str] = []

    async def _generate_reply(**kwargs) -> None:
        generated.append(kwargs["instructions"])

    agent = HoldSpaceAgent(instructions="base instructions", greet_on_enter=True)
    agent._activity = SimpleNamespace(
        session=SimpleNamespace(generate_reply=_generate_reply)
    )

    await agent.on_enter()

    assert len(generated) == 1
    assert "Greet the user briefly and warmly" in generated[0]


@pytest.mark.asyncio
async def test_on_user_turn_completed_switches_to_understanding_agent() -> None:
    """The shared therapeutic hook should hand off when the phase changes."""

    updates: list[TherapeuticAgent] = []
    userdata = SessionData(
        therapeutic_instructions="base instructions",
    )
    fake_session = SimpleNamespace(
        userdata=userdata,
        update_agent=lambda agent: updates.append(agent),
    )
    agent = HoldSpaceAgent(instructions="base instructions")
    agent._activity = SimpleNamespace(session=fake_session)

    turn_ctx = ChatContext()
    new_message = ChatContext().add_message(
        role="user",
        content="Can you help me figure out why I keep thinking I'm going to ruin everything?",
    )

    await agent.on_user_turn_completed(turn_ctx, new_message)

    assert userdata.therapeutic_state.process_stage == "examine"
    assert len(updates) == 1
    assert isinstance(updates[0], UnderstandingAgent)


@pytest.mark.asyncio
async def test_start_grounding_exercise_blocks_without_explicit_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic distress alone should not trigger a structured exercise."""

    async def _unexpected_grounding_task(**kwargs):
        raise AssertionError("GroundingTask should not run without consent")

    monkeypatch.setattr(
        livekit_agent_module, "GroundingTask", _unexpected_grounding_task
    )

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="user", content="I feel really overwhelmed right now.")
    agent = TherapeuticAgent(instructions="test", chat_ctx=chat_ctx)
    participant = _FakeLocalParticipant()
    context = SimpleNamespace(
        userdata=SessionData(),
        session=_fake_voice_activity_session(participant),
    )

    result = await agent.start_grounding_exercise(context, technique="grounding")

    assert "Do not start a structured exercise yet." in result


@pytest.mark.asyncio
async def test_start_grounding_exercise_allows_explicit_yes_after_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clear yes after an exercise offer should pass the guardrail."""

    carried_roles_and_text: list[tuple[str, str]] = []

    async def _fake_grounding_task(**kwargs):
        carried_ctx = kwargs["chat_ctx"]
        carried_roles_and_text.extend(
            [
                (getattr(item.role, "value", item.role), item.text_content)
                for item in carried_ctx.items
            ]
        )
        return SimpleNamespace(
            exercise_type=EXERCISE_BOX_BREATHING,
            display_name="Box breathing",
            steps_completed=4,
            total_steps=4,
            outcome="completed",
        )

    monkeypatch.setattr(livekit_agent_module, "GroundingTask", _fake_grounding_task)

    chat_ctx = ChatContext()
    chat_ctx.add_message(role="system", content="Base therapeutic instructions.")
    chat_ctx.items.append(SimpleNamespace(type="function_call", name="show_memory"))
    chat_ctx.add_message(
        role="assistant",
        content="Would it help if I guide you through a short breathing exercise?",
    )
    chat_ctx.add_message(role="user", content="Yes, let's try that.")
    agent = TherapeuticAgent(instructions="test", chat_ctx=chat_ctx)
    participant = _FakeLocalParticipant()
    context = SimpleNamespace(
        userdata=SessionData(),
        session=_fake_voice_activity_session(participant),
    )

    result = await agent.start_grounding_exercise(context, technique="breathing")

    assert "The user just finished Box breathing" in result
    assert carried_roles_and_text == [
        (
            "assistant",
            "Would it help if I guide you through a short breathing exercise?",
        ),
        ("user", "Yes, let's try that."),
    ]
    assert [
        (event["activity"], event["status"], event["label"])
        for event in _published_voice_activities(participant)
    ] == [
        ("exercise", "started", "Exercise active"),
        ("exercise", "completed", "Exercise completed"),
    ]


def test_console_shutdown_skips_transcript_finalization_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Console mode should favor fast exit unless explicitly opted in."""

    monkeypatch.delenv("OPENCOUCH_VOICE_CONSOLE_FINALIZE_ON_EXIT", raising=False)

    assert _should_finalize_transcript_on_shutdown(is_fake_job=True) is False
    assert _should_finalize_transcript_on_shutdown(is_fake_job=False) is True


def test_console_shutdown_can_opt_back_into_transcript_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Console-mode transcript finalization should remain opt-in."""

    monkeypatch.setenv("OPENCOUCH_VOICE_CONSOLE_FINALIZE_ON_EXIT", "true")

    assert _should_finalize_transcript_on_shutdown(is_fake_job=True) is True


@pytest.mark.asyncio
async def test_close_runtime_closes_and_clears_cached_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing the cached runtime should release it and clear module globals."""

    class _FakeRuntime:
        def __init__(self) -> None:
            self.exit_calls: list[tuple[object, object, object]] = []

        async def __aexit__(self, exc_type, exc, tb) -> None:
            self.exit_calls.append((exc_type, exc, tb))

    fake_runtime = _FakeRuntime()
    monkeypatch.setattr(livekit_agent_module, "_runtime", fake_runtime)
    monkeypatch.setattr(livekit_agent_module, "_llm_client", object())

    await _close_runtime()

    assert fake_runtime.exit_calls == [(None, None, None)]
    assert livekit_agent_module._runtime is None
    assert livekit_agent_module._llm_client is None
