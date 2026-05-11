"""Session bootstrap helpers for the LiveKit voice runtime."""

from __future__ import annotations

import logging
import os
import json
import uuid
from dataclasses import dataclass

from openai.types import realtime as openai_realtime_types

from livekit.agents import TurnHandlingOptions
from livekit.plugins import openai

from agent.memory.modes import MemoryMode
from agent.persistence import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
)
from agent.voice.config import (
    DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE,
    DEFAULT_REALTIME_TRANSCRIPTION_PROMPT,
    normalize_assistant_voice,
)
from agent.voice.session_data import (
    parse_optional_voice_memory_mode,
    parse_voice_memory_mode,
)
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

VOICE_USER_ID_ENV = "OPENCOUCH_VOICE_USER_ID"
VOICE_THREAD_ID_ENV = "OPENCOUCH_VOICE_THREAD_ID"

_runtime: PersistentAgentRuntime | None = None
_llm_client: BaseLLMClient | None = None


@dataclass(frozen=True)
class VoiceSessionMetadata:
    """Effective metadata used to start one LiveKit voice session."""

    user_id: str
    thread_id: str
    transcription_language: str | None
    assistant_voice: str
    memory_mode: MemoryMode


def parse_voice_session_metadata(
    metadata: str | None,
) -> tuple[str | None, str | None, str | None, str | None, MemoryMode | None]:
    """Parse voice session fields from a LiveKit metadata payload."""

    if not metadata:
        return None, None, None, None, None

    try:
        payload = json.loads(metadata)
    except (json.JSONDecodeError, TypeError):
        return None, None, None, None, None

    raw_user_id = payload.get("user_id")
    user_id = raw_user_id.strip() if isinstance(raw_user_id, str) else None
    raw_thread_id = payload.get("thread_id")
    thread_id = raw_thread_id.strip() if isinstance(raw_thread_id, str) else None
    transcription_language = payload.get("transcription_language")
    if isinstance(transcription_language, str):
        transcription_language = transcription_language.strip()
    else:
        transcription_language = None

    assistant_voice = payload.get("assistant_voice")
    if isinstance(assistant_voice, str):
        assistant_voice = normalize_assistant_voice(
            assistant_voice,
            default="marin",
        )
    else:
        assistant_voice = None

    raw_memory_mode = payload.get("memory_mode")
    memory_mode = (
        parse_optional_voice_memory_mode(raw_memory_mode)
        if isinstance(raw_memory_mode, str)
        else None
    )
    return (
        user_id or None,
        thread_id or None,
        transcription_language,
        assistant_voice,
        memory_mode,
    )


def resolve_livekit_session_metadata(
    *,
    job_metadata: str | None,
    participant_metadata: str | None,
) -> VoiceSessionMetadata:
    """Resolve effective owner/session metadata for a voice run."""

    user_id = os.getenv(VOICE_USER_ID_ENV, "").strip() or "voice-user"
    thread_id = (
        os.getenv(VOICE_THREAD_ID_ENV, "").strip() or f"voice-{uuid.uuid4().hex[:12]}"
    )
    transcription_language: str | None = DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE
    assistant_voice = "marin"
    memory_mode = parse_voice_memory_mode(os.getenv("OPENCOUCH_MEMORY_MODE"))

    for metadata in (job_metadata, participant_metadata):
        (
            metadata_user_id,
            metadata_thread_id,
            metadata_transcription_language,
            metadata_assistant_voice,
            metadata_memory_mode,
        ) = parse_voice_session_metadata(metadata)
        if metadata_user_id:
            user_id = metadata_user_id
        if metadata_thread_id:
            thread_id = metadata_thread_id
        if metadata_transcription_language is not None:
            transcription_language = metadata_transcription_language or None
        if metadata_assistant_voice is not None:
            assistant_voice = metadata_assistant_voice
        if metadata_memory_mode is not None:
            memory_mode = metadata_memory_mode

    return VoiceSessionMetadata(
        user_id=user_id,
        thread_id=thread_id,
        transcription_language=transcription_language,
        assistant_voice=assistant_voice,
        memory_mode=memory_mode,
    )


async def ensure_runtime() -> PersistentAgentRuntime:
    """Lazily initialize the shared voice worker runtime."""

    global _runtime, _llm_client  # noqa: PLW0603

    if _runtime is not None:
        return _runtime

    from config import create_configured_control_llm_client, get_settings

    settings = get_settings()
    runtime = PersistentAgentRuntime(
        sqlite_path=str(DEFAULT_THREAD_DB_PATH),
        memory_backend=settings.persistence_backend,
        memory_database_url=settings.memory_database_url,
        thread_persistence_backend=settings.persistence_backend,
        thread_database_url=settings.memory_database_url,
        crisis_log_persistence_backend=settings.persistence_backend,
        crisis_log_database_url=settings.memory_database_url,
        session_feedback_persistence_backend=settings.persistence_backend,
        session_feedback_database_url=settings.memory_database_url,
        memory_sqlite_path=str(DEFAULT_MEMORY_DB_PATH),
        crisis_log_sqlite_path=str(DEFAULT_CRISIS_LOG_DB_PATH),
        memory_mode=MemoryMode.LOCAL,
        finalize_active_sessions_on_close=False,
    )
    await runtime.__aenter__()

    try:
        llm_client = create_configured_control_llm_client()
    except Exception as exc:
        await runtime.__aexit__(type(exc), exc, exc.__traceback__)
        raise RuntimeError(
            "LiveKit voice requires a configured control LLM for crisis, "
            "turn-policy, lookup, and memory finalization decisions."
        ) from exc

    _runtime = runtime
    _llm_client = llm_client
    logger.info(
        "livekit agent: runtime initialized (session_default_mode=%s)",
        parse_voice_memory_mode(os.getenv("OPENCOUCH_MEMORY_MODE")).value,
    )
    return _runtime


def get_control_llm_client() -> BaseLLMClient:
    """Return the configured control LLM client for voice services."""

    if _llm_client is None:
        raise RuntimeError("LiveKit voice control LLM is not initialized.")
    return _llm_client


async def close_runtime() -> None:
    """Close the shared voice runtime if it is open."""

    global _runtime, _llm_client  # noqa: PLW0603

    runtime = _runtime
    if runtime is None:
        return

    _runtime = None
    _llm_client = None
    await runtime.__aexit__(None, None, None)


def should_finalize_transcript_on_shutdown(*, is_fake_job: bool) -> bool:
    """Return whether shutdown should run transcript finalization."""

    if not is_fake_job:
        return True

    raw = os.getenv("OPENCOUCH_VOICE_CONSOLE_FINALIZE_ON_EXIT", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_realtime_model(
    *,
    transcription_language: str | None = DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE,
    assistant_voice: str = "marin",
) -> openai.realtime.RealtimeModel:
    """Build the realtime model for session-owned turn routing."""

    return openai.realtime.RealtimeModel(
        voice=normalize_assistant_voice(assistant_voice, default="marin"),
        input_audio_transcription=openai_realtime_types.AudioTranscription(
            model="gpt-4o-transcribe",
            language=transcription_language,
            prompt=DEFAULT_REALTIME_TRANSCRIPTION_PROMPT,
        ),
        input_audio_noise_reduction="near_field",
        turn_detection=None,
    )


def build_turn_handling() -> TurnHandlingOptions:
    """Use session-owned VAD turn detection for spoken-turn control."""

    return TurnHandlingOptions(turn_detection="vad")
