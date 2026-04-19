"""OpenAI Realtime voice session manager.

This module implements a thin server-side adapter around the OpenAI
Realtime WebSocket API. It keeps the session shape close to the
documented GA event model:

- configure the session with ``session.update``
- stream user audio with ``input_audio_buffer.append``
- receive assistant audio from ``response.output_audio.delta``
- receive user transcripts from
  ``conversation.item.input_audio_transcription.completed``
- receive assistant transcripts from
  ``response.output_audio_transcript.done``
- truncate interrupted assistant audio with
  ``conversation.item.truncate``

The goal here is reliability and protocol clarity. Higher-level
behaviors such as crisis handling, tool execution, or memory
extraction should not live inside the realtime socket loop.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from openai import AsyncOpenAI

from agent.memory.modes import MemoryMode
from agent.memory.procedural import aget_procedural_profile
from agent.memory.store import MemoryStore

logger = logging.getLogger(__name__)

PCM16_AUDIO_FORMAT = {"type": "audio/pcm", "rate": 24000}
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_REALTIME_TRANSCRIPTION_MODEL = "gpt-4o-transcribe"
DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE = "en"
DEFAULT_REALTIME_TRANSCRIPTION_PROMPT = (
    "This is a real-time spoken support conversation. Prefer exact wording. "
    "Preserve names, numbers, acronyms, contractions, and filler words when "
    "clearly heard."
)
DEFAULT_ASSISTANT_VOICE = "cedar"
SUPPORTED_REALTIME_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)
SUPPORTED_REALTIME_TRANSCRIPTION_LANGUAGES = (
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "ja",
    "ko",
    "zh",
)

_MAX_PROMPT_CHARS = 12_000
_MAX_MEMORY_ITEMS = 6
_MAX_MEMORY_ITEM_CHARS = 220

AudioDeltaCallback = Callable[[bytes, str, int], Awaitable[None]]
TranscriptCallback = Callable[[str, str], Awaitable[None]]
CaptionStatus = Literal["partial", "final", "cleared"]
CaptionCallback = Callable[[str, str, str, CaptionStatus], Awaitable[None]]
InterruptionCallback = Callable[[], Awaitable[None]]
TruncatedCallback = Callable[[str, int, int], Awaitable[None]]
ErrorCallback = Callable[[str], Awaitable[None]]


def _trim_text(text: str, limit: int) -> str:
    """Trim text to a stable maximum length."""

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _trim_items(items: list[str] | None, *, max_items: int) -> list[str]:
    """Trim and bound a list of prompt items."""

    if not items:
        return []
    trimmed: list[str] = []
    for item in items[:max_items]:
        value = _trim_text(item, _MAX_MEMORY_ITEM_CHARS)
        if value:
            trimmed.append(value)
    return trimmed


def build_voice_system_prompt(
    *,
    semantic_facts: list[str] | None = None,
    episodic_arcs: list[str] | None = None,
    procedural_rules: list[str] | None = None,
) -> str:
    """Build a bounded voice-system prompt for Realtime sessions.

    The old voice prompt was large enough to exceed the Realtime
    instruction budget. This prompt stays deliberately compact and
    keeps only the guidance that materially affects live audio turns.

    Args:
        semantic_facts: Short user facts to keep in working context.
        episodic_arcs: Short summaries from prior sessions.
        procedural_rules: Short style or preference rules for the user.

    Returns:
        A compact prompt string safe for the Realtime session.
    """

    sections = [
        "\n".join(
            [
                "You are OpenCouch, a calm and emotionally intelligent voice support assistant.",
                "Speak like a thoughtful human conversation partner, not like a document.",
                "Keep most replies to one to three short sentences.",
                "Use plain spoken language. Do not use markdown, lists, headings, or long monologues.",
                "Reflect what the user said, ask one helpful follow-up when needed, and prefer grounded practical support over abstract lecturing.",
                "Do not present yourself as a licensed clinician or give medical or legal advice.",
                "If the user sounds in immediate danger or asks for crisis help, stop the normal conversation and tell them to contact local emergency services or a crisis line immediately. If they are in the US or Canada, tell them they can call or text 988.",
            ]
        )
    ]

    trimmed_rules = _trim_items(procedural_rules, max_items=_MAX_MEMORY_ITEMS)
    if trimmed_rules:
        sections.append(
            "User preferences:\n" + "\n".join(f"- {rule}" for rule in trimmed_rules)
        )

    trimmed_facts = _trim_items(semantic_facts, max_items=_MAX_MEMORY_ITEMS)
    if trimmed_facts:
        sections.append(
            "Known context about the user:\n"
            + "\n".join(f"- {fact}" for fact in trimmed_facts)
        )

    trimmed_arcs = _trim_items(episodic_arcs, max_items=3)
    if trimmed_arcs:
        sections.append(
            "Relevant prior sessions:\n" + "\n".join(f"- {arc}" for arc in trimmed_arcs)
        )

    prompt = "\n\n".join(sections)
    return _trim_text(prompt, _MAX_PROMPT_CHARS)


class RealtimeVoiceSession:
    """Manage one OpenAI Realtime speech-to-speech session.

    Args:
        openai_api_key: OpenAI API key for the backend session.
        memory_store: Memory store used for loading compact prompt context.
        memory_mode: Active memory mode for the session.
        user_id: Stable user identifier.
        thread_id: Stable thread identifier.
        voice: Assistant voice ID.
        transcription_language: Optional ISO-639-1 input transcription language.
        llm_client: Reserved for future use.
        embedding_provider: Reserved for future use.
        on_audio_delta: Callback for decoded assistant audio chunks.
        on_transcript: Callback for final user transcripts.
        on_agent_transcript: Callback for final assistant transcripts.
        on_caption: Callback for ephemeral live caption updates.
        on_interrupted: Callback when VAD detects user speech during assistant output.
        on_truncated: Callback when the server acknowledges truncation.
        on_error: Callback for surfaced Realtime errors.
    """

    def __init__(
        self,
        *,
        openai_api_key: str,
        memory_store: MemoryStore,
        memory_mode: MemoryMode,
        user_id: str,
        thread_id: str,
        voice: str = DEFAULT_ASSISTANT_VOICE,
        transcription_language: str | None = DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE,
        llm_client: Any = None,
        embedding_provider: Any = None,
        on_audio_delta: AudioDeltaCallback | None = None,
        on_transcript: TranscriptCallback | None = None,
        on_agent_transcript: TranscriptCallback | None = None,
        on_caption: CaptionCallback | None = None,
        on_interrupted: InterruptionCallback | None = None,
        on_truncated: TruncatedCallback | None = None,
        on_error: ErrorCallback | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=openai_api_key)
        self._memory_store = memory_store
        self._memory_mode = memory_mode
        self._user_id = user_id
        self._thread_id = thread_id
        self._voice = voice
        self._transcription_language = transcription_language
        self._connection = None
        self._running = False
        self._listener_task: asyncio.Task[None] | None = None

        # Stored for future session-end processing if needed.
        self._transcript: list[dict[str, str]] = []

        self._on_audio_delta = on_audio_delta
        self._on_transcript = on_transcript
        self._on_agent_transcript = on_agent_transcript
        self._on_caption = on_caption
        self._on_interrupted = on_interrupted
        self._on_truncated = on_truncated
        self._on_error = on_error

        self._current_output_item_id: str | None = None
        self._current_output_content_index = 0
        self._ignored_output_item_ids: set[str] = set()

        # Currently unused, but retained to avoid breaking callers that
        # still pass these dependencies.
        self._llm_client = llm_client
        self._embedding_provider = embedding_provider

    async def start(self) -> None:
        """Open the Realtime connection and configure the session."""

        semantic_facts = await self._load_semantic_facts()
        episodic_arcs = await self._load_episodic_arcs()
        procedural_rules = await self._load_procedural_rules()

        system_prompt = build_voice_system_prompt(
            semantic_facts=semantic_facts,
            episodic_arcs=episodic_arcs,
            procedural_rules=procedural_rules,
        )

        logger.info(
            "realtime session: starting user=%s thread=%s model=%s prompt_chars=%d language=%s",
            self._user_id,
            self._thread_id,
            DEFAULT_REALTIME_MODEL,
            len(system_prompt),
            self._transcription_language or "auto",
        )

        transcription_config: dict[str, Any] = {
            "model": DEFAULT_REALTIME_TRANSCRIPTION_MODEL,
            "prompt": DEFAULT_REALTIME_TRANSCRIPTION_PROMPT,
        }
        if self._transcription_language:
            transcription_config["language"] = self._transcription_language

        self._connection = await self._client.realtime.connect(
            model=DEFAULT_REALTIME_MODEL,
        ).__aenter__()

        await self._connection.session.update(
            session={
                "type": "realtime",
                "model": DEFAULT_REALTIME_MODEL,
                "instructions": system_prompt,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": dict(PCM16_AUDIO_FORMAT),
                        "noise_reduction": {"type": "near_field"},
                        "transcription": transcription_config,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.3,
                            "interrupt_response": True,
                            "create_response": True,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 300,
                        },
                    },
                    "output": {
                        "format": dict(PCM16_AUDIO_FORMAT),
                        "voice": self._voice,
                    },
                },
            }
        )

        self._running = True
        self._listener_task = asyncio.create_task(self._listen_events())

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Append browser audio to the Realtime input buffer."""

        if self._connection is None:
            return
        encoded = base64.b64encode(audio_bytes).decode("utf-8")
        await self._connection.input_audio_buffer.append(audio=encoded)

    async def truncate_assistant_audio(
        self,
        *,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> None:
        """Synchronize interrupted playback with the Realtime conversation."""

        if self._connection is None or not item_id:
            return

        try:
            await self._connection.conversation.item.truncate(
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=audio_end_ms,
            )
        except Exception:
            logger.warning(
                "realtime session: conversation.item.truncate failed",
                exc_info=True,
            )

    async def end_session(self) -> None:
        """Finalize the session.

        The rewritten Realtime path intentionally keeps the socket loop
        narrow and does not run summarization or extraction on teardown.
        """

        return None

    async def close(self) -> None:
        """Close the Realtime connection and stop the listener task."""

        self._running = False

        if self._listener_task is not None:
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
            self._listener_task = None

        if self._connection is not None:
            with contextlib.suppress(Exception):
                await self._connection.__aexit__(None, None, None)
            self._connection = None

    async def _listen_events(self) -> None:
        """Consume Realtime server events and map them to app callbacks."""

        if self._connection is None:
            return

        user_captions: dict[tuple[str, int], str] = {}
        assistant_transcripts: dict[tuple[str, int], str] = {}

        try:
            async for event in self._connection:
                if not self._running:
                    break

                event_type = event.type

                if event_type in {
                    "session.created",
                    "session.updated",
                    "response.done",
                }:
                    logger.info("realtime event: %s", event_type)

                if event_type == "input_audio_buffer.speech_started":
                    if self._current_output_item_id:
                        interrupted_item_id = self._current_output_item_id
                        self._ignored_output_item_ids.add(interrupted_item_id)
                        for key in list(assistant_transcripts):
                            if key[0] == interrupted_item_id:
                                assistant_transcripts.pop(key, None)
                        self._current_output_item_id = None
                        self._current_output_content_index = 0
                        if self._on_caption is not None:
                            await self._on_caption(
                                "assistant",
                                "",
                                interrupted_item_id,
                                "cleared",
                            )
                        if self._on_interrupted is not None:
                            await self._on_interrupted()

                elif event_type == "conversation.item.input_audio_transcription.delta":
                    item_id = getattr(event, "item_id", "")
                    content_index = getattr(event, "content_index", 0)
                    key = (item_id, content_index)
                    user_captions[key] = user_captions.get(key, "") + (
                        event.delta or ""
                    )

                    caption = user_captions[key].strip()
                    if caption and self._on_caption is not None:
                        await self._on_caption("user", caption, item_id, "partial")

                elif (
                    event_type
                    == "conversation.item.input_audio_transcription.completed"
                ):
                    item_id = getattr(event, "item_id", "")
                    content_index = getattr(event, "content_index", 0)
                    key = (item_id, content_index)
                    partial_transcript = user_captions.pop(key, "")
                    transcript = (
                        (event.transcript or partial_transcript) or ""
                    ).strip()
                    if transcript:
                        if self._on_caption is not None:
                            await self._on_caption("user", transcript, item_id, "final")
                        self._transcript.append({"role": "user", "content": transcript})
                        if self._on_transcript is not None:
                            await self._on_transcript(transcript, item_id)

                elif event_type == "conversation.item.input_audio_transcription.failed":
                    error = getattr(event, "error", None)
                    message = getattr(error, "message", None) or (
                        "Input audio transcription failed"
                    )
                    logger.warning("realtime transcription failed: %s", message)
                    if self._on_error is not None:
                        await self._on_error(message)

                elif event_type == "response.output_audio.delta":
                    item_id = getattr(event, "item_id", "")
                    if item_id in self._ignored_output_item_ids:
                        continue

                    content_index = getattr(event, "content_index", 0)
                    self._current_output_item_id = item_id
                    self._current_output_content_index = content_index

                    if self._on_audio_delta is not None:
                        audio_bytes = base64.b64decode(event.delta)
                        await self._on_audio_delta(
                            audio_bytes,
                            item_id,
                            content_index,
                        )

                elif event_type == "response.output_audio_transcript.delta":
                    item_id = getattr(event, "item_id", "")
                    if item_id in self._ignored_output_item_ids:
                        continue

                    content_index = getattr(event, "content_index", 0)
                    key = (item_id, content_index)
                    assistant_transcripts[key] = assistant_transcripts.get(key, "") + (
                        event.delta or ""
                    )

                    caption = assistant_transcripts[key].strip()
                    if caption and self._on_caption is not None:
                        await self._on_caption(
                            "assistant",
                            caption,
                            item_id,
                            "partial",
                        )

                elif event_type == "response.output_audio_transcript.done":
                    item_id = getattr(event, "item_id", "")
                    content_index = getattr(event, "content_index", 0)
                    key = (item_id, content_index)

                    if item_id in self._ignored_output_item_ids:
                        assistant_transcripts.pop(key, None)
                        continue

                    partial_transcript = assistant_transcripts.pop(key, "")
                    transcript = (
                        (event.transcript or partial_transcript) or ""
                    ).strip()
                    if transcript:
                        if self._on_caption is not None:
                            await self._on_caption(
                                "assistant",
                                transcript,
                                item_id,
                                "final",
                            )
                        self._transcript.append(
                            {"role": "assistant", "content": transcript}
                        )
                        if self._on_agent_transcript is not None:
                            await self._on_agent_transcript(transcript, item_id)

                    if self._current_output_item_id == item_id:
                        self._current_output_item_id = None
                        self._current_output_content_index = 0

                elif event_type == "conversation.item.truncated":
                    if self._on_truncated is not None:
                        await self._on_truncated(
                            getattr(event, "item_id", ""),
                            getattr(event, "content_index", 0),
                            getattr(event, "audio_end_ms", 0),
                        )

                elif event_type == "response.done":
                    self._current_output_item_id = None
                    self._current_output_content_index = 0

                elif event_type == "error":
                    error = getattr(event, "error", None)
                    message = getattr(error, "message", None) or str(error or event)
                    logger.error("realtime session error: %s", message)
                    if self._on_error is not None:
                        await self._on_error(message)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("realtime session: event loop crashed")
            if self._on_error is not None:
                await self._on_error("Realtime session event loop crashed")
        finally:
            self._running = False

    async def _load_semantic_facts(self) -> list[str]:
        """Load compact semantic facts for the prompt."""

        if self._memory_mode == MemoryMode.INCOGNITO:
            return []

        namespace = (self._user_id, "semantic")
        try:
            records = await self._memory_store.asearch(
                namespace,
                query=None,
                limit=_MAX_MEMORY_ITEMS,
            )
        except Exception:
            logger.warning(
                "realtime session: failed to load semantic facts",
                exc_info=True,
            )
            return []

        facts: list[str] = []
        for record in records:
            evidence = record.value.get("evidence_quote", "")
            if evidence:
                facts.append(f"Previously noted: {evidence}")
        return facts

    async def _load_episodic_arcs(self) -> list[str]:
        """Load compact episodic summaries for the prompt."""

        if self._memory_mode == MemoryMode.INCOGNITO:
            return []

        namespace = (self._user_id, "episodic")
        try:
            records = await self._memory_store.asearch(namespace, query=None, limit=3)
        except Exception:
            logger.warning(
                "realtime session: failed to load episodic arcs",
                exc_info=True,
            )
            return []

        arcs: list[str] = []
        for record in records:
            summary = record.value.get("summary", "")
            if summary:
                themes = record.value.get("primary_themes", [])
                themes_text = ", ".join(themes) if themes else "untagged"
                arcs.append(f"Past session ({themes_text}): {summary}")
        return arcs

    async def _load_procedural_rules(self) -> list[str]:
        """Load compact procedural user rules for the prompt."""

        if self._memory_mode == MemoryMode.INCOGNITO:
            return []

        try:
            profile = await aget_procedural_profile(
                self._memory_store,
                user_id=self._user_id,
            )
        except Exception:
            logger.warning(
                "realtime session: failed to load procedural rules",
                exc_info=True,
            )
            return []

        return [rule.rule for rule in profile.rules[:_MAX_MEMORY_ITEMS]]
