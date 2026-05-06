"""LiveKit agentic voice backend for OpenCouch (Option C).

Runs as a standalone worker process::

    # Development mode (auto-reload)
    uv run python -m agent.voice.agent dev

    # Production mode
    uv run python -m agent.voice.agent start

    # Interactive console (no room needed)
    uv run python -m agent.voice.agent console

Environment variables:
    LIVEKIT_URL          — LiveKit server URL
    LIVEKIT_API_KEY      — LiveKit API key
    LIVEKIT_API_SECRET   — LiveKit API secret
    OPENAI_API_KEY       — OpenAI API key (for RealtimeModel)
    OPENCOUCH_MEMORY_MODE — "persistent" (default) or "guest"
    OPENCOUCH_VOICE_USER_ID — optional local console/dev owner override
    OPENCOUCH_VOICE_THREAD_ID — optional local console/dev thread override
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Literal

from dotenv import load_dotenv
from openai.types import realtime as openai_realtime_types

from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    RunContext,
    TurnHandlingOptions,
    function_tool,
    room_io,
)
from livekit.plugins import openai, silero

from agent.memory.hashing import iso_now
from agent.memory.modes import MemoryMode
from agent.memory.procedural_profile import aget_procedural_profile
from agent.memory.reconciliation import filter_active_semantic_records
from agent.memory.store import MemoryStore
from agent.persistence import (
    DEFAULT_CRISIS_LOG_DB_PATH,
    DEFAULT_MEMORY_DB_PATH,
    DEFAULT_THREAD_DB_PATH,
    PersistentAgentRuntime,
)
from agent.voice.activity import emit_voice_activity
from agent.voice.finalization_status import set_voice_finalization_status
from agent.voice.session_data import (
    ProcessStage,
    SessionData,
    SessionIntent,
    TherapeuticApproach,
    TherapeuticFormulation,
    TherapeuticProcessState,
    parse_optional_voice_memory_mode,
    parse_voice_memory_mode,
)
from agent.voice.tasks import GroundingTask
from agent.voice.tools import (
    answer_grounded_factual_lookup,
    cancel_memory_deletion,
    confirm_memory_deletion,
    crisis_check,
    matches_crisis_keywords,
    prepare_indexed_memory_deletion,
    prepare_memory_deletion,
    provide_crisis_resources,
    save_insight,
    select_memory_deletion_candidate,
    set_proactive_memory_recall,
    show_memory_status,
    show_saved_memory,
)
from agent.voice.config import (
    DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE,
    DEFAULT_REALTIME_TRANSCRIPTION_PROMPT,
    build_voice_system_prompt,
    normalize_assistant_voice,
)

load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger(__name__)

_VOICE_USER_ID_ENV = "OPENCOUCH_VOICE_USER_ID"
_VOICE_THREAD_ID_ENV = "OPENCOUCH_VOICE_THREAD_ID"
_VOICE_OUTPUT_WARMUP_TOPIC = "opencouch.voice_output_warmup"
# ── Runtime singleton ───────────────────────────────────────────────
# Initialized once per worker process in the prewarm function.
# Shared across all voice sessions in this worker.

_runtime: PersistentAgentRuntime | None = None
_llm_client = None  # BaseLLMClient | None — for memory extraction


def _prewarm_process(proc: agents.JobProcess) -> None:
    """Preload blocking voice assets for a LiveKit worker process.

    Loads the Silero VAD weights and warms the OpenCouch runtime
    (SQLite connections, embedding provider, control LLM client) so
    the first session on this worker does not pay the load cost.

    Args:
        proc (agents.JobProcess): LiveKit job process shared state.

    Returns:
        None.
    """

    proc.userdata["vad"] = silero.VAD.load()
    logger.info("livekit agent: prewarmed Silero VAD")

    try:
        asyncio.run(_ensure_runtime())
        logger.info("livekit agent: prewarmed runtime + LLM client")
    except Exception:
        # Lazy init in _ensure_runtime() is the safety net — worker can
        # still serve sessions, just with cold-start cost on the first one.
        logger.warning(
            "livekit agent: runtime prewarm failed; will lazy-init on first session",
            exc_info=True,
        )


async def _ensure_runtime() -> PersistentAgentRuntime:
    """Lazily initialize the runtime. Called once per worker process."""
    global _runtime, _llm_client  # noqa: PLW0603

    if _runtime is not None:
        return _runtime

    from core.config import get_settings

    settings = get_settings()
    _runtime = PersistentAgentRuntime(
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
    await _runtime.__aenter__()

    from core.config import create_configured_control_llm_client

    try:
        _llm_client = create_configured_control_llm_client()
    except Exception:
        logger.warning("livekit agent: no LLM client configured for extraction")
        _llm_client = None

    logger.info(
        "livekit agent: runtime initialized (session_default_mode=%s)",
        parse_voice_memory_mode(os.getenv("OPENCOUCH_MEMORY_MODE")).value,
    )
    return _runtime


async def _close_runtime() -> None:
    """Close the shared runtime if it is open.

    Returns:
        None: Closes the runtime and clears the cached globals.
    """

    global _runtime, _llm_client  # noqa: PLW0603

    runtime = _runtime
    if runtime is None:
        return

    _runtime = None
    _llm_client = None
    await runtime.__aexit__(None, None, None)


def _parse_voice_session_metadata(
    metadata: str | None,
) -> tuple[str | None, str | None, str | None, str | None, MemoryMode | None]:
    """Parse voice session fields from a LiveKit metadata payload.

    Args:
        metadata (str | None): Raw JSON metadata from a job or participant.

    Returns:
        tuple[str | None, str | None, str | None, str | None, MemoryMode | None]:
            Parsed ``(user_id, thread_id, transcription_language,
            assistant_voice, memory_mode)`` values when present. An explicit
            empty language string means auto-detect.
    """
    if not metadata:
        return None, None, None, None, None

    try:
        payload = json.loads(metadata)
    except (json.JSONDecodeError, TypeError):
        return None, None, None, None, None

    user_id = payload.get("user_id")
    thread_id = payload.get("thread_id")
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


def _resolve_livekit_session_metadata(
    *,
    job_metadata: str | None,
    participant_metadata: str | None,
) -> tuple[str, str, str | None, str, MemoryMode]:
    """Resolve the effective owner/session metadata for a voice run.

    Args:
        job_metadata (str | None): Metadata attached to the LiveKit job dispatch.
        participant_metadata (str | None): Metadata attached to the connected participant.

    Returns:
        tuple[str, str, str | None, str, MemoryMode]: Effective
            ``(user_id, thread_id, transcription_language, assistant_voice, memory_mode)``
            for the session.
    """
    user_id = os.getenv(_VOICE_USER_ID_ENV, "").strip() or "voice-user"
    thread_id = (
        os.getenv(_VOICE_THREAD_ID_ENV, "").strip() or f"voice-{uuid.uuid4().hex[:12]}"
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
        ) = _parse_voice_session_metadata(metadata)
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

    return user_id, thread_id, transcription_language, assistant_voice, memory_mode


# ── Memory loading helpers ──────────────────────────────────────────
# Compact memory loaders for the voice system prompt.  These return
# string lists suitable for build_voice_system_prompt().

_MAX_MEMORY_ITEMS = 6
_MID_SESSION_MEMORY_ITEMS = 3
_EXERCISE_KEYWORD_PATTERN = (
    r"breath(?:ing)?|ground(?:ing)?|body scan|relaxation|muscle relaxation|"
    r"progressive muscle relaxation|box breathing|stop technique|exercise|"
    r"technique"
)
_EXPLICIT_EXERCISE_REQUEST_PATTERNS: tuple[str, ...] = (
    rf"\b(guide|walk)\s+me\s+through\b.*\b({_EXERCISE_KEYWORD_PATTERN})\b",
    rf"\b(can|could|would)\s+you\s+(guide|walk)\s+me\b.*\b({_EXERCISE_KEYWORD_PATTERN})\b",
    rf"\b(can|could|would)\s+we\s+do\b.*\b({_EXERCISE_KEYWORD_PATTERN})\b",
    rf"\blet'?s\s+do\b.*\b({_EXERCISE_KEYWORD_PATTERN})\b",
    rf"\bi\s+(want|need|would like)\b.*\b({_EXERCISE_KEYWORD_PATTERN})\b",
    r"\b(exercises?|techniques?)\b.*\b(cope|manage|relax|release|tension)\b",
    r"\b(help|teach)\s+me\b.*\b(breathe|breathing|ground|grounding|calm down)\b",
    r"\bground me\b",
)
_EXERCISE_AGREEMENT_PATTERNS: tuple[str, ...] = (
    r"^\s*(yes|yeah|yep|ok|okay|sure|yes please)\b",
    r"^\s*let'?s try\b",
    r"^\s*that would help\b",
    r"^\s*i'?m open to that\b",
    r"^\s*okay[, ]+let'?s do (it|that)\b",
    r"^\s*(maybe\s+)?(a|the)?\s*(muscle\s+)?relaxation\s+(technique|exercise)\??\s*$",
)
_EXERCISE_OFFER_PATTERNS: tuple[str, ...] = (
    r"\bwould (you )?like\b.*\b(exercise|breathing|grounding|relaxation|technique)\b",
    r"\bwould it help\b.*\b(exercise|breathing|grounding|relaxation|technique)\b",
    r"\bwant to try\b.*\b(exercise|breathing|grounding|relaxation|technique)\b",
    r"\bwe can try\b.*\b(exercise|breathing|grounding|relaxation|technique)\b",
    r"\bi can (guide|walk) you through\b.*\b(exercise|breathing|grounding|relaxation|technique)\b",
    r"\bwant me to\b.*\b(guide|walk)\b.*\b(exercise|breathing|grounding|relaxation|technique)\b",
    r"\bready to try\b.*\b(exercise|breathing|grounding|relaxation|technique)\b",
)
_GENERAL_GUIDANCE_REQUEST_PATTERNS: tuple[str, ...] = (
    r"\b(can|could|would) you help me\b",
    r"\bcan you help me figure\b",
    r"\bhelp me figure out\b",
    r"\bguide me\b",
    r"\bwalk me through\b",
    r"\bwhat should i do\b",
    r"\bwhat can i do\b",
    r"\bhow do i\b",
    r"\bwhere do i start\b",
)
_GENERAL_GUIDANCE_AGREEMENT_PATTERNS: tuple[str, ...] = (
    r"^\s*(yes|yeah|yep|ok|okay|sure|yes please)\b",
    r"^\s*let'?s do that\b",
    r"^\s*that would help\b",
    r"^\s*i'?m open to that\b",
    r"^\s*please do\b",
)
_GENERAL_GUIDANCE_OFFER_PATTERNS: tuple[str, ...] = (
    r"\bwould it help if i\b",
    r"\bdo you want me to\b",
    r"\bi have a thought that may help\b",
    r"\bwe can look at this together\b",
    r"\bwant to try\b",
)
_GUIDANCE_WITHDRAWAL_PATTERNS: tuple[str, ...] = (
    r"\bjust listen\b",
    r"\bi just need you to listen\b",
    r"\bi don'?t want advice\b",
    r"\bno advice\b",
    r"\bnot looking for (advice|solutions|tips)\b",
    r"\bcan we just talk\b",
)
_UNDERSTANDING_PATTERNS: tuple[str, ...] = (
    r"\bwhy do i\b",
    r"\bwhy am i\b",
    r"\bwhy does this\b",
    r"\bi don'?t understand\b",
    r"\bmake sense of\b",
    r"\buntangle\b",
    r"\bwhat'?s happening\b",
    r"\bis it normal\b",
)
_PATTERN_PATTERNS: tuple[str, ...] = (
    r"\bi keep\b",
    r"\bi always\b",
    r"\bevery time\b",
    r"\bsame thing\b",
    r"\bpattern\b",
    r"\bcycle\b",
    r"\bwhy does this keep\b",
)
_ACTION_PATTERNS: tuple[str, ...] = (
    r"\bwhat should i do\b",
    r"\bwhat do i do\b",
    r"\bwhat can i do\b",
    r"\bnext step\b",
    r"\bhow do i handle\b",
)
_SHIFT_PATTERNS: tuple[str, ...] = (
    r"\bmaybe\b",
    r"\bi guess\b",
    r"\bnot always\b",
    r"\bbut also\b",
    r"\bon the other hand\b",
    r"\bthat doesn'?t fully fit\b",
)
_CBT_PATTERNS: tuple[str, ...] = (
    r"\bthought\b",
    r"\bbelief\b",
    r"\bevidence\b",
    r"\bcatastroph",
    r"\bworst[- ]case\b",
    r"\bprediction\b",
)
_ACT_PATTERNS: tuple[str, ...] = (
    r"\bfighting this feeling\b",
    r"\bmake it go away\b",
    r"\bstep back from this thought\b",
    r"\bavoiding because of\b",
    r"\bvalues\b",
)
_GRIEF_PATTERNS: tuple[str, ...] = (
    r"\bgrief\b",
    r"\bgrieving\b",
    r"\bmiss (him|her|them)\b",
    r"\blost my\b",
    r"\bdeath\b",
    r"\bfuneral\b",
)
_INTERPERSONAL_PATTERNS: tuple[str, ...] = (
    r"\bmy (mom|mother|dad|father|partner|boyfriend|girlfriend|wife|husband|friend|roommate|boss|coworker)\b",
    r"\bargument\b",
    r"\bboundary\b",
    r"\brelationship\b",
    r"\blonely\b",
)
_PFA_PATTERNS: tuple[str, ...] = (
    r"\boverwhelm",
    r"\bcan'?t calm down\b",
    r"\bflooded\b",
    r"\bpanick",
    r"\bspiral",
    r"\btoo much right now\b",
)
_EMOTION_MARKERS: tuple[tuple[str, str], ...] = (
    ("overwhelmed", r"\boverwhelm"),
    ("anxious", r"\banxious|panic|panick"),
    ("sad", r"\bsad|cry|crying"),
    ("angry", r"\bangry|furious|resent"),
    ("ashamed", r"\basham|embarrass"),
    ("guilty", r"\bguilt|guilty"),
    ("grieving", r"\bgrief|grieving|miss (him|her|them)|lost my"),
    ("lonely", r"\blonely|alone"),
    ("numb", r"\bnumb|empty"),
)
_HOT_THOUGHT_PATTERNS: tuple[str, ...] = (
    r"\b(i am|i'm) [^.?!]{0,80}",
    r"\b(i always|i never|no one|everyone) [^.?!]{0,80}",
    r"\bwhat if [^.?!]{0,80}",
    r"\bthat means [^.?!]{0,80}",
)
_TARGET_MAX_CHARS = 140
_PERSISTED_THERAPEUTIC_SYSTEM_SOURCES = {
    "therapeutic_process_controller",
    "semantic_memory_injection",
}

TherapeuticAgentKind = Literal[
    "hold_space",
    "reflective",
    "understanding",
    "technique",
]

_THERAPEUTIC_AGENT_SPECIALIZATION_BLOCKS: dict[TherapeuticAgentKind, str] = {
    "hold_space": (
        "Voice posture: stay close.\n"
        "- Let the user feel heard before moving anywhere.\n"
        "- Reflect the specific weight of what they said without turning it into a plan too quickly.\n"
        "- Do not rush into advice or structured technique, but do not simply echo the user.\n"
        "- When there is enough context, offer one gentle direction or question, not a plan.\n"
        "- If the user has not invited active guidance, keep the help collaborative and low-pressure."
    ),
    "reflective": (
        "Voice posture: notice gently.\n"
        "- Name one recurring theme or pattern at a time.\n"
        "- Keep observations tentative: say what you're noticing, not what is definitively true.\n"
        "- Stay in the user's language instead of clinical labels.\n"
        "- After naming the pattern, let the user correct it or say where it begins."
    ),
    "understanding": (
        "Voice posture: help it make sense.\n"
        "- Help the user stay with one concrete moment, feeling, thought, or stuck point.\n"
        "- Untangle the experience one step at a time rather than giving a broad explanation.\n"
        "- Offer one concise hypothesis when the pattern is visible, then invite the user to correct it.\n"
        "- Use one focused question at a time, and make it feel like curiosity rather than assessment."
    ),
    "technique": (
        "Voice posture: help practically without becoming mechanical.\n"
        "- Carry the thread forward rather than resetting broadly.\n"
        "- Briefly acknowledge what is happening, then offer one concrete next step, reframe, or experiment.\n"
        "- Keep structure conversational and immediately usable.\n"
        "- Prefer conversational micro-steps before formal exercises unless the user explicitly asks for an exercise.\n"
        "- Once the user asks for an exercise or says yes to one you offered, do not ask for readiness again; start the first step.\n"
        "- Do not overwhelm the user with lists, lectures, or multiple options."
    ),
}


def _should_finalize_transcript_on_shutdown(*, is_fake_job: bool) -> bool:
    """Return whether shutdown should run transcript finalization.

    Args:
        is_fake_job (bool): Whether the session is the local console fake job.

    Returns:
        bool: ``True`` when shutdown should run transcript finalization.
    """

    if not is_fake_job:
        return True

    raw = os.getenv("OPENCOUCH_VOICE_CONSOLE_FINALIZE_ON_EXIT", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_realtime_model(
    *,
    transcription_language: str | None = DEFAULT_REALTIME_TRANSCRIPTION_LANGUAGE,
    assistant_voice: str = "marin",
) -> openai.realtime.RealtimeModel:
    """Build the realtime model for agent-controlled turn routing.

    Args:
        transcription_language (str | None): Preferred transcription
            language, or ``None`` to let the model auto-detect.
        assistant_voice (str): Realtime output voice name.

    Returns:
        openai.realtime.RealtimeModel: Realtime model with provider-side
            turn detection disabled so the session can own turn
            completion and run ``on_user_turn_completed()`` first.
    """

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


def _build_turn_handling() -> TurnHandlingOptions:
    """Use session-owned VAD turn detection for spoken-turn control.

    Returns:
        TurnHandlingOptions: Turn handling that routes spoken turns
            through ``on_user_turn_completed()`` before replying.
    """

    return TurnHandlingOptions(
        turn_detection="vad",
    )


def _chat_item_role(item: object) -> str:
    """Return a normalized role string for a LiveKit chat item.

    Args:
        item: LiveKit chat-context item.

    Returns:
        Normalized role text, or an empty string when the item has no role.
    """

    role = getattr(item, "role", "")
    return str(getattr(role, "value", role))


def _is_non_empty_dialogue_message(item: object) -> bool:
    """Return whether a chat item is user/assistant dialogue.

    Args:
        item: LiveKit chat-context item.

    Returns:
        ``True`` for non-empty user or assistant message items.
    """

    if getattr(item, "type", None) != "message":
        return False
    if _chat_item_role(item) not in {"user", "assistant"}:
        return False
    return bool((getattr(item, "text_content", None) or "").strip())


def _copy_handoff_chat_ctx(chat_ctx: ChatContext | None) -> ChatContext | None:
    """Carry only dialogue across crisis/task handoff boundaries.

    Args:
        chat_ctx: Current context to hand off.

    Returns:
        Filtered context containing user/assistant dialogue only.
    """

    if chat_ctx is None:
        return None

    return ChatContext(
        [item for item in chat_ctx.items if _is_non_empty_dialogue_message(item)]
    )


def _serialize_session_history(chat_ctx: ChatContext) -> list[dict[str, str]]:
    """Convert session history into the persisted transcript format."""
    transcript: list[dict[str, str]] = []
    for item in chat_ctx.items:
        if item.type != "message":
            continue

        role = getattr(item.role, "value", item.role)
        if role not in {"user", "assistant"}:
            continue

        text = (item.text_content or "").strip()
        if not text:
            continue

        transcript.append({"role": role, "content": text})

    return transcript


def _clip_text(text: str, limit: int = _TARGET_MAX_CHARS) -> str:
    """Return whitespace-normalized text clipped to a stable length.

    Args:
        text (str): Raw input text.
        limit (int): Maximum character length to keep.

    Returns:
        str: Normalized and clipped text.
    """

    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _previous_assistant_turn(chat_ctx: ChatContext | None) -> str:
    """Return the most recent assistant message from the chat context.

    Args:
        chat_ctx (ChatContext | None): Conversation context to inspect.

    Returns:
        str: Most recent assistant text, or an empty string when absent.
    """

    if chat_ctx is None:
        return ""

    for item in reversed(chat_ctx.items):
        if item.type != "message":
            continue

        role = getattr(item.role, "value", item.role)
        text = (item.text_content or "").strip()
        if role == "assistant" and text:
            return text

    return ""


def _latest_user_and_previous_assistant_turn(
    chat_ctx: ChatContext | None,
) -> tuple[str, str]:
    """Return the latest user turn and the assistant turn before it.

    Args:
        chat_ctx (ChatContext | None): Conversation context for the
            current LiveKit session.

    Returns:
        tuple[str, str]: Latest user text and the previous assistant
            text. Empty strings are returned when either is missing.
    """

    if chat_ctx is None:
        return "", ""

    latest_user = ""
    previous_assistant = ""
    for item in reversed(chat_ctx.items):
        if item.type != "message":
            continue

        role = getattr(item.role, "value", item.role)
        text = (item.text_content or "").strip()
        if not text:
            continue

        if not latest_user and role == "user":
            latest_user = text
            continue

        if latest_user and role == "assistant":
            previous_assistant = text
            break

    return latest_user, previous_assistant


def _matches_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    """Return whether the text matches any regex in the pattern set.

    Args:
        text (str): Input text to classify.
        patterns (tuple[str, ...]): Regex patterns to test.

    Returns:
        bool: ``True`` when any pattern matches.
    """

    lowered = text.lower()
    return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)


def _has_exercise_consent(chat_ctx: ChatContext | None) -> bool:
    """Return whether the latest turn clearly permits a structured exercise.

    Args:
        chat_ctx (ChatContext | None): Current conversation context.

    Returns:
        bool: ``True`` when the latest user turn explicitly requests an
            exercise or clearly agrees to an offered one.
    """

    user_text, assistant_text = _latest_user_and_previous_assistant_turn(chat_ctx)
    if not user_text:
        return False

    if _matches_any_pattern(user_text, _EXPLICIT_EXERCISE_REQUEST_PATTERNS):
        return True

    return _matches_any_pattern(
        user_text, _EXERCISE_AGREEMENT_PATTERNS
    ) and _matches_any_pattern(assistant_text, _EXERCISE_OFFER_PATTERNS)


def _has_general_guidance_permission(
    user_text: str,
    assistant_text: str,
) -> bool:
    """Return whether the user has clearly invited more active guidance.

    Args:
        user_text (str): Latest user message text.
        assistant_text (str): Previous assistant message text.

    Returns:
        bool: ``True`` when the user explicitly asked for guidance or
            clearly agreed after the assistant offered it.
    """

    return _matches_any_pattern(user_text, _GENERAL_GUIDANCE_REQUEST_PATTERNS) or (
        _matches_any_pattern(user_text, _GENERAL_GUIDANCE_AGREEMENT_PATTERNS)
        and _matches_any_pattern(assistant_text, _GENERAL_GUIDANCE_OFFER_PATTERNS)
    )


def _has_guidance_withdrawal(user_text: str) -> bool:
    """Return whether the user is asking for listening instead of guidance.

    Args:
        user_text (str): Latest user message text.

    Returns:
        bool: ``True`` when prior permission for active guidance should
            be paused.
    """

    return _matches_any_pattern(user_text, _GUIDANCE_WITHDRAWAL_PATTERNS)


def _has_enough_context_for_orientation(user_text: str) -> bool:
    """Return whether a venting turn has enough detail to gently orient.

    Args:
        user_text (str): Latest user message text.

    Returns:
        bool: ``True`` when the assistant can move beyond pure holding
            without becoming advice-heavy.
    """

    normalized = " ".join(user_text.split())
    if len(normalized) >= 120:
        return True

    lowered = normalized.lower()
    return bool(
        re.search(
            r"\b(because|for so long|every day|at work|with my|my boss|my partner|my family|i keep|i always|i never)\b",
            lowered,
            flags=re.IGNORECASE,
        )
    )


def _infer_primary_emotion(user_text: str, previous: str) -> str:
    """Infer the primary emotion signal from the latest user turn.

    Args:
        user_text (str): Latest user message text.
        previous (str): Previously inferred emotion.

    Returns:
        str: Emotion label or the prior value when no new signal exists.
    """

    lowered = user_text.lower()
    for label, pattern in _EMOTION_MARKERS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return label
    return previous


def _infer_hot_thought(user_text: str, previous: str) -> str:
    """Infer a hot thought or belief statement from the latest user turn.

    Args:
        user_text (str): Latest user message text.
        previous (str): Previously inferred thought.

    Returns:
        str: Short thought text when one is detected.
    """

    for pattern in _HOT_THOUGHT_PATTERNS:
        match = re.search(pattern, user_text, flags=re.IGNORECASE)
        if match:
            return _clip_text(match.group(0), limit=100)
    return previous


def _infer_pattern_summary(user_text: str, previous: str) -> str:
    """Infer whether the user is naming a recurring pattern.

    Args:
        user_text (str): Latest user message text.
        previous (str): Previously inferred pattern text.

    Returns:
        str: Short pattern summary or the previous value.
    """

    if _matches_any_pattern(user_text, _PATTERN_PATTERNS):
        return _clip_text(user_text, limit=100)
    return previous


def _infer_session_intent(user_text: str, permission_granted: bool) -> SessionIntent:
    """Infer the user's current session intent from the latest turn.

    Args:
        user_text (str): Latest user message text.
        permission_granted (bool): Whether the user invited active guidance.

    Returns:
        SessionIntent: High-level therapeutic intent for this turn.
    """

    lowered = user_text.lower()
    if re.search(
        r"\b(i should go|i have to go|need to go|talk later|goodnight|bye)\b", lowered
    ):
        return "close"
    if _matches_any_pattern(lowered, _EXPLICIT_EXERCISE_REQUEST_PATTERNS):
        return "regulate"
    if _matches_any_pattern(lowered, _UNDERSTANDING_PATTERNS):
        return "understand"
    if permission_granted or _matches_any_pattern(lowered, _ACTION_PATTERNS):
        return "work"
    if _matches_any_pattern(lowered, _PATTERN_PATTERNS):
        return "reflect"
    if _matches_any_pattern(lowered, _PFA_PATTERNS):
        return "regulate"
    return "vent"


def _infer_therapeutic_approach(user_text: str) -> TherapeuticApproach:
    """Infer the most relevant therapeutic approach for the current turn.

    Args:
        user_text (str): Latest user message text.

    Returns:
        TherapeuticApproach: Best-fit therapeutic approach label.
    """

    lowered = user_text.lower()
    if _matches_any_pattern(lowered, _GRIEF_PATTERNS):
        return "grief_support"
    if _matches_any_pattern(lowered, _PFA_PATTERNS):
        return "pfa"
    if _matches_any_pattern(lowered, _INTERPERSONAL_PATTERNS):
        return "interpersonal_therapy"
    if _matches_any_pattern(lowered, _ACT_PATTERNS):
        return "act"
    if _matches_any_pattern(lowered, _CBT_PATTERNS):
        return "cbt"
    return "motivational_interviewing"


def _infer_process_stage(
    *,
    intent: SessionIntent,
    user_text: str,
    previous: TherapeuticProcessState,
    hot_thought: str,
    permission_granted: bool,
) -> ProcessStage:
    """Infer the therapeutic process stage for the next reply.

    Args:
        intent (SessionIntent): Current high-level user intent.
        user_text (str): Latest user message text.
        previous (TherapeuticProcessState): Prior controller state.
        hot_thought (str): Current hot-thought summary.
        permission_granted (bool): Whether the user invited more active
            guidance on this turn.

    Returns:
        ProcessStage: Process stage to emphasize on the next turn.
    """

    lowered = user_text.lower()

    if intent == "vent":
        if _has_enough_context_for_orientation(user_text):
            if previous.formulation.pattern or _matches_any_pattern(
                lowered, _PATTERN_PATTERNS
            ):
                return "identify"
            return "orient"
        return "hold"
    if intent == "regulate":
        return "ground" if permission_granted else "hold"
    if intent == "close":
        return "ground"
    if _matches_any_pattern(lowered, _ACTION_PATTERNS):
        return "ground"
    if previous.process_stage == "examine" and _matches_any_pattern(
        lowered, _SHIFT_PATTERNS
    ):
        return "shift"
    if intent == "work":
        if hot_thought:
            return "examine"
        if _matches_any_pattern(lowered, _PATTERN_PATTERNS):
            return "identify"
        return "orient"
    if intent in {"understand", "reflect"}:
        if hot_thought or _matches_any_pattern(lowered, _PATTERN_PATTERNS):
            return "identify"
        return "orient"
    return "hold"


def _assess_therapeutic_process_state(
    *,
    turn_ctx: ChatContext | None,
    user_text: str,
    previous: TherapeuticProcessState,
) -> TherapeuticProcessState:
    """Assess the therapeutic process state for the next voice turn.

    Args:
        turn_ctx (ChatContext | None): Conversation context before the
            latest user turn is added.
        user_text (str): Latest user message text.
        previous (TherapeuticProcessState): Previously stored controller
            state for the session.

    Returns:
        TherapeuticProcessState: Updated session-scoped controller state.
    """

    assistant_text = _previous_assistant_turn(turn_ctx)
    permission_requested = _has_general_guidance_permission(user_text, assistant_text)
    permission_withdrawn = _has_guidance_withdrawal(user_text)
    permission_granted = permission_requested or (
        previous.guidance_permission == "granted" and not permission_withdrawn
    )
    guidance_permission = "granted" if permission_granted else "not_yet"
    intent = _infer_session_intent(user_text, permission_granted)
    hot_thought = _infer_hot_thought(user_text, previous.formulation.hot_thought)
    process_stage = _infer_process_stage(
        intent=intent,
        user_text=user_text,
        previous=previous,
        hot_thought=hot_thought,
        permission_granted=permission_granted,
    )

    if intent == "vent":
        user_goal = "feel heard before moving into problem-solving"
    elif intent == "understand":
        user_goal = "make sense of what is happening"
    elif intent == "reflect":
        user_goal = "notice and name the recurring pattern"
    elif intent == "work":
        user_goal = "work through the issue more actively"
    elif intent == "regulate":
        user_goal = "settle the immediate overwhelm"
    else:
        user_goal = "wrap up cleanly"

    formulation = TherapeuticFormulation(
        situation=_clip_text(user_text),
        primary_emotion=_infer_primary_emotion(
            user_text,
            previous.formulation.primary_emotion,
        ),
        hot_thought=hot_thought,
        pattern=_infer_pattern_summary(user_text, previous.formulation.pattern),
        user_goal=user_goal,
    )

    return TherapeuticProcessState(
        session_intent=intent,
        guidance_permission=guidance_permission,
        process_stage=process_stage,
        therapeutic_approach=_infer_therapeutic_approach(user_text),
        active_target=_clip_text(user_text),
        formulation=formulation,
    )


def _build_therapeutic_process_guidance(
    state: TherapeuticProcessState,
) -> str:
    """Build a compact controller message for the next reply.

    Args:
        state (TherapeuticProcessState): Current therapeutic process
            state for the session.

    Returns:
        str: Short system guidance for the next reply.
    """

    stage_guidance: dict[ProcessStage, str] = {
        "hold": (
            "Stay close without becoming passive. Reflect specifically, then "
            "offer one gentle direction, question, or possibility only if the "
            "user has given enough context."
        ),
        "orient": (
            "Stay with what feels most present. If the user seems scattered, "
            "gently help them choose one place to start."
        ),
        "identify": (
            "Help the user notice the main feeling, thought, or recurring "
            "pattern underneath the distress. Keep it tentative and let them "
            "correct it."
        ),
        "examine": (
            "Stay with one thought or belief and help the user look at it from "
            "another angle without sounding like a worksheet."
        ),
        "shift": (
            "Reflect the nuance the user is already finding. Help them put the "
            "fuller picture in their own words, without forcing a reframe."
        ),
        "ground": (
            "Keep the next move very small and concrete. Offer one doable move, "
            "not a list."
        ),
    }

    permission_line = (
        "The user has given permission for more active guidance."
        if state.guidance_permission == "granted"
        else "The user has not yet invited directive guidance. Stay collaborative: you may orient, formulate, or ask one focused question, but ask permission before prescribing a technique."
    )

    approach_line = {
        "cbt": "Use CBT as quiet scaffolding. Stay conversational: notice the situation, the thought, and one possible way to look at it without sounding like a worksheet.",
        "act": "Use ACT as quiet scaffolding: make space for the experience, reduce struggle, and reconnect to one workable move.",
        "interpersonal_therapy": "Stay with the relationship dynamic and the user's position inside it, without over-mapping it.",
        "grief_support": "Lead with companionship and listening before interpretation.",
        "pfa": "Prioritize stabilization and practical steadiness over deeper exploration.",
    }.get(
        state.therapeutic_approach,
        "Use reflective listening, autonomy support, and at most one focused question.",
    )

    lines = [
        "Therapeutic controller state for this turn:",
        f"- intent: {state.session_intent}",
        f"- process_stage: {state.process_stage}",
        f"- guidance_permission: {state.guidance_permission}",
        f"- approach: {state.therapeutic_approach}",
    ]
    if state.formulation.primary_emotion:
        lines.append(f"- primary_emotion: {state.formulation.primary_emotion}")
    if state.formulation.hot_thought:
        lines.append(f"- hot_thought: {state.formulation.hot_thought}")
    if state.formulation.pattern:
        lines.append(f"- pattern: {state.formulation.pattern}")
    if state.formulation.user_goal:
        lines.append(f"- user_goal: {state.formulation.user_goal}")

    lines.extend(
        [
            "",
            stage_guidance[state.process_stage],
            permission_line,
            approach_line,
        ]
    )
    return "\n".join(lines)


def _therapeutic_agent_kind_for_state(
    state: TherapeuticProcessState,
) -> TherapeuticAgentKind:
    """Pick the specialized therapeutic agent for the next reply.

    Args:
        state (TherapeuticProcessState): Session-scoped controller state.

    Returns:
        TherapeuticAgentKind: Specialized agent kind best suited for the
            next reply.
    """

    if state.process_stage == "hold":
        return "hold_space"

    if (
        state.process_stage in {"identify", "shift"}
        or state.session_intent == "reflect"
    ):
        return "reflective"

    if (
        state.process_stage in {"orient", "examine"}
        or state.session_intent == "understand"
    ):
        return "understanding"

    if state.process_stage == "ground":
        if state.guidance_permission == "granted" or state.session_intent in {
            "work",
            "regulate",
        }:
            return "technique"
        return "hold_space"

    if state.session_intent == "work":
        return "technique"

    return "hold_space"


def _compose_therapeutic_agent_instructions(
    *,
    base_instructions: str,
    agent_kind: TherapeuticAgentKind,
) -> str:
    """Compose the base prompt with one specialization block.

    Args:
        base_instructions (str): Session-level therapeutic instructions.
        agent_kind (TherapeuticAgentKind): Specialized voice agent kind.

    Returns:
        str: Full instruction string for the active therapeutic agent.
    """

    return (
        f"{base_instructions}\n\n"
        "Memory control:\n"
        "- If the user asks what you remember or what is saved, call show_saved_memory.\n"
        "- If the user asks for memory status, call show_memory_status.\n"
        "- If the user asks you not to bring up past sessions or old memories, call set_proactive_memory_recall with enabled=false.\n"
        "- If the user says you may bring up past sessions when relevant, call set_proactive_memory_recall with enabled=true.\n"
        "- If the user asks you to forget or delete saved memory, first call a preparation tool. Do not delete anything until the user clearly confirms, then call confirm_memory_deletion.\n"
        "- If the user declines a pending deletion, call cancel_memory_deletion.\n\n"
        "Grounded factual lookup:\n"
        "- If the user explicitly asks you to look up, verify, check current information, or find factual resources, call answer_grounded_factual_lookup.\n"
        "- Do not call it for ordinary therapeutic support, coping tips, reflections, or exercise requests.\n"
        "- If a factual answer depends on location and the user did not provide one, ask what location they mean instead of guessing.\n\n"
        "Specialized role for this phase of the conversation:\n"
        f"{_THERAPEUTIC_AGENT_SPECIALIZATION_BLOCKS[agent_kind]}"
    )


def _copy_therapeutic_handoff_chat_ctx(
    chat_ctx: ChatContext | None,
) -> ChatContext | None:
    """Carry over dialogue plus whitelisted therapeutic system messages.

    Args:
        chat_ctx (ChatContext | None): Current context to hand off.

    Returns:
        ChatContext | None: Filtered context safe to pass to another
            therapeutic agent.
    """

    if chat_ctx is None:
        return None

    carried_items = []
    for item in chat_ctx.items:
        if item.type in {"function_call", "function_call_output"}:
            continue

        if _is_non_empty_dialogue_message(item):
            carried_items.append(item)
            continue

        if item.type == "message" and _chat_item_role(item) in {
            "system",
            "developer",
        }:
            source = (item.extra or {}).get("source")
            if source not in _PERSISTED_THERAPEUTIC_SYSTEM_SOURCES:
                continue
            if not (item.text_content or "").strip():
                continue
            carried_items.append(item)
            continue

    return ChatContext(carried_items)


async def _load_semantic_facts(
    store: MemoryStore, user_id: str, mode: MemoryMode
) -> list[str]:
    if mode == MemoryMode.INCOGNITO:
        return []
    try:
        records = await store.asearch(
            (user_id, "semantic"), query=None, limit=_MAX_MEMORY_ITEMS
        )
    except Exception:
        logger.warning("failed to load semantic facts", exc_info=True)
        return []
    records = filter_active_semantic_records(records)
    return [
        f"Previously noted: {r.value.get('evidence_quote', '')}"
        for r in records
        if r.value.get("evidence_quote")
    ]


async def _load_episodic_arcs(
    store: MemoryStore, user_id: str, mode: MemoryMode
) -> list[str]:
    if mode == MemoryMode.INCOGNITO:
        return []
    try:
        records = await store.asearch((user_id, "episodic"), query=None, limit=3)
    except Exception:
        logger.warning("failed to load episodic arcs", exc_info=True)
        return []
    arcs = []
    for r in records:
        summary = r.value.get("summary", "")
        if summary:
            themes = ", ".join(r.value.get("primary_themes", [])) or "untagged"
            arcs.append(f"Past session ({themes}): {summary}")
    return arcs


async def _load_procedural_memory(
    store: MemoryStore, user_id: str, mode: MemoryMode
) -> tuple[list[str], bool]:
    if mode == MemoryMode.INCOGNITO:
        return [], False
    try:
        profile = await aget_procedural_profile(store, user_id=user_id)
    except Exception:
        logger.warning("failed to load procedural rules", exc_info=True)
        return [], False
    return [rule.rule for rule in profile.rules[:_MAX_MEMORY_ITEMS]], (
        profile.proactive_recall_enabled
    )


async def _load_turn_relevant_semantic_facts(
    store: MemoryStore,
    *,
    user_id: str,
    mode: MemoryMode,
    query: str,
) -> list[tuple[str, str]]:
    """Fetch semantic facts relevant to the current user turn."""
    if mode == MemoryMode.INCOGNITO or not query.strip():
        return []

    try:
        records = await store.asearch(
            (user_id, "semantic"),
            query=query,
            limit=_MID_SESSION_MEMORY_ITEMS,
        )
    except Exception:
        logger.warning("failed to load turn-relevant semantic facts", exc_info=True)
        return []

    facts: list[tuple[str, str]] = []
    for record in filter_active_semantic_records(records):
        quote = (record.value.get("evidence_quote") or "").strip()
        if quote:
            facts.append((record.key, f"Previously noted: {quote}"))

    return facts


async def _handle_text_input(
    session: AgentSession[SessionData],
    event: room_io.TextInputEvent,
) -> None:
    """Track typed turns so text sessions can use the broader exercise registry."""
    session.userdata.last_input_modality = "text"
    await session.interrupt()
    session.generate_reply(user_input=event.text)


# ── TherapeuticAgent ────────────────────────────────────────────────


class TherapeuticAgent(Agent):
    """Main conversational agent for OpenCouch voice sessions.

    Handles all therapeutic conversation with tools for saving
    insights, running grounding exercises, and checking for crisis.
    """

    agent_kind: TherapeuticAgentKind = "hold_space"

    def __init__(
        self,
        *,
        instructions: str,
        chat_ctx: ChatContext | None = None,
        greet_on_enter: bool = False,
        greet_delay_seconds: float = 0.0,
    ) -> None:
        self._base_instructions = instructions
        self._greet_on_enter = greet_on_enter
        self._greet_delay_seconds = greet_delay_seconds
        super().__init__(
            instructions=_compose_therapeutic_agent_instructions(
                base_instructions=instructions,
                agent_kind=self.agent_kind,
            ),
            chat_ctx=chat_ctx,
            tools=[
                save_insight,
                show_saved_memory,
                show_memory_status,
                set_proactive_memory_recall,
                prepare_memory_deletion,
                prepare_indexed_memory_deletion,
                select_memory_deletion_candidate,
                confirm_memory_deletion,
                cancel_memory_deletion,
                answer_grounded_factual_lookup,
                crisis_check,
            ],
        )

    async def on_enter(self) -> None:
        if not self._greet_on_enter:
            return

        if self._greet_delay_seconds > 0:
            await asyncio.sleep(self._greet_delay_seconds)

        await self.session.generate_reply(
            instructions="Greet the user briefly and warmly. Sound like a calm "
            "person joining them, not an intake form. One sentence is enough."
        )

    @function_tool()
    async def start_grounding_exercise(
        self,
        context: RunContext[SessionData],
        technique: str,
    ) -> str:
        """Start a guided grounding, breathing, or relaxation exercise.

        Call this only when the user explicitly asks for a structured
        exercise or clearly agrees after you offered one. Examples:
        "guide me through breathing", "can we do grounding?", "walk me
        through box breathing", "maybe a relaxation technique?", or
        "yes, let's try that exercise."

        Once the user has asked for a technique or clearly agreed, call
        this tool and start the first step. Do not ask for readiness or
        confirmation again.

        Do not call this just because the user sounds anxious,
        overwhelmed, dysregulated, or says they want to calm down.
        In those moments, stay in ordinary conversation first:
        reflect, validate, and ask permission before shifting into a
        structured exercise.

        Prefer a specific technique label when the user names one.
        If they are asking again in the same session without naming a
        specific method, vary the exercise instead of repeating the
        exact same default.

        Args:
            context (RunContext[SessionData]): LiveKit run context with
                per-session state.
            technique (str): The exercise the user explicitly asked for
                or agreed to, for example "breathing", "grounding",
                "body scan", or "muscle relaxation".

        Returns:
            str: A short handoff summary telling the model to check in
                after the exercise ends.
        """
        if not _has_exercise_consent(self.chat_ctx):
            logger.info(
                "TherapeuticAgent: blocked grounding tool without explicit consent"
            )
            return (
                "Do not start a structured exercise yet. The user has not "
                "explicitly asked for one or clearly agreed to one. Stay in "
                "ordinary conversation: reflect what they said, ask at most one "
                "light follow-up, and if you think an exercise could help, ask "
                "permission first."
            )

        await emit_voice_activity(
            context,
            activity="exercise",
            status="started",
            label="Exercise active",
            detail="A guided exercise is in progress.",
        )
        result = await GroundingTask(
            technique=technique,
            chat_ctx=_copy_handoff_chat_ctx(self.chat_ctx),
            recent_exercise_types=tuple(context.userdata.recent_exercise_types),
            input_modality=context.userdata.last_input_modality,
        )
        await emit_voice_activity(
            context,
            activity="exercise",
            status="completed" if result.outcome == "completed" else "cancelled",
            label=(
                "Exercise completed"
                if result.outcome == "completed"
                else "Exercise stopped"
            ),
            detail=(
                "The guided exercise finished."
                if result.outcome == "completed"
                else "The guided exercise was stopped early."
            ),
        )
        recent = [
            exercise_type
            for exercise_type in context.userdata.recent_exercise_types
            if exercise_type != result.exercise_type
        ]
        recent.append(result.exercise_type)
        context.userdata.recent_exercise_types = recent[-3:]
        outcome = (
            f"completed all {result.steps_completed} steps"
            if result.outcome == "completed"
            else f"exited at step {result.steps_completed}/{result.total_steps}"
        )
        return (
            f"The user just finished {result.display_name} ({outcome}). "
            f"Check in with them about how it felt."
        )

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """Run deterministic safety, therapeutic control, and memory injection.

        Args:
            turn_ctx (ChatContext): Conversation context before the
                latest user turn is committed.
            new_message (ChatMessage): Latest user message for the turn.

        Returns:
            None: Mutates ``turn_ctx`` and session userdata in place.
        """
        text = (new_message.text_content or "").strip()
        if not text:
            return

        userdata: SessionData = self.session.userdata

        # Hard crisis keyword check — immediate agent swap.
        if matches_crisis_keywords(text):
            logger.warning(
                "safety net: crisis keywords detected, forcing CrisisAgent swap"
            )
            userdata.crisis_level = 3
            userdata.max_crisis_level = max(userdata.max_crisis_level, 3)
            self.session.update_agent(
                CrisisAgent(chat_ctx=_copy_handoff_chat_ctx(self.chat_ctx))
            )
            return

        therapeutic_state = _assess_therapeutic_process_state(
            turn_ctx=turn_ctx,
            user_text=text,
            previous=userdata.therapeutic_state,
        )
        userdata.therapeutic_state = therapeutic_state
        desired_agent_kind = _therapeutic_agent_kind_for_state(therapeutic_state)
        turn_ctx.add_message(
            role="system",
            content=_build_therapeutic_process_guidance(therapeutic_state),
            extra={"source": "therapeutic_process_controller"},
        )

        store = userdata.memory_store
        if store is not None and userdata.proactive_recall_enabled:
            facts = await _load_turn_relevant_semantic_facts(
                store,
                user_id=userdata.user_id,
                mode=userdata.memory_mode,
                query=text,
            )
            unseen_facts = [
                (key, fact)
                for key, fact in facts
                if key not in userdata.injected_semantic_memory_keys
            ]
            if unseen_facts:
                memory_keys = [key for key, _ in unseen_facts]
                memory_lines = "\n".join(f"- {fact}" for _, fact in unseen_facts)
                turn_ctx.add_message(
                    role="system",
                    content=(
                        "Relevant background from prior sessions. "
                        "Use it only if it fits naturally:\n"
                        f"{memory_lines}"
                    ),
                    extra={
                        "source": "semantic_memory_injection",
                        "memory_keys": memory_keys,
                    },
                )

                try:
                    await self.update_chat_ctx(turn_ctx)
                except Exception:
                    logger.warning(
                        "failed to persist turn-relevant semantic memory injection",
                        exc_info=True,
                    )
                else:
                    userdata.injected_semantic_memory_keys.update(memory_keys)
                    logger.info(
                        "livekit session: injected semantic memory user=%s facts=%d",
                        userdata.user_id,
                        len(memory_keys),
                    )

        if desired_agent_kind == self.agent_kind:
            return

        logger.info(
            "TherapeuticAgent: switching specialized agent %s -> %s",
            self.agent_kind,
            desired_agent_kind,
        )
        self.session.update_agent(
            _build_therapeutic_agent(
                agent_kind=desired_agent_kind,
                instructions=userdata.therapeutic_instructions
                or self._base_instructions,
                chat_ctx=_copy_therapeutic_handoff_chat_ctx(turn_ctx),
            )
        )


class HoldSpaceAgent(TherapeuticAgent):
    """Therapeutic agent specialized for attuned presence and pacing."""

    agent_kind: TherapeuticAgentKind = "hold_space"


class ReflectiveAgent(TherapeuticAgent):
    """Therapeutic agent specialized for tentative pattern reflection."""

    agent_kind: TherapeuticAgentKind = "reflective"


class UnderstandingAgent(TherapeuticAgent):
    """Therapeutic agent specialized for collaborative meaning-making."""

    agent_kind: TherapeuticAgentKind = "understanding"


class TechniqueAgent(TherapeuticAgent):
    """Therapeutic agent specialized for active, structured guidance."""

    agent_kind: TherapeuticAgentKind = "technique"


def _build_therapeutic_agent(
    *,
    agent_kind: TherapeuticAgentKind,
    instructions: str,
    chat_ctx: ChatContext | None = None,
    greet_on_enter: bool = False,
    greet_delay_seconds: float = 0.0,
) -> TherapeuticAgent:
    """Construct the specialized therapeutic agent for a session phase.

    Args:
        agent_kind (TherapeuticAgentKind): Specialized agent kind.
        instructions (str): Session-level therapeutic instructions.
        chat_ctx (ChatContext | None): Conversation context to carry over.
        greet_on_enter (bool): Whether the agent should greet on enter.
        greet_delay_seconds (float): Optional delay before the greeting
            starts, used to let remote audio attach cleanly.

    Returns:
        TherapeuticAgent: Specialized therapeutic agent instance.
    """

    agent_cls: type[TherapeuticAgent]
    if agent_kind == "reflective":
        agent_cls = ReflectiveAgent
    elif agent_kind == "understanding":
        agent_cls = UnderstandingAgent
    elif agent_kind == "technique":
        agent_cls = TechniqueAgent
    else:
        agent_cls = HoldSpaceAgent

    return agent_cls(
        instructions=instructions,
        chat_ctx=chat_ctx,
        greet_on_enter=greet_on_enter,
        greet_delay_seconds=greet_delay_seconds,
    )


# ── CrisisAgent ─────────────────────────────────────────────────────

_CRISIS_INSTRUCTIONS = """\
You are OpenCouch in crisis support mode. The user may be in danger.

YOUR ONLY PRIORITIES:
1. Stay calm, warm, and present. Do not panic or lecture.
2. Acknowledge what the user said. Do not minimize or dismiss.
3. If in the US or Canada, tell them they can call or text 988 (Suicide & Crisis Lifeline).
4. For other countries, use provide_crisis_resources when the user gives their location or asks for local resources.
5. Stay with the user. Do not rush to end the conversation.
6. If the user de-escalates and wants to continue talking, use the de_escalate tool.

Do NOT:
- Give medical or legal advice.
- Diagnose or label the user.
- Promise confidentiality you cannot guarantee.
- Attempt therapy techniques. Just be present.
- Invent hotline numbers or infer the user's location.
"""


class CrisisAgent(Agent):
    """Crisis support agent — activated on level 2+ crisis signals."""

    def __init__(self, chat_ctx: ChatContext | None = None) -> None:
        super().__init__(
            instructions=_CRISIS_INSTRUCTIONS,
            chat_ctx=chat_ctx,
            tools=[provide_crisis_resources],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="The user may be in crisis. Acknowledge the immediate "
            "risk plainly and warmly. Give 988 for US/Canada if relevant. Keep "
            "your voice steady and stay with them; do not sound like a policy notice."
        )

    @function_tool()
    async def de_escalate(self, context: RunContext[SessionData]) -> str:
        """Return to the therapeutic agent after crisis de-escalation.

        Call this ONLY when the user has clearly de-escalated — they
        say they are safe, they want to continue talking normally, or
        they explicitly say the crisis has passed.
        """
        logger.info("CrisisAgent: de-escalating back to TherapeuticAgent")
        context.userdata.crisis_level = 0

        instructions = (
            context.userdata.therapeutic_instructions or build_voice_system_prompt()
        )
        agent_kind = _therapeutic_agent_kind_for_state(
            context.userdata.therapeutic_state
        )

        return (
            _build_therapeutic_agent(
                agent_kind=agent_kind,
                instructions=instructions,
                chat_ctx=_copy_therapeutic_handoff_chat_ctx(self.chat_ctx),
            ),
            "The user has de-escalated. Transitioning back to supportive conversation.",
        )


# ── AgentServer and session entrypoint ──────────────────────────────

server = AgentServer(
    shutdown_process_timeout=45.0,
    initialize_process_timeout=45.0,
    setup_fnc=_prewarm_process,
)


@server.rtc_session(agent_name="opencouch-voice")
async def opencouch_voice(ctx: agents.JobContext):
    """LiveKit session entrypoint — called for each new voice session.

    1. Initializes the runtime (once per worker).
    2. Extracts user_id/thread_id from job metadata.
    3. Loads memory context.
    4. Creates AgentSession with RealtimeModel.
    5. Starts the TherapeuticAgent.
    """
    runtime = await _ensure_runtime()

    # Console mode uses a fake job with no remote participant.
    await ctx.connect()
    participant = None
    participant_metadata = None
    if not ctx.is_fake_job():
        participant = await ctx.wait_for_participant()
        participant_metadata = participant.metadata

    # ── Extract identity from metadata or local env overrides ───
    user_id, thread_id, transcription_language, assistant_voice, session_memory_mode = (
        _resolve_livekit_session_metadata(
            job_metadata=ctx.job.metadata,
            participant_metadata=participant_metadata,
        )
    )

    logger.info(
        "livekit session: starting user=%s thread=%s participant=%s transcription_language=%s assistant_voice=%s memory_mode=%s",
        user_id,
        thread_id,
        participant.identity if participant is not None else "console",
        transcription_language or "auto",
        assistant_voice,
        session_memory_mode.value,
    )

    # ── Load memory context ─────────────────────────────────────
    store = runtime.memory_store
    mode = session_memory_mode

    facts, arcs, rules = [], [], []
    proactive_recall_enabled = False
    if store is not None:
        # Load compact startup memory before Realtime session creation.
        facts = await _load_semantic_facts(store, user_id, mode)
        arcs = await _load_episodic_arcs(store, user_id, mode)
        rules, proactive_recall_enabled = await _load_procedural_memory(
            store, user_id, mode
        )

    instructions = build_voice_system_prompt(
        semantic_facts=facts,
        episodic_arcs=arcs,
        procedural_rules=rules,
        proactive_recall_enabled=proactive_recall_enabled,
    )

    logger.info(
        "livekit session: memory loaded facts=%d arcs=%d rules=%d prompt_chars=%d",
        len(facts),
        len(arcs),
        len(rules),
        len(instructions),
    )

    # ── Build session userdata ──────────────────────────────────
    userdata = SessionData(
        user_id=user_id,
        thread_id=thread_id,
        memory_store=store,
        memory_mode=mode,
        llm_client=_llm_client,
        proactive_recall_enabled=proactive_recall_enabled,
        started_at=iso_now(),
        therapeutic_instructions=instructions,
    )

    # ── Create session with RealtimeModel ───────────────────────
    vad = ctx.proc.userdata.get("vad") or silero.VAD.load()
    session = AgentSession[SessionData](
        llm=_build_realtime_model(
            transcription_language=transcription_language,
            assistant_voice=assistant_voice,
        ),
        vad=vad,
        turn_handling=_build_turn_handling(),
        userdata=userdata,
    )

    def _mark_voice_input(_event: agents.UserInputTranscribedEvent) -> None:
        session.userdata.last_input_modality = "voice"

    session.on("user_input_transcribed", _mark_voice_input)

    # ── Session-close finalization for transcript-derived memory ──
    finalization_task: asyncio.Task[None] | None = None

    async def _set_disconnect_finalization_status(
        *,
        thread_id: str,
        status: Literal["in_progress", "completed", "failed"],
        detail: str | None,
    ) -> None:
        try:
            await set_voice_finalization_status(
                thread_id,
                status=status,
                detail=detail,
            )
        except Exception:
            logger.warning(
                "livekit session: failed to persist disconnect finalization status",
                extra={"thread_id": thread_id, "status": status},
                exc_info=True,
            )

    async def _finalize_voice_session_transcript(*, trigger: str) -> None:
        if not _should_finalize_transcript_on_shutdown(is_fake_job=ctx.is_fake_job()):
            logger.info(
                "livekit session: skipping transcript finalization for console session"
            )
            return

        ud = session.userdata
        await _set_disconnect_finalization_status(
            thread_id=ud.thread_id,
            status="in_progress",
            detail="Saving session memory.",
        )

        if ud.memory_mode == MemoryMode.INCOGNITO:
            await _set_disconnect_finalization_status(
                thread_id=ud.thread_id,
                status="completed",
                detail="Incognito session; no memory saved.",
            )
            return

        try:
            transcript = _serialize_session_history(session.history)
            logger.info(
                "livekit session: finalization trigger=%s user=%s thread=%s transcript_turns=%d",
                trigger,
                ud.user_id,
                ud.thread_id,
                len(transcript),
            )
            if not runtime or not transcript:
                await _set_disconnect_finalization_status(
                    thread_id=ud.thread_id,
                    status="completed",
                    detail="No transcript memory to save.",
                )
                return

            await runtime.end_transcript_session(
                thread_id=ud.thread_id,
                user_id=ud.user_id,
                transcript=transcript,
                llm_client=_llm_client,
                started_at=ud.started_at,
                crisis_level_max=ud.max_crisis_level,
            )
            logger.info(
                "livekit session: transcript saved thread=%s turns=%d",
                ud.thread_id,
                len(transcript),
            )
            await _set_disconnect_finalization_status(
                thread_id=ud.thread_id,
                status="completed",
                detail="Session memory saved.",
            )
        except Exception:
            await _set_disconnect_finalization_status(
                thread_id=ud.thread_id,
                status="failed",
                detail="Session memory save failed.",
            )
            logger.warning("livekit session: failed to save transcript", exc_info=True)

    def _schedule_finalization(trigger: str) -> None:
        nonlocal finalization_task
        if finalization_task is not None:
            return
        finalization_task = asyncio.create_task(
            _finalize_voice_session_transcript(trigger=trigger),
            name="livekit_voice_finalize_transcript",
        )

    def _on_session_close(event) -> None:
        _schedule_finalization(
            f"session_close:{getattr(event.reason, 'value', event.reason)}"
        )

    session.on("close", _on_session_close)

    output_warmup_requested = False
    output_warmup_tasks: set[asyncio.Task[None]] = set()
    output_warmup_registered = False

    async def _handle_output_warmup(reader, participant_identity: str) -> None:
        nonlocal output_warmup_requested

        try:
            await reader.read_all()
        except Exception:
            logger.warning("livekit session: failed to read output warmup request")
            return

        if participant is not None and participant_identity != participant.identity:
            return
        if output_warmup_requested:
            return

        output_warmup_requested = True
        logger.info("livekit session: warming first voice output")
        session.generate_reply(
            instructions=(
                "Say exactly: \"I'm here whenever you're ready to start.\" "
                "Keep it brief and do not ask a question."
            ),
            tools=[],
            allow_interruptions=True,
        )

    def _on_output_warmup(reader, participant_identity: str) -> None:
        task = asyncio.create_task(
            _handle_output_warmup(reader, participant_identity),
            name="livekit_voice_output_warmup",
        )
        output_warmup_tasks.add(task)
        task.add_done_callback(output_warmup_tasks.discard)

    try:
        ctx.room.register_text_stream_handler(
            _VOICE_OUTPUT_WARMUP_TOPIC,
            _on_output_warmup,
        )
        output_warmup_registered = True
    except ValueError:
        logger.warning(
            "livekit session: output warmup text stream handler already registered"
        )

    async def _on_shutdown() -> None:
        if output_warmup_registered:
            try:
                ctx.room.unregister_text_stream_handler(_VOICE_OUTPUT_WARMUP_TOPIC)
            except ValueError:
                pass

        if output_warmup_tasks:
            for task in output_warmup_tasks:
                task.cancel()
            await asyncio.gather(*output_warmup_tasks, return_exceptions=True)

        if finalization_task is not None:
            await asyncio.shield(finalization_task)
        else:
            await _finalize_voice_session_transcript(trigger="job_shutdown")

        if ctx.is_fake_job():
            await _close_runtime()

    ctx.add_shutdown_callback(_on_shutdown)

    # ── Start the agent ─────────────────────────────────────────
    await session.start(
        room=ctx.room,
        agent=_build_therapeutic_agent(
            agent_kind="hold_space",
            instructions=instructions,
            greet_on_enter=ctx.is_fake_job(),
            greet_delay_seconds=0.0,
        ),
        room_options=room_io.RoomOptions(
            text_input=room_io.TextInputOptions(text_input_cb=_handle_text_input),
        ),
    )


# ── CLI entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    agents.cli.run_app(server)
