"""Typed session userdata for LiveKit voice sessions.

Passed to ``AgentSession[SessionData]`` so all agents, tools, and
hooks share a single typed state object that survives handoffs
within the same session.

This data is session-scoped (lives in memory for the duration of
the LiveKit job). Anything that must survive process restarts
should be persisted through the configured runtime backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from agent.gates.memory_control import MemoryControlTarget
from agent.memory.modes import MemoryMode
from agent.memory.store import MemoryStore
from services.llm.base import BaseLLMClient

_PERSISTENT_MODE_VALUES = {"persistent", "local", "synced"}
_INCOGNITO_MODE_VALUES = {"guest", "incognito", "private"}

SessionIntent = Literal["vent", "understand", "reflect", "work", "regulate", "close"]
GuidancePermission = Literal["unknown", "not_yet", "granted"]
ProcessStage = Literal["hold", "orient", "identify", "examine", "shift", "ground"]
TherapeuticApproach = Literal[
    "motivational_interviewing",
    "cbt",
    "act",
    "dbt_skills",
    "grief_support",
    "interpersonal_therapy",
    "pfa",
    "none",
]


@dataclass
class TherapeuticFormulation:
    """Lightweight session formulation for the current therapeutic thread."""

    situation: str = ""
    primary_emotion: str = ""
    hot_thought: str = ""
    pattern: str = ""
    user_goal: str = ""


@dataclass
class TherapeuticProcessState:
    """Session-scoped therapeutic controller state for voice turns."""

    session_intent: SessionIntent = "vent"
    guidance_permission: GuidancePermission = "unknown"
    process_stage: ProcessStage = "hold"
    therapeutic_approach: TherapeuticApproach = "motivational_interviewing"
    active_target: str = ""
    formulation: TherapeuticFormulation = field(default_factory=TherapeuticFormulation)


@dataclass
class SessionData:
    """Per-session state accessible via ``session.userdata``."""

    # ── Identity ────────────────────────────────────────────────
    user_id: str = "voice-user"
    thread_id: str = "voice-default"

    # ── Memory layer references ─────────────────────────────────
    memory_store: MemoryStore | None = None
    memory_mode: MemoryMode = MemoryMode.LOCAL
    llm_client: BaseLLMClient | None = None
    proactive_recall_enabled: bool = False
    pending_memory_delete: MemoryControlTarget | None = None
    pending_memory_delete_candidates: list[MemoryControlTarget] = field(
        default_factory=list
    )

    # ── Therapeutic session state ───────────────────────────────
    # Captured at session start so de-escalation can return to the
    # same therapeutic instructions that were initialized with memory.
    therapeutic_instructions: str = ""
    therapeutic_state: TherapeuticProcessState = field(
        default_factory=TherapeuticProcessState
    )

    # Semantic fact keys already injected into chat context mid-session.
    injected_semantic_memory_keys: set[str] = field(default_factory=set)

    # Recent grounding exercises used in this session, most recent last.
    # Used to diversify generic "help me calm down" requests.
    recent_exercise_types: list[str] = field(default_factory=list)

    # Whether the latest user turn arrived as spoken audio or typed text.
    # Used to keep voice sessions on voice-friendly exercises while still
    # allowing the larger exercise registry for text turns.
    last_input_modality: Literal["voice", "text"] = "voice"

    # ── Session timestamps ──────────────────────────────────────
    started_at: str = ""

    # ── Crisis tracking ─────────────────────────────────────────
    # Updated by the crisis_check tool or the keyword safety net.
    # 0 = no concern, 1 = mild, 2 = moderate, 3 = severe.
    crisis_level: int = 0
    max_crisis_level: int = 0


def parse_voice_memory_mode(
    value: str | None,
    *,
    default: MemoryMode = MemoryMode.LOCAL,
) -> MemoryMode:
    """Normalize a voice-session memory mode string.

    Args:
        value: User or environment supplied memory-mode label.
        default: Mode to use when ``value`` is blank or unrecognized.

    Returns:
        Normalized memory mode for the voice session.
    """

    normalized = (value or "").strip().lower()
    if not normalized:
        return default
    if normalized in _INCOGNITO_MODE_VALUES:
        return MemoryMode.INCOGNITO
    if normalized in _PERSISTENT_MODE_VALUES:
        return MemoryMode.LOCAL
    return default


def parse_optional_voice_memory_mode(value: str | None) -> MemoryMode | None:
    """Normalize an optional explicit voice-session memory mode.

    Args:
        value: User supplied memory-mode label.

    Returns:
        The normalized memory mode, or ``None`` when no recognized mode
        was supplied.
    """

    normalized = (value or "").strip().lower()
    if normalized in _INCOGNITO_MODE_VALUES:
        return MemoryMode.INCOGNITO
    if normalized in _PERSISTENT_MODE_VALUES:
        return MemoryMode.LOCAL
    return None
