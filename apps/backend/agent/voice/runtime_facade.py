"""Facade grouping voice-specific runtime methods.

Extracted from ``PersistentAgentRuntime`` to shrink the runtime module and
give the voice surface a clear ownership boundary. The facade receives only
the runtime operations it needs through an explicit collaboration bundle and
receives its shared resources via constructor injection.

The locking protocol is identical to the text path: ``_lock_for(thread_id)``
returns the **same** per-thread ``asyncio.Lock`` instance the runtime uses,
and ``active_session_mutation`` nests inside it.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from uuid import uuid4

from agent.audit.capture import capture_crisis_outcome
from agent.audit.models import CrisisResourceLookupStatus
from agent.memory.modes import MemoryMode
from agent.memory.operations.procedural_profile import aget_procedural_profile
from agent.memory.store import MemoryStore
from agent.skills.guided_exercises.catalog.registry import get_exercise_display_name
from agent.skills.guided_exercises.lifecycle.memory import (
    ExerciseCompletionMemoryRequest,
    write_exercise_completion_fact,
)
from agent.models import Channel, CrisisAssessment
from agent.observability.decorators import trace_event, trace_span
from agent.observability.events import (
    RUNTIME_VOICE_SESSION,
    VOICE_CRISIS_RESOURCE_LOOKUP_PERSISTED,
    VOICE_RESPONSE_FINALIZED,
    VOICE_SAFETY_INTERRUPTED_TURN_RECORDED,
    VOICE_TURN_COMPLETION_METADATA_PERSIST_FAILED,
)
from agent.observability.timing import elapsed_ms
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.session import turn_count_from_state
from agent.runtime.session.active_session import ActiveSessionManager
from agent.runtime.session.service import SessionLifecycleService
from agent.runtime.state_ops import apply_state_delta
from agent.runtime.state_store import RuntimeStateStore
from agent.state import AgentState, resolve_owner_id
from agent.tools.grounded_search import (
    CrisisResourceLookupRequest,
    find_crisis_resources_for_request,
)
from agent.voice.concurrent_safety import (
    VoiceConcurrentSafetyResult,
    VoiceConcurrentSafetyService,
)
from agent.voice.post_turn_safety import (
    VoicePostTurnSafetyAuditor,
    VoicePostTurnSafetyCheck,
    VoicePostTurnSafetyScheduleResult,
)
from agent.voice.safety_overlay import (
    VoiceSafetyDecision,
    VoiceSafetyOverlayService,
    VoiceSafetyResourceResolution,
)
from agent.voice.runtime_collaboration import VoiceRuntimeCollaboration
from agent.voice.state_transition import VoiceTurnStateInputs, build_voice_turn_state
from llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

_MAX_RECORDED_VOICE_TURN_HASHES = 256
_MAX_PENDING_VOICE_RESOURCE_LOOKUPS = 32


@dataclass(frozen=True, slots=True)
class VoiceTurnRecordReceipt:
    """Stable response facts retained for an idempotent turn retry."""

    request_hash: str
    message_count: int
    post_turn_safety: dict[str, Any] | None


_CONCURRENT_SAFETY_SNAPSHOT_TIMEOUT_SECONDS = 1.0
_SAFETY_RESOURCE_LOOKUP_TIMEOUT_SECONDS = 8.0


# ── Module-level helpers (moved from runtime.py) ─────────────────


def _latest_user_text(transcript: list[dict[str, object]]) -> str:
    for turn in reversed(transcript):
        if turn.get("role") == "user":
            return str(turn.get("content") or "").strip()
    return ""


def _guided_exercise_start_conflict_result(
    result: Mapping[str, Any],
    *,
    exercise_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a stable conflict response when a concurrent start won the lock."""

    active_skill_id = exercise_state.get("exercise_type")
    active_step_index = exercise_state.get("exercise_step")
    active_step_id = exercise_state.get("exercise_step_id")
    return {
        **result,
        "status": "conflict",
        "runtime_action": "conflict",
        "skill_id": active_skill_id if isinstance(active_skill_id, str) else None,
        "current_step_index": (
            active_step_index if isinstance(active_step_index, int) else None
        ),
        "current_step_id": active_step_id if isinstance(active_step_id, str) else None,
        "exercise_state_delta": {},
        "skill_context": "",
        "response_instruction": (
            "Do not start another exercise. Re-orient to the runtime-provided "
            "active exercise or ask the user whether they want to continue it."
        ),
        "side_effect": "none",
        "retry_safe": True,
    }


def _guided_exercise_progress_conflict_result(
    result: Mapping[str, Any],
    *,
    exercise_state: Mapping[str, Any],
    expected_step_id: object,
) -> dict[str, Any]:
    """Return a stable conflict response for stale voice exercise progress."""

    active_skill_id = exercise_state.get("exercise_type")
    active_step_id = exercise_state.get("exercise_step_id")
    return {
        **result,
        "status": "conflict",
        "runtime_action": "conflict",
        "skill_id": active_skill_id if isinstance(active_skill_id, str) else None,
        "previous_step_id": (
            expected_step_id if isinstance(expected_step_id, str) else None
        ),
        "current_step_id": active_step_id if isinstance(active_step_id, str) else None,
        "next_step_id": None,
        "exercise_state_delta": {},
        "response_instruction": (
            "Do not advance the exercise. Re-orient to the runtime-provided "
            "active step or ask the user whether they want to continue."
        ),
        "side_effect": "none",
        "retry_safe": True,
    }


def _guided_exercise_completion_retry_result(
    result: Mapping[str, Any],
    *,
    exercise_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep a terminal exercise step active when its memory write failed."""

    active_skill_id = exercise_state.get("exercise_type")
    active_step_id = exercise_state.get("exercise_step_id")
    previous_step_id = result.get("previous_step_id")
    return {
        **result,
        "status": "active",
        "runtime_action": "hold",
        "skill_id": active_skill_id if isinstance(active_skill_id, str) else None,
        "previous_step_id": (
            previous_step_id if isinstance(previous_step_id, str) else None
        ),
        "current_step_id": active_step_id if isinstance(active_step_id, str) else None,
        "next_step_id": None,
        "exercise_state_delta": {},
        "response_instruction": (
            "Do not report the exercise complete yet. Keep the final step active "
            "and invite the user to confirm completion again."
        ),
        "side_effect": "none",
        "retry_safe": True,
    }


def _compact_voice_memory_context(delta: Mapping[str, Any]) -> str:
    blocks: list[str] = []
    procedural_profile = delta.get("procedural_profile") or {}
    proactive_recall_enabled = False
    if isinstance(procedural_profile, Mapping):
        proactive_recall_enabled = bool(
            procedural_profile.get("proactive_recall_enabled", False)
        )

    if proactive_recall_enabled:
        working_memory = delta.get("working_memory") or []
        if working_memory:
            rendered = [
                _compact_memory_value(item) for item in list(working_memory)[:5]
            ]
            rendered = [item for item in rendered if item]
            if rendered:
                blocks.append(
                    "Relevant saved facts:\n"
                    + "\n".join(f"- {item}" for item in rendered)
                )

        session_memory = delta.get("session_memory") or {}
        if isinstance(session_memory, Mapping):
            summary = str(session_memory.get("summary") or "").strip()
            if summary and summary != "Guest session without long-term memory.":
                blocks.append(f"Recent session summary: {summary}")

    if isinstance(procedural_profile, Mapping):
        rules = procedural_profile.get("procedural_rules") or []
        rendered_rules = [_compact_memory_value(rule) for rule in list(rules)[:5]]
        rendered_rules = [rule for rule in rendered_rules if rule]
        if rendered_rules:
            blocks.append(
                "Saved response preferences:\n"
                + "\n".join(f"- {rule}" for rule in rendered_rules)
            )
    if proactive_recall_enabled:
        blocks.append("Proactive memory recall is enabled.")
    else:
        blocks.append("Proactive saved-memory recall is disabled.")

    return "\n\n".join(blocks)[:2000]


def _compact_memory_value(value: object) -> str:
    if isinstance(value, Mapping):
        for key in (
            "evidence_quote",
            "rule",
            "summary",
            "preference",
            "text",
            "content",
        ):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return json.dumps(dict(value), sort_keys=True, default=str)[:300]
    return str(value).strip()[:300]


# ── Facade class ─────────────────────────────────────────────────


class VoiceRuntimeFacade:
    """Groups voice-specific methods previously on ``PersistentAgentRuntime``.

    Callers switch from ``runtime.method(...)`` to
    ``runtime.voice.method(...)``.  The facade shares the runtime's
    per-thread locks, state store, and active-session manager so that the
    voice and text paths cannot race on the same thread.
    """

    def __init__(
        self,
        *,
        collaboration: VoiceRuntimeCollaboration,
        state_store: RuntimeStateStore,
        memory_store: MemoryStore,
        active_session_manager: ActiveSessionManager,
        session_lifecycle: SessionLifecycleService,
        lock_for: Callable[[str], asyncio.Lock],
        memory_mode: MemoryMode,
    ) -> None:
        self._collaboration = collaboration
        self._state_store = state_store
        self._memory_store = memory_store
        self._active_session_manager = active_session_manager
        self._session_lifecycle = session_lifecycle
        self._lock_for = lock_for
        self._memory_mode = memory_mode
        self._concurrent_safety_service = VoiceConcurrentSafetyService()
        self._safety_overlay_service = VoiceSafetyOverlayService()
        self._post_turn_safety_auditor = VoicePostTurnSafetyAuditor()

    @property
    def post_turn_safety_pending_count(self) -> int:
        """Return the number of pending post-turn voice safety checks."""

        return self._post_turn_safety_auditor.pending_count

    async def drain_post_turn_safety_checks(
        self,
        timeout_seconds: float | None = None,
    ) -> int:
        """Wait for scheduled post-turn voice safety checks to finish."""

        return await self._post_turn_safety_auditor.drain(timeout_seconds)

    async def aclose(self) -> None:
        """Close voice-owned background resources."""

        await self._post_turn_safety_auditor.aclose()

    async def assess_voice_turn_safety(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        user_text: str,
        prior_message_count: int,
        pending_prior_transcript: list[dict[str, Any]],
        llm_client: BaseLLMClient | None,
    ) -> VoiceConcurrentSafetyResult:
        """Assess a voice turn against an isolated prior-transcript snapshot."""

        started_at = time.monotonic()
        try:
            async with asyncio.timeout(_CONCURRENT_SAFETY_SNAPSHOT_TIMEOUT_SECONDS):
                prior_transcript = await self._voice_prior_transcript_snapshot(
                    thread_id=thread_id,
                    prior_message_count=prior_message_count,
                    pending_prior_transcript=pending_prior_transcript,
                )
        except TimeoutError:
            return VoiceConcurrentSafetyResult(
                status="timeout",
                reason="timeout",
                assessment=None,
                duration_ms=round(elapsed_ms(started_at), 2),
            )
        except Exception:
            return VoiceConcurrentSafetyResult(
                status="failed",
                reason="state_snapshot_failed",
                assessment=None,
                duration_ms=round(elapsed_ms(started_at), 2),
            )

        result = await self._concurrent_safety_service.assess_turn(
            thread_id=thread_id,
            user_id=user_id,
            user_text=user_text,
            prior_transcript=prior_transcript,
            llm_client=llm_client,
        )
        return replace(result, duration_ms=round(elapsed_ms(started_at), 2))

    def decide_voice_safety(
        self,
        result: VoiceConcurrentSafetyResult,
    ) -> VoiceSafetyDecision:
        """Apply the server-owned concurrent safety policy."""

        return self._safety_overlay_service.decide(result)

    async def resolve_voice_safety_resources(
        self,
        *,
        thread_id: str,
        user_text: str,
        prior_message_count: int,
        pending_prior_transcript: list[dict[str, Any]],
        llm_client: BaseLLMClient | None,
    ) -> VoiceSafetyResourceResolution:
        """Resolve verified resources from an isolated snapshot without mutation."""

        if llm_client is None:
            return self._safety_overlay_service.resource_resolution(
                inferred_location="",
                resources=[],
                status="lookup_error",
            )

        try:
            async with asyncio.timeout(_SAFETY_RESOURCE_LOOKUP_TIMEOUT_SECONDS):
                transcript = await self._voice_prior_transcript_snapshot(
                    thread_id=thread_id,
                    prior_message_count=prior_message_count,
                    pending_prior_transcript=pending_prior_transcript,
                )
                (
                    inferred_location,
                    resources,
                    status,
                ) = await find_crisis_resources_for_request(
                    CrisisResourceLookupRequest(
                        current_user_message=user_text.strip(),
                        transcript=tuple(transcript),
                    ),
                    llm_client=llm_client,
                )
        except TimeoutError:
            inferred_location, resources, status = "", [], "lookup_error"
        except Exception as exc:
            logger.warning(
                "voice safety resource resolution failed: %s",
                type(exc).__name__,
            )
            inferred_location, resources, status = "", [], "lookup_error"

        return self._safety_overlay_service.resource_resolution(
            inferred_location=inferred_location,
            resources=resources,
            status=status,
        )

    async def _voice_prior_transcript_snapshot(
        self,
        *,
        thread_id: str,
        prior_message_count: int,
        pending_prior_transcript: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        async with self._lock_for(thread_id):
            prior_state = await self._state_store.load_state(thread_id)
            transcript = (
                list(prior_state.get("transcript", []) or [])
                if prior_state is not None
                else []
            )
            snapshot = copy.deepcopy(transcript[:prior_message_count])
            snapshot.extend(copy.deepcopy(pending_prior_transcript))
            return snapshot

    # ── build_voice_tool_context ─────────────────────────────────

    async def build_voice_tool_context(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        current_user_message: str,
        transcript: list[dict[str, object]],
        client_turn_id: str | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> OpenAITextRunContext:
        """Build the app-owned context used by voice function tools."""

        effective_user_message = current_user_message.strip() or _latest_user_text(
            transcript
        )
        if not effective_user_message:
            effective_user_message = "voice tool call"
        prior_state = await self._collaboration.get_state(thread_id)
        prior_turn_count = turn_count_from_state(prior_state)
        initial_state = self._collaboration.build_turn_initial_state(
            thread_id=thread_id,
            message=effective_user_message,
            channel=Channel.VOICE,
            user_id=user_id,
            installed_skills=None,
            prior_turn_count=prior_turn_count,
        )
        state = cast(AgentState, {**dict(prior_state or {}), **dict(initial_state)})
        if prior_state is not None:
            persisted_delta: dict[str, Any] = {}
            for key in (
                "memory_control",
                "procedural_profile",
                "session_memory",
                "exercise_state",
            ):
                value = prior_state.get(key)
                if isinstance(value, Mapping):
                    persisted_delta[key] = dict(value)
            if persisted_delta:
                apply_state_delta(state, persisted_delta)
        if transcript:
            state["transcript"] = cast(Any, [dict(turn) for turn in transcript])
        elif prior_state is not None:
            state["transcript"] = list(prior_state.get("transcript", []) or [])

        memory_control = state.get("memory_control", {}) or {}
        pending_memory_action = (
            memory_control.get("pending_action")
            if isinstance(memory_control, Mapping)
            else None
        )
        workflow_context = self._collaboration.build_workflow_context(
            thread_id=thread_id,
            message=effective_user_message,
            prior_state=prior_state,
            user_id=user_id,
            llm_client=llm_client,
            response_llm_client=llm_client,
            track_session=False,
        )
        context = OpenAITextRunContext(
            thread_id=thread_id,
            workflow_context=workflow_context,
            current_user_message=effective_user_message,
            user_id=user_id,
            session_id=thread_id,
            channel=Channel.VOICE,
            pending_memory_action=(
                dict(pending_memory_action)
                if isinstance(pending_memory_action, Mapping)
                else None
            ),
            agent_state=state,
            installed_skills=list(state.get("installed_skills", []) or []),
            transcript=cast(list[dict[str, Any]], list(state.get("transcript", []))),
            turn_count=turn_count_from_state(state),
        )
        correlation_hash = (
            hashlib.sha256(f"{thread_id}\0{client_turn_id}".encode("utf-8")).hexdigest()
            if client_turn_id is not None
            else None
        )
        self._rehydrate_crisis_resource_lookup(
            context,
            prior_state,
            correlation_hash=correlation_hash,
        )
        return context

    # ── _rehydrate_crisis_resource_lookup ─────────────────────────

    @staticmethod
    def _rehydrate_crisis_resource_lookup(
        context: OpenAITextRunContext,
        prior_state: Mapping[str, Any] | None,
        *,
        correlation_hash: str | None,
    ) -> None:
        """Re-seed a prior voice crisis lookup onto a freshly built context.

        Counterpart to ``persist_voice_crisis_resource_lookup``: because each
        Realtime tool call builds its own context, the resource result a prior
        ``lookup_crisis_resources`` call found only survives in thread state.
        Restoring it here lets ``latest_crisis_resource_tool_result`` return it
        so ``get_crisis_support_template`` can reuse verified resources instead
        of degrading to ``not_attempted``.

        Reads ``prior_state`` rather than the merged turn state on purpose: the
        per-turn ``build_initial_state`` defaults reset these fields to
        ``not_attempted`` / empty, so only the pre-merge persisted state still
        carries the prior lookup.
        """

        if prior_state is None:
            return
        resource_state: Mapping[str, Any] = prior_state
        if correlation_hash is not None:
            diagnostics = prior_state.get("diagnostics", {})
            pending_lookups = (
                diagnostics.get("voice_crisis_resource_lookups", {})
                if isinstance(diagnostics, Mapping)
                else {}
            )
            matched = (
                pending_lookups.get(correlation_hash)
                if isinstance(pending_lookups, Mapping)
                else None
            )
            if not isinstance(matched, Mapping):
                return
            resource_state = matched
        status = resource_state.get("resource_lookup_status")
        if not isinstance(status, str) or status in {"", "not_attempted"}:
            return
        found_resources = resource_state.get("found_resources")
        rows = (
            [dict(row) for row in found_resources]
            if isinstance(found_resources, list)
            else []
        )
        inferred_location = resource_state.get("inferred_location")
        context.record_crisis_resource_tool_result(
            response_text="",
            inferred_location=(
                inferred_location if isinstance(inferred_location, str) else ""
            ),
            found_resources=rows,
            resource_lookup_status=cast(CrisisResourceLookupStatus, status),
        )

    # ── persist_voice_memory_tool_result ──────────────────────────

    async def persist_voice_memory_tool_result(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        current_user_message: str,
        transcript: list[dict[str, object]],
        result: Mapping[str, Any],
    ) -> None:
        """Persist memory-tool state deltas between Realtime tool calls.

        OpenAI Realtime invokes each voice tool through a separate request. Text
        turns keep memory-control deltas in ``OpenAITextRunContext`` until the
        response merge step; voice needs to save the same grouped channels after
        each mutating tool call so a later ``confirm``/``cancel`` call can read a
        pending action and finalized turns preserve procedural-profile changes.
        """

        delta: dict[str, Any] = {}
        memory_control = result.get("memory_control")
        if isinstance(memory_control, Mapping):
            delta["memory_control"] = dict(memory_control)
        procedural_profile = result.get("procedural_profile")
        if isinstance(procedural_profile, Mapping):
            delta["procedural_profile"] = dict(procedural_profile)
        if bool(result.get("clear_session_buffer")):
            delta["session_memory"] = {
                "held_semantic_candidates": [],
                "held_procedural_candidates": [],
            }
        if not delta:
            return

        effective_user_message = current_user_message.strip() or _latest_user_text(
            transcript
        )
        if not effective_user_message:
            effective_user_message = "voice memory tool call"

        async with self._lock_for(thread_id):
            prior_state = await self._state_store.load_state(thread_id)
            if prior_state is None:
                state = cast(
                    AgentState,
                    dict(
                        self._collaboration.build_turn_initial_state(
                            thread_id=thread_id,
                            message=effective_user_message,
                            channel=Channel.VOICE,
                            user_id=user_id,
                            installed_skills=None,
                            prior_turn_count=-1,
                        )
                    ),
                )
                state["transcript"] = []
            else:
                state = cast(AgentState, dict(prior_state))
            apply_state_delta(state, delta)
            await self._state_store.save_state(thread_id, state)

    # ── prepare_voice_guided_exercise_tool ────────────────────────

    async def prepare_voice_guided_exercise_tool(
        self,
        *,
        thread_id: str,
        llm_client: BaseLLMClient | None,
    ) -> None:
        """Ensure voice guided-exercise tools see current session continuity.

        Realtime tool calls arrive before final turn persistence. Preparing the
        session before start or progress validation lets expired or absent
        sessions clear stale exercise state before the tool result is computed.
        """

        async with self._lock_for(thread_id):
            prior_state = await self._collaboration.get_state(thread_id)
            await self._collaboration.prepare_session_for_turn(
                thread_id=thread_id,
                prior_state=prior_state,
                llm_client=llm_client,
            )

    # ── persist_voice_guided_exercise_result ──────────────────────

    async def persist_voice_guided_exercise_result(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        current_user_message: str,
        transcript: list[dict[str, object]],
        result: Mapping[str, Any],
        memory_mode: str,
    ) -> dict[str, Any]:
        """Persist voice exercise state and record terminal completion memory."""

        delta = result.get("exercise_state_delta")
        if not isinstance(delta, Mapping) or not delta:
            return dict(result)

        effective_user_message = current_user_message.strip() or _latest_user_text(
            transcript
        )
        if not effective_user_message:
            effective_user_message = "voice guided exercise tool call"

        persisted_result = dict(result)
        completion_memory_mode = (
            MemoryMode.INCOGNITO if memory_mode == "incognito" else self._memory_mode
        )
        async with self._lock_for(thread_id):
            prior_state = await self._state_store.load_state(thread_id)
            if prior_state is None:
                state = cast(
                    AgentState,
                    dict(
                        self._collaboration.build_turn_initial_state(
                            thread_id=thread_id,
                            message=effective_user_message,
                            channel=Channel.VOICE,
                            user_id=user_id,
                            installed_skills=None,
                            prior_turn_count=-1,
                        )
                    ),
                )
                state["transcript"] = []
            else:
                state = cast(AgentState, dict(prior_state))

            if result.get("runtime_action") == "start":
                active_state = state.get("exercise_state", {}) or {}
                active_skill_id = active_state.get("exercise_type")
                if isinstance(active_skill_id, str) and active_skill_id:
                    return _guided_exercise_start_conflict_result(
                        result,
                        exercise_state=active_state,
                    )
            else:
                expected_skill_id = result.get("skill_id")
                expected_step_id = result.get("previous_step_id")
                active_state_value = state.get("exercise_state", {}) or {}
                active_state = (
                    active_state_value
                    if isinstance(active_state_value, Mapping)
                    else {}
                )
                if (
                    not isinstance(expected_skill_id, str)
                    or not isinstance(expected_step_id, str)
                    or active_state.get("exercise_type") != expected_skill_id
                    or active_state.get("exercise_step_id") != expected_step_id
                ):
                    return _guided_exercise_progress_conflict_result(
                        result,
                        exercise_state=active_state,
                        expected_step_id=expected_step_id,
                    )

            skill_id = result.get("skill_id")
            if (
                result.get("status") == "completed"
                and result.get("runtime_action") == "complete"
                and isinstance(skill_id, str)
                and skill_id
            ):
                # Do not clear the final active step until its required memory
                # effect succeeds: a cancellation or write failure must remain retryable.
                completion_persisted = await write_exercise_completion_fact(
                    request=ExerciseCompletionMemoryRequest(
                        owner_id=resolve_owner_id(state),
                        session_id=thread_id,
                        turn_count=turn_count_from_state(state) + 1,
                        exercise_type=skill_id,
                        display_name=get_exercise_display_name(
                            skill_id,
                            default=skill_id,
                        ),
                    ),
                    memory_store=self._memory_store,
                    memory_mode=completion_memory_mode,
                )
                if not completion_persisted:
                    return _guided_exercise_completion_retry_result(
                        result,
                        exercise_state=active_state,
                    )

            apply_state_delta(state, dict(delta))
            await self._state_store.save_state(thread_id, state)
        return persisted_result

    # ── persist_voice_crisis_resource_lookup ──────────────────────

    async def persist_voice_crisis_resource_lookup(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        inferred_location: str,
        found_resources: list[dict[str, str]],
        resource_lookup_status: str,
        client_turn_id: str | None = None,
    ) -> None:
        """Persist a voice crisis-resource lookup so a later tool call can reuse it.

        OpenAI Realtime invokes each app tool as a separate ``/realtime/tools``
        request, so the ``OpenAITextRunContext`` built per request starts with an
        empty ``crisis_resource_tool_calls`` list. Without persistence, a later
        ``get_crisis_support_template`` call cannot see the resource the
        immediately preceding ``lookup_crisis_resources`` call found. Recording
        the result onto thread state lets ``build_voice_tool_context`` rehydrate
        it on the next request. ``save_state`` is a whole-document replace, so
        this reads-modifies-writes under the thread lock to avoid clobbering
        concurrent state.

        Tool calls fire mid-turn, before ``record_voice_turn`` finalizes the
        turn, so on a first-turn crisis no state row exists yet. Seed a minimal
        turn state in that case rather than dropping the lookup -- a first-turn
        crisis is exactly when the scaffold must still see the resource.
        """

        async with self._lock_for(thread_id):
            prior_state = await self._state_store.load_state(thread_id)
            if prior_state is None:
                state = cast(
                    AgentState,
                    dict(
                        self._collaboration.build_turn_initial_state(
                            thread_id=thread_id,
                            message="voice tool call",
                            channel=Channel.VOICE,
                            user_id=user_id,
                            installed_skills=None,
                            prior_turn_count=-1,
                        )
                    ),
                )
                state["transcript"] = []
            else:
                state = cast(AgentState, dict(prior_state))
            state["inferred_location"] = inferred_location
            state["found_resources"] = [dict(row) for row in found_resources]
            state["resource_lookup_status"] = resource_lookup_status
            diagnostics = dict(state.get("diagnostics", {}) or {})
            if client_turn_id is not None:
                correlation_hash = hashlib.sha256(
                    f"{thread_id}\0{client_turn_id}".encode("utf-8")
                ).hexdigest()
                diagnostics["voice_crisis_resource_turn_hash"] = correlation_hash
                pending_lookups = dict(
                    diagnostics.get("voice_crisis_resource_lookups", {}) or {}
                )
                pending_lookups[correlation_hash] = {
                    "inferred_location": inferred_location,
                    "found_resources": [dict(row) for row in found_resources],
                    "resource_lookup_status": resource_lookup_status,
                }
                diagnostics["voice_crisis_resource_lookups"] = dict(
                    list(pending_lookups.items())[-_MAX_PENDING_VOICE_RESOURCE_LOOKUPS:]
                )
            else:
                diagnostics.pop("voice_crisis_resource_turn_hash", None)
            state["diagnostics"] = diagnostics
            await self._state_store.save_state(thread_id, state)
            trace_event(
                VOICE_CRISIS_RESOURCE_LOOKUP_PERSISTED,
                {
                    "voice_runtime": "openai_realtime",
                    "resource_lookup_status": resource_lookup_status,
                    "resource_count": len(found_resources),
                },
            )

    # ── voice_session_memory_context ─────────────────────────────

    async def voice_session_memory_context(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        memory_mode: str | None = None,
    ) -> str:
        """Return compact saved-memory context for a Realtime voice session.

        Voice sessions are created before the user has said anything, so a
        semantic recall here has no real query to match on. Bootstrap context
        is therefore limited to the user's standing procedural rules and the
        proactive-recall toggle; topic-specific recall happens mid-session via
        the ``recall_saved_memory`` tool when the user introduces a topic.
        """

        if memory_mode == "incognito" or self._memory_mode == MemoryMode.INCOGNITO:
            return ""

        owner_id = resolve_owner_id({"user_id": user_id, "session_id": thread_id})
        profile = await aget_procedural_profile(self._memory_store, user_id=owner_id)
        delta: dict[str, Any] = {
            "working_memory": [],
            "session_memory": {"summary": ""},
            "procedural_profile": {
                "procedural_rules": [{"rule": rule.rule} for rule in profile.rules],
                "proactive_recall_enabled": profile.proactive_recall_enabled,
            },
        }
        return _compact_voice_memory_context(delta)

    async def voice_session_message_count(self, *, thread_id: str) -> int:
        """Return the persisted transcript watermark for a new voice session."""

        async with self._lock_for(thread_id):
            state = await self._state_store.load_state(thread_id)
            return len(state.get("transcript", []) or []) if state is not None else 0

    async def recorded_voice_turn_receipt(
        self,
        *,
        thread_id: str,
        correlation_hash: str,
        request_hash: str,
    ) -> VoiceTurnRecordReceipt | None:
        """Return the original response facts for an idempotent retry."""

        async with self._lock_for(thread_id):
            state = await self._state_store.load_state(thread_id)
            if state is None or not _voice_turn_hash_recorded(state, correlation_hash):
                return None
            _validate_voice_turn_request_hash(state, correlation_hash, request_hash)
            diagnostics = state.get("diagnostics", {})
            receipts = (
                diagnostics.get("voice_recorded_turn_receipts", {})
                if isinstance(diagnostics, Mapping)
                else {}
            )
            receipt = (
                receipts.get(correlation_hash)
                if isinstance(receipts, Mapping)
                else None
            )
            if isinstance(receipt, Mapping):
                post_turn_safety = receipt.get("post_turn_safety")
                return VoiceTurnRecordReceipt(
                    request_hash=request_hash,
                    message_count=int(receipt.get("message_count") or 0),
                    post_turn_safety=(
                        dict(post_turn_safety)
                        if isinstance(post_turn_safety, Mapping)
                        else None
                    ),
                )
            post_turn_safety = (
                diagnostics.get("voice_post_turn_safety")
                if isinstance(diagnostics, Mapping)
                else None
            )
            return VoiceTurnRecordReceipt(
                request_hash=request_hash,
                message_count=len(state.get("transcript", []) or []),
                post_turn_safety=(
                    dict(post_turn_safety)
                    if isinstance(post_turn_safety, Mapping)
                    else None
                ),
            )

    # ── record_voice_turn ────────────────────────────────────────

    @trace_span(
        RUNTIME_VOICE_SESSION,
        attrs={"voice_runtime": "openai_realtime"},
    )
    async def record_voice_turn(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        user_text: str,
        assistant_text: str,
        outcome: Literal[
            "completed", "connection_interrupted", "safety_interrupted"
        ] = "completed",
        route: str | None = None,
        response_style: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        llm_client: BaseLLMClient | None = None,
        correlation_hash: str | None = None,
        request_hash: str | None = None,
        safety_assessment: CrisisAssessment | None = None,
    ) -> AgentState:
        """Persist a finalized voice turn without running the text agent."""

        if outcome not in {
            "completed",
            "connection_interrupted",
            "safety_interrupted",
        }:
            raise ValueError(f"Unsupported voice turn outcome: {outcome}")
        if (
            outcome in {"connection_interrupted", "safety_interrupted"}
            and not user_text.strip()
        ):
            raise ValueError(f"{outcome} requires user_text")
        voice_tool_calls = list(tool_calls or [])
        if outcome in {"connection_interrupted", "safety_interrupted"}:
            voice_tool_calls = [
                call
                for call in voice_tool_calls
                if call.get("status") in {"completed", "failed"}
            ]
            assistant_text = ""
            route = f"voice_{outcome}"
            response_style = f"voice_{outcome}"
        async with self._lock_for(thread_id):
            self._collaboration.remember_llm_client(thread_id, llm_client)
            prior_state = await self._collaboration.get_state(thread_id)
            if (
                correlation_hash is not None
                and prior_state is not None
                and _voice_turn_hash_recorded(prior_state, correlation_hash)
            ):
                if request_hash is not None:
                    _validate_voice_turn_request_hash(
                        prior_state,
                        correlation_hash,
                        request_hash,
                    )
                return prior_state
            pending_turn = (
                _pending_voice_turn(prior_state, correlation_hash)
                if prior_state is not None and correlation_hash is not None
                else None
            )
            if pending_turn is not None:
                pending_turn_instance_id = pending_turn.get("turn_instance_id")
                legacy_request_hash = pending_turn.get("request_hash")
                legacy_identity = (
                    legacy_request_hash
                    if isinstance(legacy_request_hash, str)
                    else correlation_hash
                )
                turn_instance_id = (
                    pending_turn_instance_id
                    if isinstance(pending_turn_instance_id, str)
                    and pending_turn_instance_id
                    else hashlib.sha256(
                        f"legacy-voice-turn\0{correlation_hash}\0{legacy_identity}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                )
                if request_hash is not None:
                    _validate_voice_turn_request_hash(
                        prior_state,
                        correlation_hash,
                        request_hash,
                    )
                state = prior_state
                prior_message_count = int(pending_turn.get("prior_message_count") or 0)
                safety_prior_state = cast(
                    AgentState,
                    {
                        "transcript": list(state.get("transcript", []) or [])[
                            :prior_message_count
                        ]
                    },
                )
                turn_route = str(state.get("route") or route or "")
                turn_response_style = str(
                    state.get("response_style") or response_style or ""
                )
            else:
                turn_instance_id = uuid4().hex
                await self._collaboration.prepare_session_for_turn(
                    thread_id=thread_id,
                    prior_state=prior_state,
                    llm_client=llm_client,
                )
                prior_state = await self._collaboration.get_state(thread_id)
                safety_prior_state = prior_state
                prior_turn_count = turn_count_from_state(prior_state)
                seed_message = user_text.strip() or assistant_text.strip()
                initial_state = self._collaboration.build_turn_initial_state(
                    thread_id=thread_id,
                    message=seed_message,
                    channel=Channel.VOICE,
                    user_id=user_id,
                    installed_skills=None,
                    prior_turn_count=prior_turn_count,
                )
                transition = build_voice_turn_state(
                    VoiceTurnStateInputs(
                        thread_id=thread_id,
                        user_id=user_id,
                        user_text=user_text,
                        assistant_text=assistant_text,
                        outcome=outcome,
                        route=route,
                        response_style=response_style,
                        tool_calls=voice_tool_calls,
                        prior_state=prior_state,
                        initial_state=initial_state,
                        prior_turn_count=prior_turn_count,
                        correlation_hash=correlation_hash,
                        safety_assessment=safety_assessment,
                    )
                )
                state = transition.state
                turn_route = transition.metadata.route
                turn_response_style = transition.metadata.response_style
                if correlation_hash is not None and request_hash is not None:
                    diagnostics = dict(state.get("diagnostics", {}) or {})
                    pending_turns = dict(
                        diagnostics.get("voice_pending_turns", {}) or {}
                    )
                    pending_turns[correlation_hash] = {
                        "request_hash": request_hash,
                        "prior_message_count": len(
                            prior_state.get("transcript", []) or []
                        )
                        if prior_state is not None
                        else 0,
                        "turn_instance_id": turn_instance_id,
                    }
                    diagnostics["voice_pending_turns"] = pending_turns
                    state["diagnostics"] = diagnostics

            retry_turn_instance_id = (
                turn_instance_id
                if pending_turn is not None
                or (correlation_hash is not None and request_hash is not None)
                else None
            )

            async with self._active_session_manager.active_session_mutation(
                thread_id,
                mutation_kind="voice_turn",
            ) as mutation_token:
                post_turn_context = self._collaboration.build_workflow_context(
                    thread_id=thread_id,
                    message=state.get("message", ""),
                    prior_state=prior_state,
                    user_id=user_id,
                    llm_client=llm_client,
                    response_llm_client=llm_client,
                    track_session=False,
                )
                await self._session_lifecycle.complete_successful_turn(
                    thread_id=thread_id,
                    user_message=user_text,
                    final_state=state,
                    workflow_context=post_turn_context,
                    mutation_token=mutation_token,
                    ensure_sdk_turn_recorded=(
                        self._collaboration.ensure_sdk_turn_recorded
                    ),
                    session_transcript_soft_limit=None,
                    capture_safety_event=turn_route == "crisis",
                )
                event_name = (
                    VOICE_SAFETY_INTERRUPTED_TURN_RECORDED
                    if outcome == "safety_interrupted"
                    else VOICE_RESPONSE_FINALIZED
                )
                attributes: dict[str, object] = {
                    "voice_runtime": "openai_realtime",
                    "route": turn_route,
                    "response_style": turn_response_style,
                    "memory_mode": self._memory_mode.value,
                    "resource_lookup_status": state.get("resource_lookup_status"),
                    "tool_call_count": len(voice_tool_calls),
                }
                if outcome != "completed" and correlation_hash is not None:
                    attributes["correlation_hash"] = correlation_hash
                trace_event(event_name, attributes)

                if outcome == "safety_interrupted" and safety_assessment is not None:
                    await capture_crisis_outcome(state, post_turn_context)

            if outcome == "safety_interrupted" and safety_assessment is not None:
                safety_schedule = VoicePostTurnSafetyScheduleResult(
                    scheduled=False,
                    reason="safety_interruption_verified",
                    pending_count=self._post_turn_safety_auditor.pending_count,
                )
            elif outcome == "connection_interrupted":
                safety_schedule = VoicePostTurnSafetyScheduleResult(
                    scheduled=False,
                    reason="connection_interrupted",
                    pending_count=self._post_turn_safety_auditor.pending_count,
                )
            else:
                post_turn_safety_state = cast(AgentState, dict(state))
                if correlation_hash is not None:
                    post_turn_diagnostics = dict(
                        post_turn_safety_state.get("diagnostics", {}) or {}
                    )
                    thread_hash = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
                    post_turn_diagnostics["voice_missed_crisis_audit_id"] = (
                        f"voice-missed-crisis:{thread_hash}:{turn_instance_id}"
                    )
                    post_turn_safety_state["diagnostics"] = post_turn_diagnostics
                safety_schedule = self._post_turn_safety_auditor.schedule_check(
                    VoicePostTurnSafetyCheck(
                        thread_id=thread_id,
                        user_id=user_id,
                        user_text=user_text,
                        realtime_route=turn_route,
                        response_style=turn_response_style,
                        state=post_turn_safety_state,
                        prior_state=(
                            cast(AgentState, dict(safety_prior_state))
                            if safety_prior_state is not None
                            else None
                        ),
                        context=post_turn_context,
                        llm_client=llm_client,
                        turn_instance_id=retry_turn_instance_id,
                    )
                )
            diagnostics = dict(state.get("diagnostics", {}) or {})
            diagnostics["voice_post_turn_safety"] = safety_schedule.as_dict()
            if correlation_hash is not None:
                recorded_hashes = [
                    str(value)
                    for value in diagnostics.get("voice_recorded_turn_hashes", [])
                    if isinstance(value, str)
                ]
                diagnostics["voice_recorded_turn_hashes"] = [
                    *recorded_hashes,
                    correlation_hash,
                ][-_MAX_RECORDED_VOICE_TURN_HASHES:]
                pending_turns = dict(diagnostics.get("voice_pending_turns", {}) or {})
                pending_turns.pop(correlation_hash, None)
                if pending_turns:
                    diagnostics["voice_pending_turns"] = pending_turns
                else:
                    diagnostics.pop("voice_pending_turns", None)
            if correlation_hash is not None and request_hash is not None:
                recorded_requests = dict(
                    diagnostics.get("voice_recorded_turn_requests", {}) or {}
                )
                recorded_requests[correlation_hash] = request_hash
                retained_hashes = set(diagnostics["voice_recorded_turn_hashes"])
                diagnostics["voice_recorded_turn_requests"] = {
                    key: value
                    for key, value in recorded_requests.items()
                    if key in retained_hashes
                }
                receipts = dict(
                    diagnostics.get("voice_recorded_turn_receipts", {}) or {}
                )
                receipts[correlation_hash] = {
                    "request_hash": request_hash,
                    "message_count": len(state.get("transcript", []) or []),
                    "post_turn_safety": safety_schedule.as_dict(),
                    "turn_instance_id": turn_instance_id,
                }
                retained_hashes = set(
                    diagnostics.get("voice_recorded_turn_hashes", []) or []
                )
                diagnostics["voice_recorded_turn_receipts"] = {
                    key: value
                    for key, value in receipts.items()
                    if key in retained_hashes
                }
            state["diagnostics"] = diagnostics
            try:
                await self._state_store.save_state(thread_id, state)
            except Exception as exc:
                logger.warning(
                    "voice turn completion metadata persistence failed",
                    extra={
                        "voice_runtime": "openai_realtime",
                        "outcome": outcome,
                        "route": turn_route,
                        "memory_mode": self._memory_mode.value,
                        "error_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                attributes: dict[str, object] = {
                    "voice_runtime": "openai_realtime",
                    "outcome": outcome,
                    "route": turn_route,
                    "memory_mode": self._memory_mode.value,
                    "error_type": type(exc).__name__,
                }
                if correlation_hash is not None:
                    attributes["correlation_hash"] = correlation_hash
                trace_event(
                    VOICE_TURN_COMPLETION_METADATA_PERSIST_FAILED,
                    attributes,
                )
                await self._state_store.save_state(thread_id, state)
            self._post_turn_safety_auditor.forget_schedule_result(
                thread_id=thread_id,
                turn_instance_id=retry_turn_instance_id,
            )
            return state


def _voice_turn_hash_recorded(state: Mapping[str, Any], correlation_hash: str) -> bool:
    diagnostics = state.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return False
    recorded_hashes = diagnostics.get("voice_recorded_turn_hashes", [])
    return isinstance(recorded_hashes, list) and correlation_hash in recorded_hashes


def _pending_voice_turn(
    state: Mapping[str, Any], correlation_hash: str
) -> Mapping[str, Any] | None:
    diagnostics = state.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return None
    pending_turns = diagnostics.get("voice_pending_turns", {})
    if not isinstance(pending_turns, Mapping):
        return None
    pending_turn = pending_turns.get(correlation_hash)
    return pending_turn if isinstance(pending_turn, Mapping) else None


def _validate_voice_turn_request_hash(
    state: Mapping[str, Any],
    correlation_hash: str,
    request_hash: str,
) -> None:
    diagnostics = state.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return
    recorded_requests = diagnostics.get("voice_recorded_turn_requests", {})
    existing = (
        recorded_requests.get(correlation_hash)
        if isinstance(recorded_requests, Mapping)
        else None
    )
    if existing is None:
        pending_turn = _pending_voice_turn(state, correlation_hash)
        existing = (
            pending_turn.get("request_hash") if pending_turn is not None else None
        )
    if isinstance(existing, str) and existing != request_hash:
        raise ValueError("client_turn_id was already used for a different voice turn")


__all__ = ["VoiceRuntimeFacade"]
