"""Facade grouping voice-specific runtime methods.

Extracted from ``PersistentAgentRuntime`` to shrink the runtime module and
give the voice surface a clear ownership boundary.  The facade holds a
back-pointer to the runtime for the handful of private orchestration helpers
it still needs (``_context_for_turn``, ``_prepare_session_for_turn``, etc.)
and receives the most-used shared resources via constructor injection.

The locking protocol is identical to the text path: ``_lock_for(thread_id)``
returns the **same** per-thread ``asyncio.Lock`` instance the runtime uses,
and ``active_session_mutation`` nests inside it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

from agent.audit.crisis_log import record_crisis_outcome
from agent.audit.models import CrisisResourceLookupStatus
from agent.memory.modes import MemoryMode
from agent.memory.operations.procedural_profile import aget_procedural_profile
from agent.memory.store import MemoryStore
from agent.models import Channel
from agent.observability.decorators import trace_event, trace_span
from agent.observability.events import (
    RUNTIME_VOICE_SESSION,
    VOICE_CRISIS_RESOURCE_LOOKUP_PERSISTED,
    VOICE_RESPONSE_FINALIZED,
)
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.session import turn_count_from_state
from agent.runtime.session.active_session import ActiveSessionManager
from agent.runtime.state_store import RuntimeStateStore
from agent.state import AgentState, resolve_owner_id
from agent.voice.post_turn_safety import (
    VoicePostTurnSafetyAuditor,
    VoicePostTurnSafetyCheck,
)
from agent.voice.state_transition import VoiceTurnStateInputs, build_voice_turn_state
from llm.base import BaseLLMClient

if TYPE_CHECKING:
    from agent.runtime.runtime import PersistentAgentRuntime

logger = logging.getLogger(__name__)


# ── Module-level helpers (moved from runtime.py) ─────────────────


def _latest_user_text(transcript: list[dict[str, object]]) -> str:
    for turn in reversed(transcript):
        if turn.get("role") == "user":
            return str(turn.get("content") or "").strip()
    return ""


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
        runtime: PersistentAgentRuntime,
        state_store: RuntimeStateStore,
        memory_store: MemoryStore,
        active_session_manager: ActiveSessionManager,
        lock_for: Callable[[str], asyncio.Lock],
        memory_mode: MemoryMode,
    ) -> None:
        self._runtime = runtime
        self._state_store = state_store
        self._memory_store = memory_store
        self._active_session_manager = active_session_manager
        self._lock_for = lock_for
        self._memory_mode = memory_mode
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

    # ── build_voice_tool_context ─────────────────────────────────

    async def build_voice_tool_context(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        current_user_message: str,
        transcript: list[dict[str, object]],
        llm_client: BaseLLMClient | None = None,
    ) -> OpenAITextRunContext:
        """Build the app-owned context used by voice function tools."""

        effective_user_message = current_user_message.strip() or _latest_user_text(
            transcript
        )
        if not effective_user_message:
            effective_user_message = "voice tool call"
        prior_state = await self._runtime.get_state(thread_id)
        prior_turn_count = turn_count_from_state(prior_state)
        initial_state = self._runtime._build_turn_initial_state(
            thread_id=thread_id,
            message=effective_user_message,
            channel=Channel.VOICE,
            user_id=user_id,
            installed_skills=None,
            prior_turn_count=prior_turn_count,
        )
        state = cast(AgentState, {**dict(prior_state or {}), **dict(initial_state)})
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
        workflow_context = self._runtime._context_for_turn(
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
        self._rehydrate_crisis_resource_lookup(context, prior_state)
        return context

    # ── _rehydrate_crisis_resource_lookup ─────────────────────────

    @staticmethod
    def _rehydrate_crisis_resource_lookup(
        context: OpenAITextRunContext,
        prior_state: Mapping[str, Any] | None,
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
        status = prior_state.get("resource_lookup_status")
        if not isinstance(status, str) or status in {"", "not_attempted"}:
            return
        found_resources = prior_state.get("found_resources")
        rows = (
            [dict(row) for row in found_resources]
            if isinstance(found_resources, list)
            else []
        )
        inferred_location = prior_state.get("inferred_location")
        context.record_crisis_resource_tool_result(
            response_text="",
            inferred_location=(
                inferred_location if isinstance(inferred_location, str) else ""
            ),
            found_resources=rows,
            resource_lookup_status=cast(CrisisResourceLookupStatus, status),
        )

    # ── persist_voice_crisis_resource_lookup ──────────────────────

    async def persist_voice_crisis_resource_lookup(
        self,
        *,
        thread_id: str,
        user_id: str | None,
        inferred_location: str,
        found_resources: list[dict[str, str]],
        resource_lookup_status: str,
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
                        self._runtime._build_turn_initial_state(
                            thread_id=thread_id,
                            message="voice tool call",
                            channel=Channel.VOICE,
                            user_id=user_id,
                            installed_skills=None,
                            prior_turn_count=0,
                        )
                    ),
                )
            else:
                state = cast(AgentState, dict(prior_state))
            state["inferred_location"] = inferred_location
            state["found_resources"] = [dict(row) for row in found_resources]
            state["resource_lookup_status"] = resource_lookup_status
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

        initial_state = self._runtime._build_turn_initial_state(
            thread_id=thread_id,
            message="",
            channel=Channel.VOICE,
            user_id=user_id,
            installed_skills=None,
            prior_turn_count=0,
        )
        owner_id = resolve_owner_id(cast(AgentState, initial_state))
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
        route: str | None = None,
        response_style: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        llm_client: BaseLLMClient | None = None,
    ) -> AgentState:
        """Persist a finalized voice turn without running the text agent."""

        voice_tool_calls = list(tool_calls or [])
        async with self._lock_for(thread_id):
            self._runtime._remember_llm_client(thread_id, llm_client)
            prior_state = await self._runtime.get_state(thread_id)
            await self._runtime._prepare_session_for_turn(
                thread_id=thread_id,
                prior_state=prior_state,
                llm_client=llm_client,
            )
            prior_state = await self._runtime.get_state(thread_id)
            prior_turn_count = turn_count_from_state(prior_state)
            seed_message = user_text.strip() or assistant_text.strip()
            initial_state = self._runtime._build_turn_initial_state(
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
                    route=route,
                    response_style=response_style,
                    tool_calls=voice_tool_calls,
                    prior_state=prior_state,
                    initial_state=initial_state,
                    prior_turn_count=prior_turn_count,
                )
            )
            state = transition.state

            async with self._active_session_manager.active_session_mutation(
                thread_id,
                mutation_kind="voice_turn",
            ) as mutation_token:
                await self._runtime._record_successful_turn_tracking(
                    thread_id,
                    state,
                    session_transcript_soft_limit=None,
                )
                await self._state_store.save_state(thread_id, state)
                # Audit after save_state (synthesis happened before it) so the
                # persisted turn and its crisis record can never diverge.
                if transition.metadata.route == "crisis":
                    crisis_context = self._runtime._context_for_turn(
                        thread_id=thread_id,
                        message=state.get("message", ""),
                        prior_state=prior_state,
                        user_id=user_id,
                        llm_client=llm_client,
                        response_llm_client=llm_client,
                        track_session=False,
                    )
                    await record_crisis_outcome(state, crisis_context)
                await self._runtime._ensure_openai_sdk_turn_recorded(
                    thread_id,
                    user_message=user_text,
                    final_state=state,
                )
                await self._active_session_manager.clear_active_session_mutation(
                    thread_id,
                    mutation_token,
                )
                trace_event(
                    VOICE_RESPONSE_FINALIZED,
                    {
                        "voice_runtime": "openai_realtime",
                        "route": transition.metadata.route,
                        "response_style": transition.metadata.response_style,
                        "memory_mode": self._memory_mode.value,
                        "resource_lookup_status": state.get("resource_lookup_status"),
                        "tool_call_count": len(voice_tool_calls),
                    },
                )

        post_turn_context = self._runtime._context_for_turn(
            thread_id=thread_id,
            message=state.get("message", ""),
            prior_state=prior_state,
            user_id=user_id,
            llm_client=llm_client,
            response_llm_client=llm_client,
            track_session=False,
        )
        safety_schedule = self._post_turn_safety_auditor.schedule_check(
            VoicePostTurnSafetyCheck(
                thread_id=thread_id,
                user_id=user_id,
                user_text=user_text,
                realtime_route=transition.metadata.route,
                response_style=transition.metadata.response_style,
                state=cast(AgentState, dict(state)),
                prior_state=(
                    cast(AgentState, dict(prior_state))
                    if prior_state is not None
                    else None
                ),
                context=post_turn_context,
                llm_client=llm_client,
            )
        )
        diagnostics = dict(state.get("diagnostics", {}) or {})
        diagnostics["voice_post_turn_safety"] = safety_schedule.as_dict()
        state["diagnostics"] = diagnostics
        return state


__all__ = [
    "VoiceRuntimeFacade",
    "_compact_voice_memory_context",
    "_latest_user_text",
    "_compact_memory_value",
]
