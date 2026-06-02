"""Guided-exercise routing and lifecycle helpers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from agent.runtime.state_ops import apply_state_delta
from agent.runtime.workflow_context import WorkflowContext
from agent.skills.guided_exercises.registry import (
    available_exercise_definitions,
    iter_exercise_selection_aliases,
)
from agent.skills.guided_exercises.engine.state import clear_exercise_delta
from agent.state import AgentState


def guided_exercise_selection_basis(state: Mapping[str, object]) -> str | None:
    exercise_state = state.get("exercise_state", {}) or {}
    active_exercise = (
        isinstance(exercise_state, Mapping)
        and exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    )
    message = str(state.get("message") or "")
    if active_exercise:
        if message_is_operational_side_request(message):
            return None
        if guided_exercise_runtime_action(state) == "preserve":
            return None
        return "active_exercise"
    if message_explicitly_requests_guided_exercise(state, message):
        return "explicit_user_request"
    return None


def guided_exercise_runtime_action(state: Mapping[str, object]) -> str:
    exercise_state = state.get("exercise_state", {}) or {}
    if (
        isinstance(exercise_state, Mapping)
        and exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    ):
        text = normalize_message_text(str(state.get("message") or ""))
        if any(phrase in text for phrase in ("resume", "return to", "back to")) and (
            "exercise" in text or "grounding" in text or "breathing" in text
        ):
            return "resume"
        if any(
            phrase in text
            for phrase in (
                "do you mean",
                "right now or",
                "right now, or",
                "or just around me",
                "what do you mean",
            )
        ):
            return "preserve"
        return "continue"
    return "start"


async def prepare_guided_exercise_route(
    state: AgentState,
    context: WorkflowContext,
    *,
    load_turn_memory: Callable[
        [AgentState, WorkflowContext],
        Awaitable[AgentState],
    ],
) -> tuple[AgentState, bool]:
    """Load turn memory and resolve whether guided exercise should execute."""

    state = await load_turn_memory(state, context)
    exercise_state = state.get("exercise_state", {}) or {}
    has_active_exercise = (
        isinstance(exercise_state, Mapping)
        and exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    )
    turn_lifecycle = state.get("turn_lifecycle", {}) or {}
    lifecycle_action = (
        turn_lifecycle.get("action") if isinstance(turn_lifecycle, Mapping) else None
    )
    lifecycle_metadata = {}
    if isinstance(turn_lifecycle, Mapping):
        for key in (
            "tentative_route",
            "triage_confidence",
            "clarification_needed",
            "clarification_kind",
            "secondary_route",
            "intent_summary",
            "clarification_question",
            "no_clarification_reason",
        ):
            if turn_lifecycle.get(key) is not None:
                lifecycle_metadata[key] = turn_lifecycle[key]
        if (
            turn_lifecycle.get("clarification_needed") is True
            and turn_lifecycle.get("clarification_kind") == "blocking"
        ):
            return state, False
    if has_active_exercise and lifecycle_action == "clear":
        apply_state_delta(state, clear_exercise_delta(state))
        if state.get("route") != "guided_exercise":
            apply_state_delta(
                state,
                {
                    "turn_lifecycle": {
                        "active_flow": "none",
                        "action": "none",
                    }
                },
            )
            return state, False
    if has_active_exercise and lifecycle_action == "preserve":
        apply_state_delta(
            state,
            {
                "route": "therapeutic",
                "response_style": "clarifying",
                "turn_lifecycle": {
                    "active_flow": "guided_exercise",
                    "action": "preserve",
                    **lifecycle_metadata,
                },
            },
        )
        return state, False

    action = guided_exercise_runtime_action(state)
    guided_exercise_basis = guided_exercise_selection_basis(state)
    if guided_exercise_basis is None:
        if action == "preserve":
            apply_state_delta(
                state,
                {
                    "route": "therapeutic",
                    "response_style": "clarifying",
                    "turn_lifecycle": {
                        "active_flow": "guided_exercise",
                        "action": "preserve",
                        **lifecycle_metadata,
                    },
                },
            )
        return state, False
    apply_state_delta(
        state,
        {
            "route": "therapeutic",
            "response_style": "guided_exercise",
            "therapeutic_approach": state.get("therapeutic_approach") or "none",
            "turn_lifecycle": {
                "active_flow": "guided_exercise",
                "action": action,
                **lifecycle_metadata,
            },
            "diagnostics": {
                "openai_guided_exercise_selection_basis": guided_exercise_basis,
            },
        },
    )
    return state, True


def message_is_operational_side_request(message: str) -> bool:
    text = normalize_message_text(message)
    if not text:
        return False
    memory_phrases = (
        "what do you remember",
        "what have you saved",
        "saved memory",
        "memory status",
        "forget",
        "delete memory",
        "delete that memory",
        "remove that memory",
        "turn proactive recall",
    )
    lookup_phrases = (
        "look this up",
        "look that up",
        "look up",
        "search",
        "source",
        "sources",
        "official",
        "current",
        "latest",
        "verify",
    )
    return any(phrase in text for phrase in (*memory_phrases, *lookup_phrases))


def message_explicitly_requests_guided_exercise(
    state: Mapping[str, object],
    message: str,
) -> bool:
    text = normalize_message_text(message)
    if not text:
        return False
    request_phrases = (
        "can we do",
        "could we do",
        "let's do",
        "lets do",
        "start",
        "walk me through",
        "guide me through",
        "take me through",
        "lead me through",
        "help me do",
        "help me with",
        "i need",
        "need a",
    )
    exercise_terms = (
        "exercise",
        "grounding",
        "breathing",
        "box breathing",
        "5 4 3 2 1",
        "5-4-3-2-1",
        "54321",
        "thought record",
        "values",
        "emotion regulation",
    )
    if any(phrase in text for phrase in request_phrases) and any(
        term in text for term in exercise_terms
    ):
        return True
    if "exercise" in text and any(
        phrase in text for phrase in ("do", "try", "start", "practice")
    ):
        return True
    aliases = available_exercise_aliases_for_state(state)
    return (
        bool(aliases)
        and any(phrase in text for phrase in request_phrases)
        and any(alias in text for alias in aliases)
    )


def available_exercise_aliases_for_state(
    state: Mapping[str, object],
) -> tuple[str, ...]:
    try:
        definitions = available_exercise_definitions(
            installed_skills=tuple(state.get("installed_skills") or ()),
            channel=str(state.get("channel") or "text"),
            therapeutic_approach=state.get("therapeutic_approach"),
        )
    except Exception:
        return ()
    aliases: set[str] = set()
    for definition in definitions:
        aliases.add(definition.id.replace("_", " "))
        aliases.add(definition.display_name)
    for alias, _definition in iter_exercise_selection_aliases(definitions=definitions):
        aliases.add(alias)
    return tuple(
        sorted(
            normalized
            for alias in aliases
            if (normalized := normalize_message_text(alias))
        )
    )


def normalize_message_text(message: str) -> str:
    return " ".join(message.casefold().replace("_", " ").split())


__all__ = [
    "available_exercise_aliases_for_state",
    "guided_exercise_runtime_action",
    "guided_exercise_selection_basis",
    "message_explicitly_requests_guided_exercise",
    "message_is_operational_side_request",
    "normalize_message_text",
    "prepare_guided_exercise_route",
]
