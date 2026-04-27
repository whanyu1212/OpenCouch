"""Therapeutic dispatch node - public routing surface.

The dispatcher uses an LLM-primary classifier with narrow deterministic guards.
Implementation details live under ``agent.therapeutic.dispatch`` so prompt
construction, guard regexes, fallback routing, and graph-node orchestration can
change independently.
"""

from __future__ import annotations

import logging

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.dispatch.classifier import _pick_mode_and_modality_with_llm
from agent.therapeutic.dispatch.constants import (
    CLARIFYING_NODE,
    CLOSING_NODE,
    GUIDED_EXERCISE_NODE,
    PSYCHOEDUCATION_NODE,
    REFLECTIVE_NODE,
    SUPPORTIVE_NODE,
    TECHNIQUE_NODE,
    TherapeuticNodeName,
    _MODE_NODE_MAP,
)
from agent.therapeutic.dispatch.fallback import pick_therapeutic_mode
from agent.therapeutic.dispatch.guards import (
    _active_exercise_modality,
    _has_active_exercise,
    _has_pending_exercise_selection,
    _is_active_exercise_clarification,
    _is_advice_request_without_exercise_consent,
    _is_bare_ack_to_open_question,
    _is_coping_advice_without_exercise_consent,
    _looks_like_pending_exercise_choice,
    _matches_any,
    _message_is_acceptance_of_offer,
    _word_count,
)
from agent.therapeutic.dispatch.regex_catalog import (
    ACCEPTANCE_PATTERNS,
    ANAPHORIC_GUIDANCE_PATTERNS,
    CLARIFYING_MAX_WORD_COUNT,
    CONFUSION_PATTERNS,
    COPING_ADVICE_REQUEST_PATTERNS,
    EXERCISE_CONSENT_PATTERNS,
    EXERCISE_EXIT_PATTERNS,
    EXERCISE_OFFER_PATTERNS,
    EXPLICIT_EXERCISE_REQUEST_PATTERNS,
    INFORMATIONAL_WALKTHROUGH_NOUN_PATTERN,
    INFORMATIONAL_WALKTHROUGH_PATTERN,
    _NOUN_PHRASE_COMPLETERS,
    REFLECTIVE_PATTERNS,
    SELF_REPORT_PATTERNS,
    _TERMINATOR,
    WALKTHROUGH_CONSENT_PATTERN,
    WALKTHROUGH_HOWTO_CONSENT_PATTERN,
    _PROMPT_GUIDED_EXERCISE_TRIGGERS,
    _TRIGGER_LIST_SENTENCE,
    _WALKTHROUGH_NOUNS,
    _format_prompt_trigger_phrases,
    _trigger_to_regex,
)
from agent.therapeutic.dispatch.prompt import (
    build_therapeutic_dispatch_prompt,
    build_therapeutic_dispatch_system_prompt,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ACCEPTANCE_PATTERNS",
    "ANAPHORIC_GUIDANCE_PATTERNS",
    "CLARIFYING_MAX_WORD_COUNT",
    "CLARIFYING_NODE",
    "CLOSING_NODE",
    "CONFUSION_PATTERNS",
    "COPING_ADVICE_REQUEST_PATTERNS",
    "EXERCISE_CONSENT_PATTERNS",
    "EXERCISE_EXIT_PATTERNS",
    "EXERCISE_OFFER_PATTERNS",
    "EXPLICIT_EXERCISE_REQUEST_PATTERNS",
    "GUIDED_EXERCISE_NODE",
    "INFORMATIONAL_WALKTHROUGH_NOUN_PATTERN",
    "INFORMATIONAL_WALKTHROUGH_PATTERN",
    "PSYCHOEDUCATION_NODE",
    "REFLECTIVE_NODE",
    "REFLECTIVE_PATTERNS",
    "SELF_REPORT_PATTERNS",
    "SUPPORTIVE_NODE",
    "TECHNIQUE_NODE",
    "TherapeuticNodeName",
    "WALKTHROUGH_CONSENT_PATTERN",
    "WALKTHROUGH_HOWTO_CONSENT_PATTERN",
    "_MODE_NODE_MAP",
    "_NOUN_PHRASE_COMPLETERS",
    "_PROMPT_GUIDED_EXERCISE_TRIGGERS",
    "_TERMINATOR",
    "_TRIGGER_LIST_SENTENCE",
    "_WALKTHROUGH_NOUNS",
    "_active_exercise_modality",
    "_format_prompt_trigger_phrases",
    "_has_active_exercise",
    "_has_pending_exercise_selection",
    "_is_active_exercise_clarification",
    "_is_advice_request_without_exercise_consent",
    "_is_bare_ack_to_open_question",
    "_is_coping_advice_without_exercise_consent",
    "_looks_like_pending_exercise_choice",
    "_matches_any",
    "_message_is_acceptance_of_offer",
    "_pick_mode_and_modality_with_llm",
    "_trigger_to_regex",
    "_word_count",
    "build_therapeutic_dispatch_prompt",
    "build_therapeutic_dispatch_system_prompt",
    "pick_therapeutic_mode",
    "run_therapeutic_dispatch_node",
]


async def run_therapeutic_dispatch_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[TherapeuticNodeName]:
    """Route the current turn to the correct therapeutic response node.

    Args:
        state: The current agent state.
        runtime: The LangGraph runtime carrying injected dependencies.

    Returns:
        A ``Command`` pointing at the next therapeutic mode node, with any
        required routing or exercise-state updates.
    """

    message = state.get("message", "")
    lowered = message.lower()
    llm_client = runtime.context.llm_client

    def _routing_update(modality: str) -> dict:
        return {"therapeutic_approach": modality}

    def _clear_active_exercise_update(modality: str) -> dict:
        return {
            **_routing_update(modality),
            "exercise_state": {
                "exercise_type": None,
                "exercise_step": None,
                "exercise_modality": None,
                "exercise_selection_options": None,
            },
        }

    exercise_active = _has_active_exercise(state)
    exercise_selection_pending = _has_pending_exercise_selection(state)

    # Honor explicit exercise opt-outs without waiting for the LLM.
    if exercise_active and _matches_any(lowered, EXERCISE_EXIT_PATTERNS):
        logger.debug("therapeutic_dispatch: active-exercise exit override")
        return Command(
            update=_clear_active_exercise_update("none"),
            goto=SUPPORTIVE_NODE,
        )

    if exercise_selection_pending and _looks_like_pending_exercise_choice(message):
        logger.debug("therapeutic_dispatch: pending exercise selection choice")
        existing_modality = state.get("therapeutic_approach") or "none"
        return Command(
            update=_routing_update(existing_modality),
            goto=GUIDED_EXERCISE_NODE,
        )

    if exercise_active and _is_active_exercise_clarification(message):
        logger.debug("therapeutic_dispatch: active-exercise clarification override")
        existing_modality = _active_exercise_modality(state) or "none"
        return Command(
            update=_routing_update(existing_modality),
            goto=CLARIFYING_NODE,
        )

    if (
        not exercise_active
        and not exercise_selection_pending
        and _is_bare_ack_to_open_question(state, message)
    ):
        logger.debug("therapeutic_dispatch: bare acknowledgment needs clarification")
        return Command(
            update=_routing_update("none"),
            goto=CLARIFYING_NODE,
        )

    if llm_client is not None:
        try:
            mode, modality = await _pick_mode_and_modality_with_llm(state, llm_client)
            logger.debug(
                "therapeutic_dispatch: LLM picked mode=%s modality=%s",
                mode,
                modality,
            )

            if exercise_active:
                if mode == "guided_exercise":
                    existing_modality = _active_exercise_modality(state) or modality
                    return Command(
                        update=_routing_update(existing_modality),
                        goto=GUIDED_EXERCISE_NODE,
                    )

                if mode == "clarifying":
                    existing_modality = _active_exercise_modality(state) or modality
                    logger.debug(
                        "therapeutic_dispatch: mid-exercise clarifying "
                        "(exercise state preserved, modality=%s)",
                        existing_modality,
                    )
                    return Command(
                        update=_routing_update(existing_modality),
                        goto=_MODE_NODE_MAP["clarifying"],
                    )

                if mode == "psychoeducation":
                    logger.debug(
                        "therapeutic_dispatch: mid-exercise psychoeducation "
                        "(exercise state preserved, modality=%s)",
                        modality,
                    )
                    return Command(
                        update=_routing_update(modality),
                        goto=_MODE_NODE_MAP["psychoeducation"],
                    )

                logger.debug(
                    "therapeutic_dispatch: LLM exit from active exercise -> %s",
                    mode,
                )
                return Command(
                    update=_clear_active_exercise_update(modality),
                    goto=_MODE_NODE_MAP[mode],
                )

            if mode == "guided_exercise" and _is_coping_advice_without_exercise_consent(
                message
            ):
                logger.debug(
                    "therapeutic_dispatch: guided_exercise advice guard -> "
                    "psychoeducation"
                )
                return Command(
                    update=_routing_update(modality),
                    goto=PSYCHOEDUCATION_NODE,
                )

            if (
                mode == "guided_exercise"
                and _is_advice_request_without_exercise_consent(state, message)
            ):
                logger.debug(
                    "therapeutic_dispatch: anaphoric/walkthrough guidance guard -> "
                    "psychoeducation"
                )
                return Command(
                    update=_routing_update(modality),
                    goto=PSYCHOEDUCATION_NODE,
                )

            return Command(
                update=_routing_update(modality),
                goto=_MODE_NODE_MAP[mode],
            )
        except Exception:
            logger.warning(
                "therapeutic_dispatch LLM classifier failed; falling back to regex.",
                exc_info=True,
            )

    # Without an LLM, active exercises continue unless a deterministic exit fired.
    if exercise_active:
        logger.debug("therapeutic_dispatch: regex fallback - continuing exercise")
        existing_modality = _active_exercise_modality(state) or "none"
        return Command(
            update=_routing_update(existing_modality), goto=GUIDED_EXERCISE_NODE
        )

    mode = pick_therapeutic_mode(message)
    logger.debug("therapeutic_dispatch: regex fallback picked mode=%s", mode)
    fallback_modality = "motivational_interviewing" if mode == "supportive" else "none"
    return Command(
        update=_routing_update(fallback_modality),
        goto=_MODE_NODE_MAP[mode],
    )
