"""Therapeutic dispatch node - public routing surface.

The dispatcher uses an LLM-primary classifier with narrow deterministic guards.
Implementation details live under ``agent.therapeutic.dispatch`` so prompt
construction, guard regexes, fallback routing, and graph-node orchestration can
change independently.

Boundary invariant:
``response_style`` is the routing axis and maps to the subgraph node.
``therapeutic_approach`` is prompt context; it shapes how the selected node
responds but must not choose the node. The only special handling is active
guided-exercise continuity, where the pinned exercise approach is reused when
the existing exercise route continues or clarifies.
"""

from __future__ import annotations

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.dispatch.classifier import (
    _pick_response_style_and_approach_with_llm,
)
from agent.therapeutic.dispatch.constants import (
    CLARIFYING_NODE,
    CLOSING_NODE,
    GUIDED_EXERCISE_NODE,
    PSYCHOEDUCATION_NODE,
    REFLECTIVE_NODE,
    SUPPORTIVE_NODE,
    TECHNIQUE_NODE,
    THERAPEUTIC_RESPONSE_NODE,
    TherapeuticNodeName,
    _RESPONSE_STYLE_NODE_MAP,
)
from agent.therapeutic.dispatch.fallback import pick_therapeutic_response_style
from agent.therapeutic.dispatch.guards import (
    _active_exercise_therapeutic_approach,
    _blocks_unconsented_exercise_start,
    _exercise_lifecycle,
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
from agent.therapeutic.dispatch.router import DispatchPlan, plan_therapeutic_route
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


def _routing_update(response_style: str, approach: str) -> dict:
    """Build the therapeutic routing state delta.

    Args:
        response_style: Therapeutic response style selected for this turn.
        approach: Therapeutic approach selected for this turn.

    Returns:
        State delta carrying the selected response style and therapeutic approach.
    """

    return {
        "response_style": response_style,
        "therapeutic_approach": approach,
    }


def _clear_active_exercise_update(response_style: str, approach: str) -> dict:
    """Build a state delta that clears active guided-exercise state.

    Args:
        response_style: Therapeutic response style selected for this turn.
        approach: Therapeutic approach selected for this turn.

    Returns:
        State delta carrying routing metadata and a cleared exercise state.
    """

    return {
        **_routing_update(response_style, approach),
        "exercise_state": {
            "exercise_type": None,
            "exercise_step": None,
            "exercise_therapeutic_approach": None,
            "exercise_selection_options": None,
        },
    }


def _command_from_plan(plan: DispatchPlan) -> Command[TherapeuticNodeName]:
    """Convert a dispatch plan into the LangGraph command.

    Args:
        plan: Internal routing plan.

    Returns:
        LangGraph command for the planned response-style node.
    """

    update = (
        _clear_active_exercise_update(plan.response_style, plan.therapeutic_approach)
        if plan.clear_exercise
        else _routing_update(plan.response_style, plan.therapeutic_approach)
    )
    return Command(
        update=update,
        goto=_RESPONSE_STYLE_NODE_MAP[plan.response_style],
    )


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
    "THERAPEUTIC_RESPONSE_NODE",
    "TherapeuticNodeName",
    "WALKTHROUGH_CONSENT_PATTERN",
    "WALKTHROUGH_HOWTO_CONSENT_PATTERN",
    "DispatchPlan",
    "_RESPONSE_STYLE_NODE_MAP",
    "_NOUN_PHRASE_COMPLETERS",
    "_PROMPT_GUIDED_EXERCISE_TRIGGERS",
    "_TERMINATOR",
    "_TRIGGER_LIST_SENTENCE",
    "_WALKTHROUGH_NOUNS",
    "_active_exercise_therapeutic_approach",
    "_blocks_unconsented_exercise_start",
    "_command_from_plan",
    "_exercise_lifecycle",
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
    "_pick_response_style_and_approach_with_llm",
    "_routing_update",
    "_trigger_to_regex",
    "_word_count",
    "build_therapeutic_dispatch_prompt",
    "build_therapeutic_dispatch_system_prompt",
    "pick_therapeutic_response_style",
    "plan_therapeutic_route",
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
        A ``Command`` pointing at the next therapeutic response node, with any
        required routing or exercise-state updates.
    """

    plan = await plan_therapeutic_route(state, runtime.context.llm_client)
    return _command_from_plan(plan)
