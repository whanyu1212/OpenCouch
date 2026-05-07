"""Therapeutic dispatch — routing the turn to the right response node.

The package is the public surface for therapeutic dispatch. The
high-level entry points (``run_therapeutic_dispatch_node``, the routing
constants, the prompt builders) are re-exported here so callers can
import from ``agent.therapeutic.dispatch`` directly. Implementation
details (classifier, guards, regex catalog, fallback, plan composition)
live in sibling modules and are accessible by deeper imports when
needed.
"""

from __future__ import annotations

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
from agent.therapeutic.dispatch.prompt import (
    build_therapeutic_dispatch_prompt,
    build_therapeutic_dispatch_system_prompt,
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
    REFLECTIVE_PATTERNS,
    SELF_REPORT_PATTERNS,
    WALKTHROUGH_CONSENT_PATTERN,
    WALKTHROUGH_HOWTO_CONSENT_PATTERN,
    _NOUN_PHRASE_COMPLETERS,
    _PROMPT_GUIDED_EXERCISE_TRIGGERS,
    _TERMINATOR,
    _TRIGGER_LIST_SENTENCE,
    _WALKTHROUGH_NOUNS,
    _format_prompt_trigger_phrases,
    _trigger_to_regex,
)
from agent.therapeutic.dispatch.router import DispatchPlan, plan_therapeutic_route
from agent.therapeutic.dispatch.routing import (
    _clear_active_exercise_update,
    _command_from_plan,
    _routing_update,
    run_therapeutic_dispatch_node,
)

__all__ = [
    "ACCEPTANCE_PATTERNS",
    "ANAPHORIC_GUIDANCE_PATTERNS",
    "CLARIFYING_MAX_WORD_COUNT",
    "CLARIFYING_NODE",
    "CLOSING_NODE",
    "CONFUSION_PATTERNS",
    "COPING_ADVICE_REQUEST_PATTERNS",
    "DispatchPlan",
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
    "_NOUN_PHRASE_COMPLETERS",
    "_PROMPT_GUIDED_EXERCISE_TRIGGERS",
    "_RESPONSE_STYLE_NODE_MAP",
    "_TERMINATOR",
    "_TRIGGER_LIST_SENTENCE",
    "_WALKTHROUGH_NOUNS",
    "_active_exercise_therapeutic_approach",
    "_blocks_unconsented_exercise_start",
    "_clear_active_exercise_update",
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
