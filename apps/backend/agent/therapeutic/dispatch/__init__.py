"""Therapeutic dispatch — routing the turn to the right response node.

Simplified LLM-primary dispatch. The LLM classifier decides response style
and therapeutic approach; the only non-LLM logic is exercise-state
bookkeeping. Implementation details live in sibling modules.
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
from agent.therapeutic.dispatch.prompt import (
    _PROMPT_GUIDED_EXERCISE_TRIGGERS,
    _format_prompt_trigger_phrases,
    build_therapeutic_dispatch_prompt,
    build_therapeutic_dispatch_system_prompt,
)
from agent.therapeutic.dispatch.router import DispatchPlan, plan_therapeutic_route
from agent.therapeutic.dispatch.routing import (
    _clear_active_exercise_update,
    _command_from_plan,
    _routing_update,
    run_therapeutic_dispatch_node,
)

__all__ = [
    "CLARIFYING_NODE",
    "CLOSING_NODE",
    "DispatchPlan",
    "GUIDED_EXERCISE_NODE",
    "PSYCHOEDUCATION_NODE",
    "REFLECTIVE_NODE",
    "SUPPORTIVE_NODE",
    "TECHNIQUE_NODE",
    "THERAPEUTIC_RESPONSE_NODE",
    "TherapeuticNodeName",
    "_PROMPT_GUIDED_EXERCISE_TRIGGERS",
    "_RESPONSE_STYLE_NODE_MAP",
    "_clear_active_exercise_update",
    "_command_from_plan",
    "_format_prompt_trigger_phrases",
    "_pick_response_style_and_approach_with_llm",
    "_routing_update",
    "build_therapeutic_dispatch_prompt",
    "build_therapeutic_dispatch_system_prompt",
    "plan_therapeutic_route",
    "run_therapeutic_dispatch_node",
]
