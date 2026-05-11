"""Therapeutic dispatch — routing the turn to the right response node.

Simplified LLM-primary dispatch. The LLM classifier decides response style
and therapeutic approach; the only non-LLM logic is exercise-state
bookkeeping. Implementation details live in sibling modules.
"""

from __future__ import annotations

from agent.therapeutic.dispatch.constants import (
    GUIDED_EXERCISE_NODE,
    THERAPEUTIC_RESPONSE_NODE,
    TherapeuticNodeName,
    node_for_response_style,
)
from agent.therapeutic.dispatch.prompt import (
    _PROMPT_GUIDED_EXERCISE_TRIGGERS,
    _format_prompt_trigger_phrases,
    build_therapeutic_dispatch_prompt,
    build_therapeutic_dispatch_system_prompt,
)
from agent.therapeutic.dispatch.node import run_therapeutic_dispatch_node
from agent.therapeutic.dispatch.planner import DispatchPlan, plan_therapeutic_route

__all__ = [
    "DispatchPlan",
    "GUIDED_EXERCISE_NODE",
    "THERAPEUTIC_RESPONSE_NODE",
    "TherapeuticNodeName",
    "_PROMPT_GUIDED_EXERCISE_TRIGGERS",
    "_format_prompt_trigger_phrases",
    "build_therapeutic_dispatch_prompt",
    "build_therapeutic_dispatch_system_prompt",
    "node_for_response_style",
    "plan_therapeutic_route",
    "run_therapeutic_dispatch_node",
]
