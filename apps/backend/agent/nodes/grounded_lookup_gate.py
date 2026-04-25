"""Grounded-lookup routing gate for explicit factual search requests."""

from __future__ import annotations

import re
import time
from typing import Any, Literal

from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.runtime_context import WorkflowContext
from agent.state import AgentState

_LOOKUP_VERB_RE = re.compile(
    r"\b(look up|search(?: for| online| the web)?|web search|google|"
    r"check official|check current|verify(?: whether| if| that)?|"
    r"find (?:official|current|verified|local|nearby|resources|services|"
    r"clinics|directories))\b",
    re.I,
)
_CHECK_IF_RE = re.compile(r"\bcan you check (?:if|whether)\b", re.I)
_CURRENT_INFO_RE = re.compile(
    r"\b(latest|current|up[- ]?to[- ]?date|still available|still works|"
    r"eligibility|official rules?|law|regulation|policy|price|cost|schedule)\b",
    re.I,
)
_THERAPEUTIC_SUBJECTIVE_RE = re.compile(
    r"\b(being unreasonable|overreacting|bad person|wrong for feeling|"
    r"should i feel|why do i feel|what does it mean that i)\b",
    re.I,
)


def _detect_grounded_lookup_action(message: str) -> dict[str, Any] | None:
    """Detect an explicit factual/current lookup request.

    Args:
        message: Current user message.

    Returns:
        A serializable lookup action, or ``None`` for ordinary therapeutic
        routing.
    """

    stripped = message.strip()
    if not stripped:
        return None
    if _THERAPEUTIC_SUBJECTIVE_RE.search(stripped):
        return None

    has_lookup_verb = bool(_LOOKUP_VERB_RE.search(stripped))
    has_current_info = bool(_CURRENT_INFO_RE.search(stripped))
    has_check_if = bool(_CHECK_IF_RE.search(stripped))
    has_numeric_or_url = bool(re.search(r"\d|https?://|www\.", stripped, re.I))
    is_question = stripped.endswith("?") or bool(
        re.match(r"\s*(what|which|where|how|is|are|can|do|does)\b", stripped, re.I)
    )

    if has_lookup_verb or (has_check_if and (has_current_info or has_numeric_or_url)):
        return {"query": stripped}
    if has_current_info and is_question:
        return {"query": stripped}
    return None


async def run_grounded_lookup_gate_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],  # noqa: ARG001 - required node shape
) -> Command[Literal["grounded_answer_node", "load_memory_node"]]:
    """Route explicit factual lookup requests before therapeutic generation.

    Args:
        state: Current graph state after memory-control routing.
        runtime: LangGraph runtime carrying workflow dependencies.

    Returns:
        State update plus the next node to run.
    """

    start = time.monotonic()
    action = _detect_grounded_lookup_action(state.get("message", ""))
    diagnostics = {
        "grounded_lookup_gate_ms": round((time.monotonic() - start) * 1000, 2)
    }

    if action is None:
        return Command(
            update={
                "grounded_lookup_query": "",
                "grounded_lookup_status": "not_attempted",
                "diagnostics": diagnostics,
            },
            goto="load_memory_node",
        )

    return Command(
        update={
            "route": "grounded_lookup",
            "grounded_lookup_query": action["query"],
            "grounded_lookup_status": "not_attempted",
            "diagnostics": diagnostics,
        },
        goto="grounded_answer_node",
    )
