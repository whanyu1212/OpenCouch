"""LLM-primary turn dispatch policy for non-crisis turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.conversation import format_recent_history
from agent.gates.memory_control.actions import MemoryControlAction
from agent.state import AgentState
from llm.base import BaseLLMClient


TurnRoute = Literal["memory_control", "grounded_lookup", "therapeutic"]
MemoryControlActionType = Literal[
    "list",
    "status",
    "set_recall",
    "forget_by_index",
    "forget_by_query",
    "save_preference",
    "confirm_pending",
    "cancel_pending",
]


class TurnDispatchDecision(BaseModel):
    """Structured output for safe-turn dispatch."""

    route: TurnRoute = Field(
        description=(
            "Exactly one destination for the current safe turn: memory_control, "
            "grounded_lookup, or therapeutic."
        )
    )
    reasoning: str = Field(min_length=1, max_length=360)
    confidence: Literal["low", "medium", "high"]
    memory_action_type: MemoryControlActionType | None = Field(
        default=None,
        description="Required when route is memory_control.",
    )
    enabled: bool | None = Field(
        default=None,
        description="Desired proactive recall state for set_recall.",
    )
    target_kind: Literal["fact", "session", "rule"] | None = Field(
        default=None,
        description="Saved-memory kind for forget_by_index.",
    )
    target_index: int | None = Field(
        default=None,
        ge=1,
        description="One-based saved-memory index for forget_by_index.",
    )
    query: str | None = Field(
        default=None,
        description="Saved-memory deletion query or grounded lookup query.",
    )
    preference_text: str | None = Field(
        default=None,
        description="User preference phrase for save_preference.",
    )


@dataclass(frozen=True)
class TurnDispatchPlan:
    """Resolved graph dispatch plan."""

    route: TurnRoute
    reason: str
    confidence: str
    memory_action: MemoryControlAction | None = None
    grounded_lookup_query: str | None = None
    clear_pending_action: bool = False


def _compact_text(value: Any, *, max_chars: int = 280) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 1].rstrip()}..."


def _pending_action_summary(state: AgentState) -> str:
    pending = (state.get("memory_control", {}) or {}).get("pending_action")
    if not isinstance(pending, dict):
        return "(none)"

    pending_type = _compact_text(pending.get("type") or "unknown")
    target = pending.get("target")
    if not isinstance(target, dict):
        return f"type={pending_type}"

    kind = _compact_text(target.get("kind") or "memory")
    preview = _compact_text(target.get("preview") or "")
    if not preview:
        return f"type={pending_type}; target_kind={kind}"
    return f"type={pending_type}; target_kind={kind}; target_preview={preview}"


def build_turn_dispatch_prompt(state: AgentState) -> str:
    """Build the structured dispatch prompt for one safe turn.

    Args:
        state (AgentState): Current graph state.

    Returns:
        str: Prompt asking the model to choose one safe-turn destination.
    """

    recent_history = format_recent_history(state, limit=8, empty="(none)")
    pending_summary = _pending_action_summary(state)
    return (
        "Choose the one graph destination for the current non-crisis user turn.\n\n"
        "Destinations:\n"
        "- memory_control: explicit management of saved assistant memory only. "
        "Examples: show/list saved memories, memory status, turn proactive recall "
        "on/off, save a response-style preference, delete/forget a saved memory, "
        "ask what is remembered about a topic, or confirm/cancel a pending "
        "saved-memory deletion.\n"
        "- grounded_lookup: the user asks for external factual, current, official, "
        "research, evidence, price, eligibility, schedule, local resource, URL, "
        "product, service, credible article, reading-list, or named-resource "
        "information that should be verified outside the conversation.\n"
        "- therapeutic: emotional support, reflection, coaching, relationship "
        "advice, ordinary conversation, exercises, or subjective reassurance.\n\n"
        "Important boundaries:\n"
        "- Do not route ordinary references to memory, memories, or past events to "
        "memory_control unless the user is asking to inspect or modify saved "
        "assistant memory. 'Do you remember I ...?' is usually therapeutic when "
        "the user is asking for support around a remembered topic. 'What do you "
        "remember about ...?' or 'what is saved in memory about ...?' MUST be "
        "memory_control because the user is inspecting saved assistant memory, "
        "even when the topic is emotional or personal.\n"
        "- For memory_control actions, use list when the user asks what content "
        "is saved, remembered, or known about a topic. Use status only for "
        "memory-system status, counts, or proactive recall settings.\n"
        "- Requests to find credible articles, reading lists, named external "
        "resources, official pages, or source-backed mental-health information "
        "MUST be grounded_lookup. General coaching or explanation without a "
        "source/resource request can stay therapeutic.\n"
        "- If a pending memory deletion exists, route to memory_control only when "
        "the current user message confirms or cancels that pending deletion. If "
        "the user clearly confirms or cancels the deletion, route to "
        "memory_control even if the message includes another request. If the user "
        "moves on without clearly confirming or canceling, choose the appropriate "
        "non-memory route.\n"
        "- If unsure, choose therapeutic.\n\n"
        "When route=memory_control, set memory_action_type and the required fields:\n"
        "- list/status/confirm_pending/cancel_pending: no extra fields.\n"
        "- set_recall: enabled=true or false.\n"
        "- forget_by_index: target_kind and target_index.\n"
        "- forget_by_query: query with the concrete memory target.\n"
        "- save_preference: preference_text as the user's response or memory-use "
        "preference, not a finalized rule. Examples: 'direct answers when I am "
        "spiraling', 'do not suggest journaling', 'ask fewer questions'.\n\n"
        "When route=grounded_lookup, set query to a concise search query.\n\n"
        f"Pending memory action: {pending_summary}\n\n"
        "Recent conversation:\n"
        f"{recent_history}\n\n"
        f'Current user message: "{state.get("message", "")}"'
    )


def build_turn_dispatch_system_prompt() -> str:
    """Build the system instruction for turn dispatch.

    Returns:
        str: System instruction for structured turn routing.
    """

    return (
        "You are a strict graph router for a mental-health support application. "
        "Return only the structured routing decision. Do not answer the user."
    )


def _required_text(value: str | None, *, field_name: str) -> str:
    text = " ".join((value or "").strip().split())
    if not text:
        raise ValueError(f"Turn dispatch selected a route without {field_name}.")
    return text


def _memory_action_from_decision(
    decision: TurnDispatchDecision,
) -> MemoryControlAction:
    action_type = decision.memory_action_type
    if action_type is None:
        raise ValueError("Turn dispatch selected memory_control without an action.")

    if action_type in {"list", "status", "confirm_pending", "cancel_pending"}:
        return MemoryControlAction({"type": action_type})

    if action_type == "set_recall":
        if decision.enabled is None:
            raise ValueError("Turn dispatch selected set_recall without enabled.")
        return MemoryControlAction({"type": "set_recall", "enabled": decision.enabled})

    if action_type == "forget_by_index":
        if decision.target_kind is None or decision.target_index is None:
            raise ValueError(
                "Turn dispatch selected forget_by_index without target kind/index."
            )
        return MemoryControlAction(
            {
                "type": "forget_by_index",
                "target_kind": decision.target_kind,
                "target_index": decision.target_index,
            }
        )

    if action_type == "forget_by_query":
        query = _required_text(decision.query, field_name="query")
        if query.lower() in {"this", "that", "it"}:
            raise ValueError(
                "Turn dispatch selected forget_by_query with a vague target."
            )
        return MemoryControlAction({"type": "forget_by_query", "query": query})

    if action_type == "save_preference":
        preference_text = _required_text(
            decision.preference_text,
            field_name="preference_text",
        )
        return MemoryControlAction(
            {"type": "save_preference", "preference_text": preference_text}
        )

    raise ValueError(f"Unsupported memory action type: {action_type}")


def _plan_from_decision(
    state: AgentState,
    decision: TurnDispatchDecision,
) -> TurnDispatchPlan:
    pending_action = (state.get("memory_control", {}) or {}).get("pending_action")
    clear_pending_action = bool(pending_action) and decision.route != "memory_control"

    if decision.route == "memory_control":
        return TurnDispatchPlan(
            route=decision.route,
            reason=decision.reasoning,
            confidence=decision.confidence,
            memory_action=_memory_action_from_decision(decision),
            clear_pending_action=False,
        )

    if decision.route == "grounded_lookup":
        return TurnDispatchPlan(
            route=decision.route,
            reason=decision.reasoning,
            confidence=decision.confidence,
            grounded_lookup_query=_required_text(decision.query, field_name="query"),
            clear_pending_action=clear_pending_action,
        )

    return TurnDispatchPlan(
        route=decision.route,
        reason=decision.reasoning,
        confidence=decision.confidence,
        clear_pending_action=clear_pending_action,
    )


async def plan_turn_route(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None,
) -> TurnDispatchPlan:
    """Plan the next graph route for a safe user turn.

    Args:
        state (AgentState): Current graph state.
        llm_client (BaseLLMClient | None): Required LLM client for routing.

    Returns:
        TurnDispatchPlan: Resolved graph destination and turn-scoped payloads.

    Raises:
        RuntimeError: If no LLM client is configured.
        ValueError: If the structured decision is missing required route payload.
    """

    if llm_client is None:
        raise RuntimeError("Turn dispatch requires an LLM client.")

    decision = await llm_client.generate_structured(
        prompt=build_turn_dispatch_prompt(state),
        response_schema=TurnDispatchDecision,
        system_instruction=build_turn_dispatch_system_prompt(),
    )
    return _plan_from_decision(state, decision)


__all__ = [
    "TurnDispatchDecision",
    "TurnDispatchPlan",
    "build_turn_dispatch_prompt",
    "build_turn_dispatch_system_prompt",
    "plan_turn_route",
]
