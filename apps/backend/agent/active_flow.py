"""Turn-level active-flow lifecycle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from agent.state import AgentState, cleared_exercise_state

ActiveFlow = Literal["none", "guided_exercise", "pending_memory_action"]
ActiveFlowAction = Literal["none", "continue", "preserve", "resume", "clear"]

_ACTIVE_FLOWS = {"none", "guided_exercise", "pending_memory_action"}
_ACTIVE_FLOW_ACTIONS = {"none", "continue", "preserve", "resume", "clear"}
_GUIDED_EXERCISE_ACTIONS = {"continue", "preserve", "resume", "clear"}
_PENDING_MEMORY_ACTIONS = {"continue", "clear"}
_PENDING_MEMORY_CONTINUE_ACTION_TYPES = {"confirm_pending", "cancel_pending"}
_EXERCISE_CLEARING_MEMORY_ACTION_TYPES = {"forget_by_index", "forget_by_query"}


@dataclass(frozen=True)
class ActiveFlowDecision:
    """Resolved active-flow lifecycle effect for the current turn."""

    active_flow: ActiveFlow
    action: ActiveFlowAction
    state_delta: dict[str, object]


def detect_active_flow(state: AgentState) -> ActiveFlow:
    """Return the currently active turn-level flow.

    Args:
        state: Current graph state.

    Returns:
        ActiveFlow: The active flow relevant to turn dispatch.
    """

    pending = (state.get("memory_control", {}) or {}).get("pending_action")
    if isinstance(pending, dict):
        return "pending_memory_action"

    exercise_state = state.get("exercise_state", {}) or {}
    if (
        exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    ):
        return "guided_exercise"

    return "none"


def active_flow_summary(state: AgentState) -> str:
    """Return a compact active-flow summary for routing prompts.

    Args:
        state: Current graph state.

    Returns:
        str: Prompt-ready active-flow summary.
    """

    active_flow = detect_active_flow(state)
    if active_flow == "pending_memory_action":
        return f"pending_memory_action: {_pending_action_summary(state)}"
    if active_flow == "guided_exercise":
        exercise_state = state.get("exercise_state", {}) or {}
        exercise_type = _compact_text(exercise_state.get("exercise_type") or "unknown")
        step = exercise_state.get("exercise_step")
        step_id = _compact_text(exercise_state.get("exercise_step_id") or "")
        if step_id:
            return (
                f"guided_exercise: type={exercise_type}; step={step}; step_id={step_id}"
            )
        return f"guided_exercise: type={exercise_type}; step={step}"
    return "none"


def resolve_active_flow_decision(
    state: AgentState,
    *,
    action: ActiveFlowAction,
    route: str,
    memory_action_type: str | None = None,
) -> ActiveFlowDecision:
    """Validate and resolve the active-flow lifecycle effect for a dispatch.

    Args:
        state: Current graph state.
        action: Structured LLM active-flow action.
        route: Structured turn route.
        memory_action_type: Memory-control action when route is memory_control.

    Returns:
        ActiveFlowDecision: Resolved active flow, action, and state delta.

    Raises:
        ValueError: If the route/action combination contradicts current state.
    """

    active_flow = detect_active_flow(state)
    _validate_active_flow_action(
        active_flow=active_flow,
        action=action,
        route=route,
        memory_action_type=memory_action_type,
    )
    return ActiveFlowDecision(
        active_flow=active_flow,
        action=action,
        state_delta=_state_delta_for_action(active_flow=active_flow, action=action),
    )


def current_turn_active_flow(state: AgentState) -> ActiveFlowDecision:
    """Read turn-dispatch active-flow metadata from state diagnostics.

    Args:
        state: Current graph state after turn dispatch.

    Returns:
        ActiveFlowDecision: Diagnostic active-flow metadata with no state delta.
    """

    raw = (state.get("diagnostics") or {}).get("turn_dispatch_active_flow") or {}
    if not isinstance(raw, Mapping):
        return ActiveFlowDecision("none", "none", {})

    active_flow = raw.get("active_flow")
    action = raw.get("action")
    if active_flow not in _ACTIVE_FLOWS or action not in _ACTIVE_FLOW_ACTIONS:
        return ActiveFlowDecision("none", "none", {})
    return ActiveFlowDecision(active_flow, action, {})


def clear_all_active_flows_delta() -> dict[str, object]:
    """Return a delta that clears all turn-level active flows.

    Returns:
        dict[str, object]: State delta clearing exercise and pending memory action.
    """

    return {
        "exercise_state": cleared_exercise_state(),
        "memory_control": {"action": {}, "pending_action": None},
    }


def _validate_active_flow_action(
    *,
    active_flow: ActiveFlow,
    action: ActiveFlowAction,
    route: str,
    memory_action_type: str | None,
) -> None:
    if active_flow == "none":
        if action != "none":
            raise ValueError(
                "Turn dispatch set active_flow_action but no active flow exists."
            )
        return

    if action == "none":
        raise ValueError("Turn dispatch omitted active_flow_action for active flow.")

    if active_flow == "guided_exercise":
        _validate_guided_exercise_action(
            action=action,
            route=route,
            memory_action_type=memory_action_type,
        )
        return

    _validate_pending_memory_action(
        action=action,
        route=route,
        memory_action_type=memory_action_type,
    )


def _validate_guided_exercise_action(
    *,
    action: ActiveFlowAction,
    route: str,
    memory_action_type: str | None,
) -> None:
    if action not in _GUIDED_EXERCISE_ACTIONS:
        raise ValueError(f"Unsupported guided exercise active_flow_action={action!r}.")
    if action in {"continue", "resume"} and route != "therapeutic":
        raise ValueError(
            "Guided exercise continue/resume must route through therapeutic."
        )
    if (
        action == "preserve"
        and route == "memory_control"
        and memory_action_type in _EXERCISE_CLEARING_MEMORY_ACTION_TYPES
    ):
        raise ValueError(
            "Memory deletion requests during an exercise must clear the exercise."
        )


def _validate_pending_memory_action(
    *,
    action: ActiveFlowAction,
    route: str,
    memory_action_type: str | None,
) -> None:
    if action not in _PENDING_MEMORY_ACTIONS:
        raise ValueError(f"Unsupported pending memory active_flow_action={action!r}.")
    if action == "continue":
        if (
            route != "memory_control"
            or memory_action_type not in _PENDING_MEMORY_CONTINUE_ACTION_TYPES
        ):
            raise ValueError(
                "Pending memory continue requires confirm_pending or cancel_pending."
            )
        return
    if route == "memory_control" and (
        memory_action_type in _PENDING_MEMORY_CONTINUE_ACTION_TYPES
    ):
        raise ValueError(
            "Pending memory confirm/cancel must use active_flow_action=continue."
        )


def _state_delta_for_action(
    *,
    active_flow: ActiveFlow,
    action: ActiveFlowAction,
) -> dict[str, object]:
    if active_flow == "guided_exercise" and action == "clear":
        return {"exercise_state": cleared_exercise_state()}
    if active_flow == "pending_memory_action" and action == "clear":
        return {"memory_control": {"pending_action": None}}
    return {}


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


__all__ = [
    "ActiveFlow",
    "ActiveFlowAction",
    "ActiveFlowDecision",
    "active_flow_summary",
    "clear_all_active_flows_delta",
    "current_turn_active_flow",
    "detect_active_flow",
    "resolve_active_flow_decision",
]
