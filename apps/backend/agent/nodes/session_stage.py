"""Session-stage inference helpers."""

from __future__ import annotations

from typing import Literal

from agent.context import infer_session_stage_deterministically
from agent.state import AgentState
from pydantic import BaseModel
from services.llm.base import BaseLLMClient

StageLabel = Literal["opening", "deepening", "stabilizing", "closing"]


class SessionStageAssessmentSchema(BaseModel):
    """Structured schema for session-stage model output."""

    stage: str
    reason: str


def _recent_modes_from_transcript(state: AgentState, *, limit: int = 4) -> list[str]:
    """Return recent assistant modes inferred from the stored transcript.

    Args:
        state: Shared agent state for the current turn.
        limit: Maximum number of recent mode hints to derive.

    Returns:
        A list of approximate recent mode labels.
    """

    transcript = state.get("transcript", [])
    modes: list[str] = []
    for turn in reversed(transcript):
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "").lower()
        if (
            "grounding" in content
            or "thought record" in content
            or "exercise" in content
        ):
            modes.append("guided_exercise")
        elif "pattern i notice" in content or "pattern" in content:
            modes.append("reflection")
        elif "i can help you talk through" in content:
            modes.append("orientation")
        else:
            modes.append("support")
        if len(modes) >= limit:
            break
    return list(reversed(modes))


def infer_session_stage(
    state: AgentState,
) -> tuple[StageLabel, str, str]:
    """Infer the session stage from deterministic state signals.

    Args:
        state: Shared agent state for the current turn.

    Returns:
        A tuple of `(stage, source, reason)`.
    """

    stage, reason = infer_session_stage_deterministically(
        previous_stage=state.get("session_stage"),
        session_intent=state.get("session_intent"),
        current_message=state["message"],
        turn_count=state["turn_count"],
        recent_modes=_recent_modes_from_transcript(state),
        needs_crisis_response=state["crisis"].needs_crisis_response,
        needs_clarification=state["crisis"].needs_clarification,
    )
    return stage, "deterministic", reason


async def refine_session_stage_with_llm(
    state: AgentState,
    *,
    llm_client: BaseLLMClient,
) -> tuple[StageLabel, str]:
    """Refine the session stage using a provider-backed model call.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Provider-backed client used to assess the session stage.

    Returns:
        A tuple of `(stage, reason)` from the model.
    """

    recent_transcript = "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
        for turn in state.get("transcript", [])[-6:]
        if turn.get("content")
    )
    prompt = (
        "Assess the current stage of this mental health support session.\n\n"
        "Return only the structured schema.\n"
        "Allowed stages:\n"
        "- opening\n"
        "- deepening\n"
        "- stabilizing\n"
        "- closing\n\n"
        "Choose the stage that best matches where the conversation is right now.\n\n"
        f"Session intent: {state.get('session_intent') or '(none)'}\n"
        f"Current goal: {state.get('current_goal') or '(none)'}\n"
        f"Session summary: {state.get('session_summary', '')}\n"
        f"Open loops: {'; '.join(state.get('open_loops', [])) or '(none)'}\n"
        f"Recent transcript:\n{recent_transcript}\n\n"
        f"Current user message:\nuser: {state['message']}"
    )
    raw = await llm_client.generate_structured(
        prompt=prompt,
        response_schema=SessionStageAssessmentSchema,
        system_instruction=(
            "You are classifying the stage of a mental health support session. "
            "Use the allowed stage labels only and prefer stable progression over frequent changes."
        ),
        temperature=0,
    )
    stage = (
        raw.stage
        if raw.stage in {"opening", "deepening", "stabilizing", "closing"}
        else "deepening"
    )
    return stage, raw.reason


async def update_session_stage(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Update the structured session stage for the current turn.

    Args:
        state: Shared agent state for the current turn.
        llm_client: Optional provider-backed client for stage refinement.

    Returns:
        Updated state with stage fields populated.
    """

    stage, source, reason = infer_session_stage(state)
    state["session_stage"] = stage
    state["session_stage_source"] = source
    state["session_stage_reason"] = reason

    if llm_client is None or state["crisis"].needs_crisis_response:
        return state

    try:
        refined_stage, refined_reason = await refine_session_stage_with_llm(
            state,
            llm_client=llm_client,
        )
    except Exception:
        return state

    if state["session_stage"] == "closing":
        return state

    state["session_stage"] = refined_stage
    state["session_stage_source"] = "llm"
    state["session_stage_reason"] = refined_reason
    return state
