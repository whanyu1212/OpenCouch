"""Minimal agent graph helpers for the MVP kernel.

The initial execution path should stay small and explicit:
- build initial state
- run crisis gate
- choose normal or crisis response
- return a normalized output
"""

from copy import deepcopy

from agent.context import (
    build_session_summary,
    extract_active_concerns,
    extract_open_loops,
    infer_current_goal,
    trim_history,
    update_session_intent,
)
from agent.models import AgentInput, AgentOutput, Channel, CrisisAssessment, ResponseKind
from agent.nodes.crisis_gate import run_crisis_gate
from agent.nodes.session_stage import update_session_stage
from agent.state import AgentState
from agent.subgraphs import run_crisis_subgraph, run_therapeutic_subgraph
from services.llm.base import BaseLLMClient


def prepare_turn_state(state: AgentState) -> AgentState:
    """Refresh turn-scoped fields from the current message and persisted transcript.

    Args:
        state: The shared state before crisis routing for the current turn.

    Returns:
        The updated state with derived session-context fields refreshed.
    """

    transcript = deepcopy(state.get("transcript", []))
    active_concerns = extract_active_concerns(
        transcript,
        current_message=state["message"],
    )
    current_goal = infer_current_goal(
        transcript,
        current_message=state["message"],
    )
    open_loops = extract_open_loops(
        transcript,
        current_message=state["message"],
    )
    session_intent, session_intent_source = update_session_intent(
        transcript,
        current_message=state["message"],
        existing_intent=state.get("session_intent"),
        existing_source=state.get("session_intent_source"),
    )
    session_summary = build_session_summary(
        transcript,
        current_message=state["message"],
        active_concerns=active_concerns,
        current_goal=current_goal,
    )

    state["channel"] = state.get("channel", Channel.TEST)
    state["installed_skills"] = list(state.get("installed_skills", []))
    state["history"] = trim_history(transcript)
    state["transcript"] = transcript
    state["working_memory"] = list(state.get("working_memory", []))
    state["session_summary"] = session_summary
    state["active_concerns"] = active_concerns
    state["open_loops"] = open_loops
    state["current_goal"] = current_goal
    state["session_intent"] = session_intent
    state["session_intent_source"] = session_intent_source
    state["turn_count"] = sum(1 for turn in transcript if turn.get("role") == "user") + 1
    state["crisis"] = CrisisAssessment()
    state["route"] = "therapeutic"
    state["mode"] = "support"
    state["response_type"] = ResponseKind.THERAPEUTIC
    state["response_text"] = ""
    state["should_persist_memory"] = False
    state["session_stage"] = state.get("session_stage", "opening")
    state["session_stage_source"] = state.get("session_stage_source")
    state["session_stage_reason"] = state.get("session_stage_reason", "")
    return state


def build_initial_state(agent_input: AgentInput) -> AgentState:
    """Convert external input into the internal state dictionary.

    Args:
        agent_input: External user input for the current turn.

    Returns:
        The initialized agent state for the current turn.
    """

    state = AgentState(
        # Raw user input for the current turn.
        message=agent_input.message,
        # Normalized channel value so nodes do not care how the message arrived.
        channel=agent_input.channel,
        # Reserved for later once auth and persistence wrap the agent kernel.
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        # Skills are passed as simple names first; later the loader can resolve them.
        installed_skills=list(agent_input.installed_skills),
        # Transcript keeps the full prior session so persistent threads can resume cleanly.
        transcript=[message.model_dump(mode="json") for message in agent_input.history],
        # Empty until memory retrieval exists.
        working_memory=[],
        # History/session fields are filled by `prepare_turn_state`.
        history=[],
        session_summary="",
        active_concerns=[],
        open_loops=[],
        current_goal=None,
        session_intent=None,
        session_intent_source=None,
        session_stage="opening",
        session_stage_source=None,
        session_stage_reason="",
        turn_count=1,
        # Turn-scoped fields are reset by `prepare_turn_state`.
        crisis=CrisisAssessment(),
        route="therapeutic",
        mode="support",
        should_persist_memory=False,
    )
    return prepare_turn_state(state)


def finalize_turn_state(state: AgentState) -> AgentState:
    """Append the current turn to the durable transcript and prompt window.

    Args:
        state: The shared agent state after the selected response path has completed.

    Returns:
        The updated state with transcript/history including the current turn.
    """

    transcript = deepcopy(state.get("transcript", state["history"]))
    transcript.append({"role": "user", "content": state["message"]})
    transcript.append({"role": "assistant", "content": state.get("response_text", "")})
    state["transcript"] = transcript
    state["history"] = trim_history(transcript)
    return state


def state_to_output(state: AgentState) -> AgentOutput:
    """Normalize graph state into the public agent output contract.

    Args:
        state: Shared agent state after graph execution completes.

    Returns:
        The normalized public agent output.
    """

    return AgentOutput(
        # Response text is filled by whichever node wins the route decision.
        response_text=state.get("response_text", ""),
        # Falls back to therapeutic to keep the output shape valid during early wiring.
        response_type=state.get("response_type", ResponseKind.THERAPEUTIC),
        # Crisis assessment is always returned so callers can inspect safety behavior.
        crisis=state["crisis"],
        mode=state.get("mode"),
        should_persist_memory=state.get("should_persist_memory", False),
    )


async def run_agent(
    agent_input: AgentInput,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentOutput:
    """Minimal MVP graph entrypoint.

    Args:
        agent_input: External input for the current turn.
        llm_client: Optional provider-backed client for classification and generation.

    Returns:
        The normalized public output for the completed turn.
    """

    # Step 1: normalize input into graph state.
    state = build_initial_state(agent_input)
    # Step 2: run the hybrid crisis gate before normal conversation.
    state = await run_crisis_gate(
        state,
        llm_client=llm_client,
    )
    # Step 3: update the current session stage before choosing the response path.
    state = await update_session_stage(state, llm_client=llm_client)
    # Step 4: dispatch into the appropriate subgraph based on the gate result.
    if state["route"] == "crisis":
        state = await run_crisis_subgraph(state, llm_client=llm_client)
    else:
        state = await run_therapeutic_subgraph(state, llm_client=llm_client)
    # Step 5: fold the completed turn back into durable history.
    state = finalize_turn_state(state)
    # Step 6: convert internal state back into the public result shape.
    return state_to_output(state)
