"""Minimal agent graph helpers for the MVP kernel.

The initial execution path should stay small and explicit:
- build initial state
- run crisis gate
- choose normal or crisis response
- return a normalized output
"""

from agent.models import AgentInput, AgentOutput, CrisisAssessment, ResponseKind
from agent.nodes.crisis_gate import run_crisis_gate
from agent.nodes.crisis_response import run_crisis_response
from agent.nodes.therapeutic import run_therapeutic_response
from agent.state import AgentState
from services.llm.base import BaseLLMClient


def build_initial_state(agent_input: AgentInput) -> AgentState:
    """Convert external input into the internal state dictionary.

    This is the first stable contract for the MVP agent.
    """

    return AgentState(
        # Raw user input for the current turn.
        message=agent_input.message,
        # Normalized channel value so nodes do not care how the message arrived.
        channel=agent_input.channel,
        # Reserved for later once auth and persistence wrap the agent kernel.
        user_id=agent_input.user_id,
        session_id=agent_input.session_id,
        # Skills are passed as simple names first; later the loader can resolve them.
        installed_skills=list(agent_input.installed_skills),
        # History is converted into plain serializable data for easy graph/state handling.
        history=[message.model_dump(mode="json") for message in agent_input.history],
        # Empty until memory retrieval exists.
        working_memory=[],
        # Every run starts as safe until the crisis gate updates this field.
        crisis=CrisisAssessment(),
        # Default path is normal support unless the crisis gate redirects it.
        route="therapeutic",
        # Default false because the MVP kernel does not persist on every turn.
        should_persist_memory=False,
    )


def state_to_output(state: AgentState) -> AgentOutput:
    """Normalize graph state into the public agent output contract."""

    return AgentOutput(
        # Response text is filled by whichever node wins the route decision.
        response_text=state.get("response_text", ""),
        # Falls back to therapeutic to keep the output shape valid during early wiring.
        response_kind=state.get("response_kind", ResponseKind.THERAPEUTIC),
        # Crisis assessment is always returned so callers can inspect safety behavior.
        crisis=state["crisis"],
        should_persist_memory=state.get("should_persist_memory", False),
    )


async def run_agent(
    agent_input: AgentInput,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentOutput:
    """Minimal MVP graph entrypoint.

    The real implementation will call concrete nodes, but the contract is already
    useful for tests and a local harness.
    """

    # Step 1: normalize input into graph state.
    state = build_initial_state(agent_input)
    # Step 2: run the hybrid crisis gate before normal conversation.
    state = await run_crisis_gate(
        state,
        llm_client=llm_client,
    )
    # Step 3: choose the correct response path based on the gate result.
    if state["route"] == "crisis":
        state = await run_crisis_response(state)
    else:
        state = await run_therapeutic_response(state)
    # Step 4: convert internal state back into the public result shape.
    return state_to_output(state)
