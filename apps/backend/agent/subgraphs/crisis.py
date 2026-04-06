"""Crisis subgraph entrypoint."""

from agent.nodes.crisis_response import run_crisis_response
from agent.state import AgentState
from services.llm.base import BaseLLMClient


async def run_crisis_subgraph(
    state: AgentState,
    *,
    llm_client: BaseLLMClient | None = None,
) -> AgentState:
    """Run the crisis response path for the current turn.

    Args:
        state: The shared agent state after crisis-gate routing.
        llm_client: Optional provider-backed client for text generation.

    Returns:
        The updated shared agent state after the crisis path completes.
    """

    return await run_crisis_response(state, llm_client=llm_client)
