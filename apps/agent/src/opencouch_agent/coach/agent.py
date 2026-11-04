"""The coach agent."""

from pydantic_ai import Agent
from pydantic_ai.models import Model

from opencouch_agent.coach.deps import CoachDeps
from opencouch_agent.coach.instructions import coach_instructions


def build_coach_agent(model: Model | str) -> Agent[CoachDeps, str]:
    """Build the coach agent, with instructions built per turn from `CoachDeps`.

    Args:
        model: A Pydantic AI model instance or model string, such as
            `"openai:gpt-6.1-sol"` or `"test"`.

    Returns:
        The configured coach agent.
    """
    return Agent(
        model,
        deps_type=CoachDeps,
        instructions=coach_instructions,
        name="coach",
    )
