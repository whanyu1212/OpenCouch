"""The companion agent."""

from pydantic_ai import Agent
from pydantic_ai.models import Model

from opencouch_agent.agent.deps import CompanionDeps

# Placeholder until the therapeutic prompts are ported from main.
INSTRUCTIONS = """\
You are OpenCouch, a warm and supportive companion. Listen carefully, reflect
what you hear, and keep replies short and conversational. You are not a
therapist and do not diagnose.
"""


def build_companion_agent(model: Model | str) -> Agent[CompanionDeps, str]:
    """Build the companion agent.

    Args:
        model: A Pydantic AI model instance or model string, such as
            `"openai:gpt-5.4-mini"` or `"test"`.

    Returns:
        The configured companion agent.
    """
    return Agent(
        model,
        deps_type=CompanionDeps,
        instructions=INSTRUCTIONS,
        name="companion",
    )
