"""The companion agent."""

from importlib import resources
from typing import Final

from pydantic_ai import Agent
from pydantic_ai.models import Model

from opencouch_agent.agent.deps import CompanionDeps

# An interim coach prompt for the demo (HY-351), until HY-215 builds the
# coach's instructions from program state. Its three skills are demo-only until
# a clinician reviews them. It names 995 as the one number the coach may give,
# when a life is in danger; every other helpline comes from the crisis path.
INSTRUCTIONS: Final = (
    resources.files(__package__).joinpath("coach_prompt.md").read_text(encoding="utf-8")
)

# Added to a turn whose message the safety check couldn't screen (HY-229).
UNSCREENED_INSTRUCTIONS: Final = """\
The safety check that reads each message for risk couldn't run on this \
conversation's latest messages, so no other part of the app will step in this \
turn. Read them yourself with extra care for any sign of risk to the person \
or anyone else. If you see one, follow the safety rules above yourself: say \
what you heard, ask directly whether they are safe right now, stay with them, \
and if a life may be in danger tell them to call 995 now. Don't say that help \
or helplines are on their way, and don't mention the check."""


def build_companion_agent(model: Model | str) -> Agent[CompanionDeps, str]:
    """Build the companion agent.

    Args:
        model: A Pydantic AI model instance or model string, such as
            `"openai:gpt-6.1-sol"` or `"test"`.

    Returns:
        The configured companion agent.
    """
    return Agent(
        model,
        deps_type=CompanionDeps,
        instructions=INSTRUCTIONS,
        name="companion",
    )
