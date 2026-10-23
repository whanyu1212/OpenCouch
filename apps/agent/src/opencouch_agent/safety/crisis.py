"""The crisis responder: the agent that replies whenever the gate says CRISIS.

It has no tools at all, so no coach tool can ever run on a crisis turn, and
its behaviour is all in this file. A stub until HY-230 builds it out (the
resource card, a fixed-text fallback when the model fails, the check-in).
"""

from pydantic_ai import Agent
from pydantic_ai.models import Model

from opencouch_agent.agent import CompanionDeps

# Placeholder until HY-230; the helplines are to be replaced by HY-209's
# vetted catalog.
CRISIS_INSTRUCTIONS = """\
The person you are talking with may be in crisis or at risk of harming
themselves. Stay with them and keep replies short, calm and direct.

- Ask whether they are safe right now.
- If they are in immediate danger, ask them to call 995 now.
- Encourage them to reach a person today: the Samaritans of Singapore 24-hour
  line at 1767, or the National mindline at 1771.
- Never say or imply that you are an emergency service or that someone is
  coming.
- No exercises, advice lists or small talk. Don't end the conversation.
"""


def build_crisis_agent(model: Model | str) -> Agent[CompanionDeps, str]:
    """Build the crisis responder, with no tools.

    Args:
        model: A Pydantic AI model instance or model string.

    Returns:
        The crisis responder. It shares the companion's deps type so it can
        read the same conversation history.
    """
    return Agent(
        model,
        deps_type=CompanionDeps,
        instructions=CRISIS_INSTRUCTIONS,
        name="crisis_responder",
    )
