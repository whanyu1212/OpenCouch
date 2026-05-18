"""Therapeutic text agent definition."""

from __future__ import annotations

from typing import Any, Sequence

from agents import Agent

from llm.openai_client import DEFAULT_OPENAI_MODEL
from agent.runtime.agents.common import (
    AgentDefinition,
    build_agent,
    definition_with_instructions,
)
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.tools.grounded import build_grounded_lookup_tools
from agent.runtime.tools.guided_exercise import build_guided_exercise_discovery_tools
from agent.runtime.tools.memory import build_memory_tools
from agent.runtime.tools.therapeutic import build_therapeutic_response_tools


THERAPEUTIC_AGENT_NAME = "OpenCouch therapeutic response agent"

THERAPEUTIC_AGENT_INSTRUCTIONS = """\
You are the default OpenCouch therapeutic text agent for safe, non-crisis turns.
Use concise, grounded, emotionally precise support. Keep product state and tool
results consistent with OpenCouch runtime guidance.

Memory tools:
- Call show_saved_memory only when the user explicitly asks what is saved,
  remembered, or known about them.
- Call show_memory_status only when the user asks whether memory is enabled,
  how much memory exists, or whether proactive recall is on.
- Call mutating memory tools only when the runtime prompt explicitly requires
  the matching action or the user clearly asks to change saved memory.
- Deletion tools preserve OpenCouch confirmation semantics: prepare first, then
  confirm or cancel only when a pending deletion exists.

Grounded lookup:
- Call answer_grounded_lookup only when the user explicitly asks for current,
  factual, official, source-backed, or external resource information.

Guided exercise discovery:
- Call list_guided_exercise_skills only when considering whether to offer a
  structured guided exercise. Use the returned metadata to offer one suitable
  option; do not start or run the exercise yourself.

Therapeutic response skills:
- Call load_therapeutic_response_skill before drafting an ordinary non-crisis
  therapeutic reply when no memory or grounded lookup tool owns the answer.
- Choose the response_style argument that best fits the user's current turn:
  supportive, reflective, clarifying, psychoeducation, closing, or technique.
- Use the returned skill_context as response-style guidance. Do not recite the
  skill name, internal labels, or tool metadata to the user.

Do not claim to own crisis classification or guided-exercise state. Those
remain application-owned.
"""

_THERAPEUTIC_DEFINITION = AgentDefinition(
    name=THERAPEUTIC_AGENT_NAME,
    handoff_description="Default owner for safe OpenCouch therapeutic text replies.",
    instructions=THERAPEUTIC_AGENT_INSTRUCTIONS,
)


def build_therapeutic_agent(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    tools: Sequence[Any] | None = None,
    instructions: str | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build the safe-turn OpenAI therapeutic agent definition."""

    return build_agent(
        definition_with_instructions(_THERAPEUTIC_DEFINITION, instructions),
        model=model,
        tools=(
            tools
            if tools is not None
            else [
                *build_therapeutic_response_tools(),
                *build_memory_tools(),
                *build_guided_exercise_discovery_tools(),
                *build_grounded_lookup_tools(),
            ]
        ),
    )


__all__ = [
    "THERAPEUTIC_AGENT_INSTRUCTIONS",
    "THERAPEUTIC_AGENT_NAME",
    "build_therapeutic_agent",
]
