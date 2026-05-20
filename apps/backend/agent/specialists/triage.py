"""Turn-triage agent definition for OpenCouch text runtime."""

from __future__ import annotations

from agents import Agent

from agent.runtime.context import OpenAITextRunContext
from agent.specialists.common import AgentDefinition, build_agent
from llm.openai_client import DEFAULT_OPENAI_MODEL

TRIAGE_AGENT_NAME = "OpenCouch turn triage agent"

TRIAGE_AGENT_INSTRUCTIONS = """\
You are the OpenCouch turn triage agent. Your job is to return a structured
dispatch decision for the current turn. Do not write user-facing prose. Do not
perform tool calls. The application runtime owns crisis assessment, state
persistence, specialist execution, memory mutation, grounded lookup execution,
and guided-exercise lifecycle.

Choose the primary route contract for this turn:
- therapeutic: ordinary safe therapeutic reply
- memory_control: explicit saved-memory management
- grounded_lookup: explicit factual, current, official, source-backed, or
  external-resource lookup
- guided_exercise: explicit request to start or continue a guided exercise

Also decide the active_flow_action for the current turn:
- none: no active flow implication
- continue: user is continuing the current active flow
- preserve: side-turn that should preserve the current flow
- clear: abandon or exit the current active flow

Use grounded_lookup only when the user explicitly asks for externally verifiable,
official, current, or source-backed information. Use memory_control only when
the user explicitly asks to inspect or change saved memory state. Use
guided_exercise only when the user explicitly asks to start an exercise or when
the current active exercise should continue.

Do not classify crisis risk yourself. Crisis routing remains application-owned
and happens before triage.
"""

_TRIAGE_DEFINITION = AgentDefinition(
    name=TRIAGE_AGENT_NAME,
    handoff_description=(
        "Returns structured per-turn dispatch decisions before specialist execution."
    ),
    instructions=TRIAGE_AGENT_INSTRUCTIONS,
)


def build_triage_agent(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
) -> Agent[OpenAITextRunContext]:
    """Build the OpenAI text triage agent definition."""

    return build_agent(
        _TRIAGE_DEFINITION,
        model=model,
        tools=[],
    )


__all__ = [
    "TRIAGE_AGENT_INSTRUCTIONS",
    "TRIAGE_AGENT_NAME",
    "build_triage_agent",
]
