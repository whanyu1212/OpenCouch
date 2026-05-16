"""Dormant OpenAI Agents SDK definitions for future text runtime slices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from agents import Agent

from llm.openai_client import DEFAULT_OPENAI_MODEL
from agent.text_runtime.openai_agents.context import OpenAITextRunContext
from agent.text_runtime.openai_agents.memory_tools import build_read_only_memory_tools


THERAPEUTIC_AGENT_NAME = "OpenCouch therapeutic text agent"
CRISIS_AGENT_NAME = "OpenCouch crisis response text agent"
GUIDED_EXERCISE_AGENT_NAME = "OpenCouch guided exercise text agent"


THERAPEUTIC_AGENT_INSTRUCTIONS = """\
You are the default OpenCouch therapeutic text agent for safe, non-crisis turns.
Use concise, grounded, emotionally precise support. Keep product state and tool
results consistent with OpenCouch runtime guidance.

Memory tools:
- Call show_saved_memory only when the user explicitly asks what is saved,
  remembered, or known about them.
- Call show_memory_status only when the user asks whether memory is enabled,
  how much memory exists, or whether proactive recall is on.
- Read-only memory tools do not save, delete, or update memory.

Do not claim to own crisis classification, durable memory writes, deletion
confirmation, grounded lookup, or guided-exercise state. Those remain
application-owned until later migration slices attach tested tools or handoffs.
"""


CRISIS_AGENT_INSTRUCTIONS = """\
You are the OpenCouch crisis response specialist. The application runtime must
select you only after its own crisis assessment determines either a level 1
safety clarification turn or a level 2/3 crisis response branch. Do not
classify crisis risk yourself. Provide direct, supportive, safety-oriented
language and follow any runtime-provided resource guidance.
"""


GUIDED_EXERCISE_AGENT_INSTRUCTIONS = """\
You are the OpenCouch guided exercise specialist. The application runtime owns
exercise consent, exercise id validation, step state, exit behavior, and
completion. Follow runtime-provided exercise state exactly and do not invent
unsupported steps or start an exercise without runtime state.
"""


@dataclass(frozen=True)
class OpenAITextAgentRoster:
    """Dormant OpenAI text agent definitions for the migration."""

    therapeutic_agent: Agent[OpenAITextRunContext]
    crisis_agent: Agent[OpenAITextRunContext]
    guided_exercise_agent: Agent[OpenAITextRunContext]


@dataclass(frozen=True)
class _AgentDefinition:
    name: str
    handoff_description: str
    instructions: str


_THERAPEUTIC_DEFINITION = _AgentDefinition(
    name=THERAPEUTIC_AGENT_NAME,
    handoff_description="Default owner for safe OpenCouch therapeutic text replies.",
    instructions=THERAPEUTIC_AGENT_INSTRUCTIONS,
)
_CRISIS_DEFINITION = _AgentDefinition(
    name=CRISIS_AGENT_NAME,
    handoff_description="Owns level 2/3 crisis replies after app-owned assessment.",
    instructions=CRISIS_AGENT_INSTRUCTIONS,
)
_GUIDED_EXERCISE_DEFINITION = _AgentDefinition(
    name=GUIDED_EXERCISE_AGENT_NAME,
    handoff_description=(
        "Owns guided exercise wording after runtime-owned exercise state starts."
    ),
    instructions=GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
)


def _build_agent(
    definition: _AgentDefinition,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    tools: Sequence[Any] | None = None,
) -> Agent[OpenAITextRunContext]:
    return Agent[OpenAITextRunContext](
        name=definition.name,
        handoff_description=definition.handoff_description,
        instructions=definition.instructions,
        model=model,
        tools=list(tools or ()),
    )


def build_therapeutic_agent(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    tools: Sequence[Any] | None = None,
    instructions: str | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build the initial safe-turn OpenAI therapeutic agent definition."""

    definition = (
        _THERAPEUTIC_DEFINITION
        if instructions is None
        else _AgentDefinition(
            name=_THERAPEUTIC_DEFINITION.name,
            handoff_description=_THERAPEUTIC_DEFINITION.handoff_description,
            instructions=instructions,
        )
    )
    return _build_agent(
        definition,
        model=model,
        tools=tools if tools is not None else build_read_only_memory_tools(),
    )


def build_crisis_response_agent(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    instructions: str | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build the crisis response specialist definition."""

    definition = (
        _CRISIS_DEFINITION
        if instructions is None
        else _AgentDefinition(
            name=_CRISIS_DEFINITION.name,
            handoff_description=_CRISIS_DEFINITION.handoff_description,
            instructions=instructions,
        )
    )
    return _build_agent(definition, model=model)


def build_guided_exercise_agent(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    instructions: str | None = None,
) -> Agent[OpenAITextRunContext]:
    """Build the guided exercise specialist definition."""

    definition = (
        _GUIDED_EXERCISE_DEFINITION
        if instructions is None
        else _AgentDefinition(
            name=_GUIDED_EXERCISE_DEFINITION.name,
            handoff_description=_GUIDED_EXERCISE_DEFINITION.handoff_description,
            instructions=instructions,
        )
    )
    return _build_agent(definition, model=model)


def build_openai_text_agent_roster(
    *,
    model: str = DEFAULT_OPENAI_MODEL,
) -> OpenAITextAgentRoster:
    """Build the dormant OpenAI text agent roster for tests and future runtime work."""

    return OpenAITextAgentRoster(
        therapeutic_agent=build_therapeutic_agent(model=model),
        crisis_agent=build_crisis_response_agent(model=model),
        guided_exercise_agent=build_guided_exercise_agent(model=model),
    )
