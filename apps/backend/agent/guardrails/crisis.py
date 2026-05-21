"""OpenAI Agents SDK guardrails for text-agent entrypoints."""

from __future__ import annotations

from typing import Any, cast

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    TResponseInputItem,
    input_guardrail,
)
from pydantic import BaseModel, Field

from agent.guardrails.assessment import assess_crisis_gate
from agent.models import CrisisAssessment
from agent.state import AgentState
from agent.runtime.context import OpenAITextRunContext


class CrisisInputGuardrailOutput(BaseModel):
    """Structured crisis guardrail result stored on SDK run context."""

    assessment: CrisisAssessment
    delta: dict[str, Any] = Field(
        description="OpenCouch state delta produced by the crisis guardrail."
    )


@input_guardrail(name="opencouch_crisis_input_guardrail", run_in_parallel=False)
async def crisis_input_guardrail(
    ctx: RunContextWrapper[OpenAITextRunContext],
    agent: Agent[Any],
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Classify crisis risk before any normal text-agent work starts."""

    run_context = ctx.context
    state = _guardrail_state(run_context, input)
    gate_result = await assess_crisis_gate(
        state,
        llm_client=run_context.workflow_context.llm_client,
    )
    output = CrisisInputGuardrailOutput(
        assessment=gate_result.assessment,
        delta=gate_result.delta,
    )
    run_context.crisis_guardrail_output = output
    run_context.crisis_guardrail_triggered = _should_trip(gate_result.assessment)
    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=run_context.crisis_guardrail_triggered,
    )


async def run_crisis_input_guardrail(
    *,
    agent: Agent[Any],
    input_text: str,
    context: OpenAITextRunContext,
) -> CrisisInputGuardrailOutput:
    """Run the SDK crisis input guardrail as the text-runtime entry check."""

    result = await crisis_input_guardrail.run(
        agent,
        input_text,
        RunContextWrapper(context),
    )
    output = result.output.output_info
    if isinstance(output, CrisisInputGuardrailOutput):
        return output
    return cast(CrisisInputGuardrailOutput, output)


def _should_trip(assessment: CrisisAssessment) -> bool:
    return (
        assessment.needs_crisis_response
        or assessment.needs_clarification
        or assessment.level >= 1
    )


def _guardrail_state(
    context: OpenAITextRunContext,
    input_value: str | list[TResponseInputItem],
) -> AgentState:
    if context.agent_state is not None:
        return context.agent_state
    return cast(
        AgentState,
        {
            "message": context.current_user_message or _input_text(input_value),
            "transcript": [dict(turn) for turn in context.transcript],
            "user_id": context.user_id,
            "session_id": context.session_id,
            "channel": context.channel,
            "diagnostics": {},
        },
    )


def _input_text(input_value: str | list[TResponseInputItem]) -> str:
    if isinstance(input_value, str):
        return input_value
    chunks: list[str] = []
    for item in input_value:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)
    return "\n".join(chunks)


__all__ = [
    "CrisisInputGuardrailOutput",
    "crisis_input_guardrail",
    "run_crisis_input_guardrail",
]
