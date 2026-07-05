"""Shared OpenAI text-runtime test fakes."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from agents.tool_context import ToolContext

from tests.support.persistence import FakeCrossRestartLLM


class FakeOpenAISDKRunner:
    """Deterministic Agents SDK runner fake for text-runtime tests."""

    def __init__(
        self,
        final_output: str = "openai reply",
        *,
        invoke_required_tool: bool = False,
        tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
        tool_response_as_final: bool = False,
    ) -> None:
        self.final_output = final_output
        self.invoke_required_tool = invoke_required_tool
        self.tool_calls = list(tool_calls or [])
        self.tool_response_as_final = tool_response_as_final
        self.run_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.triage_calls: list[dict[str, Any]] = []

    async def run(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
        session: Any | None = None,
    ) -> SimpleNamespace:
        if _agent_output_type_name(agent) == "TurnDispatchDecision":
            call = {
                "agent": agent,
                "input_text": input_text,
                "context": context,
                "session": session,
            }
            self.triage_calls.append(call)
            return SimpleNamespace(
                final_output=await _generate_structured_agent_output(
                    agent,
                    input_text,
                    context,
                )
            )
        self.run_calls.append(
            {
                "agent": agent,
                "input_text": input_text,
                "context": context,
                "session": session,
            }
        )
        tool_result = None
        for tool_name, arguments in self.tool_calls:
            tool_result = await _invoke_named_tool(
                agent,
                context,
                tool_name,
                arguments,
            )
        if self.invoke_required_tool:
            tool_result = await _invoke_required_tool(agent, input_text, context)
        final_output = self.final_output
        if self.tool_response_as_final and tool_result is not None:
            final_output = str(getattr(tool_result, "response_text", final_output))
        return SimpleNamespace(final_output=final_output)

    async def run_triage(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
    ) -> SimpleNamespace:
        call = {
            "agent": agent,
            "input_text": input_text,
            "context": context,
            "session": None,
        }
        self.triage_calls.append(call)
        return SimpleNamespace(
            final_output=await _generate_structured_agent_output(
                agent,
                input_text,
                context,
            )
        )

    def run_streamed(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
        session: Any | None = None,
    ) -> "FakeOpenAIStream":
        self.stream_calls.append(
            {
                "agent": agent,
                "input_text": input_text,
                "context": context,
                "session": session,
            }
        )
        return FakeOpenAIStream(
            self.final_output,
            agent=agent,
            context=context,
            tool_calls=self.tool_calls,
            tool_response_as_final=self.tool_response_as_final,
        )


class FakeOpenAIStream:
    """Deterministic streaming result fake for the Agents SDK."""

    def __init__(
        self,
        final_output: str,
        *,
        agent: Any,
        context: Any,
        tool_calls: list[tuple[str, dict[str, Any]]],
        tool_response_as_final: bool,
    ) -> None:
        self.final_output = final_output
        self._agent = agent
        self._context = context
        self._tool_calls = list(tool_calls)
        self._tool_response_as_final = tool_response_as_final

    async def stream_events(self) -> AsyncIterator[SimpleNamespace]:
        tool_result = None
        for tool_name, arguments in self._tool_calls:
            tool_result = await _invoke_named_tool(
                self._agent,
                self._context,
                tool_name,
                arguments,
            )
        if self._tool_response_as_final and tool_result is not None:
            self.final_output = str(
                getattr(tool_result, "response_text", self.final_output)
            )
        yield SimpleNamespace(
            type="raw_response_event",
            data=SimpleNamespace(
                type="response.output_text.delta",
                delta=self.final_output,
            ),
        )


def _required_tool_name(input_text: str) -> str | None:
    for tool_name in (
        "show_saved_memory",
        "show_memory_status",
        "set_proactive_memory_recall",
        "save_response_preference",
        "prepare_memory_deletion_by_index",
        "prepare_memory_deletion_by_query",
        "confirm_memory_deletion",
        "cancel_memory_deletion",
        "answer_grounded_lookup",
        "lookup_crisis_resources",
        "load_guided_exercise_skill",
    ):
        if f"Required tool: {tool_name}" in input_text:
            return tool_name
    return None


def _required_tool_arguments(input_text: str) -> str:
    marker = "Required tool arguments: "
    for line in input_text.splitlines():
        if line.startswith(marker):
            return json.dumps(json.loads(line.removeprefix(marker)))
    return "{}"


async def _invoke_required_tool(
    agent: Any, input_text: str, context: Any
) -> Any | None:
    tool_name = _required_tool_name(input_text)
    if tool_name is None:
        return None
    arguments = json.loads(_required_tool_arguments(input_text))
    return await _invoke_named_tool(agent, context, tool_name, arguments)


async def _invoke_named_tool(
    agent: Any,
    context: Any,
    tool_name: str,
    arguments: dict[str, Any],
) -> Any:
    payload = json.dumps(arguments)
    for tool in getattr(agent, "tools", []):
        if getattr(tool, "name", None) != tool_name:
            continue
        return await tool.on_invoke_tool(
            ToolContext(
                context,
                tool_name=tool.name,
                tool_call_id=f"call-{tool.name}",
                tool_arguments=payload,
            ),
            payload,
        )
    raise AssertionError(f"Required tool {tool_name!r} was not attached to agent.")


def _agent_output_type_name(agent: Any) -> str:
    output_type = getattr(agent, "output_type", None)
    return str(getattr(output_type, "__name__", ""))


async def _generate_structured_agent_output(
    agent: Any,
    input_text: str,
    context: Any,
) -> Any:
    output_type = getattr(agent, "output_type")
    workflow_context = getattr(context, "workflow_context")
    llm_client = workflow_context.llm_client
    return await llm_client.generate_structured(
        prompt=input_text,
        response_schema=output_type,
        system_instruction=getattr(agent, "instructions", None),
    )


class ScriptedOpenAITextRouteLLM(FakeCrossRestartLLM):
    """Fake control LLM for exercising OpenAI text-runtime route slices."""

    def __init__(
        self,
        *,
        route: str,
        crisis_level: int = 0,
        memory_reference_mode: str = "none",
        active_flow_action: str = "none",
        therapeutic_response_style: str = "supportive",
        therapeutic_approach: str = "none",
        exercise_start_basis: str = "ambiguous_or_none",
        exercise_type: str = "grounding_5_4_3_2_1",
        exercise_step_state: str = "complete",
        query: str = "saved memory",
        grounded_answer: str = "Official answer.\n\nSources:\n- Official source",
        grounded_status: str = "answered",
        crisis_location_status: str = "provided",
        crisis_location: str = "Singapore",
        crisis_resource_status: str = "found",
        triage_confidence: str = "high",
        clarification_needed: bool = False,
        clarification_kind: str = "none",
        secondary_route: str | None = None,
        intent_summary: str = "",
        clarification_question: str = "",
        no_clarification_reason: str = "none",
    ) -> None:
        super().__init__()
        self.route = route
        self.crisis_level = crisis_level
        self.memory_reference_mode = memory_reference_mode
        self.active_flow_action = active_flow_action
        self.therapeutic_response_style = therapeutic_response_style
        self.therapeutic_approach = therapeutic_approach
        self.exercise_start_basis = exercise_start_basis
        self.exercise_type = exercise_type
        self.exercise_step_state = exercise_step_state
        self.query = query
        self.grounded_answer = grounded_answer
        self.grounded_status = grounded_status
        self.crisis_location_status = crisis_location_status
        self.crisis_location = crisis_location
        self.crisis_resource_status = crisis_resource_status
        self.triage_confidence = triage_confidence
        self.clarification_needed = clarification_needed
        self.clarification_kind = clarification_kind
        self.secondary_route = secondary_route
        self.intent_summary = intent_summary
        self.clarification_question = clarification_question
        self.no_clarification_reason = no_clarification_reason
        self.structured_prompts: list[tuple[str, str]] = []

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        schema_name = response_schema.__name__
        self.structured_prompts.append((schema_name, prompt))
        if schema_name == "CrisisAssessmentSchema":
            return response_schema(
                level=self.crisis_level,
                confidence="high",
                reason="scripted crisis verdict",
                needs_crisis_response=self.crisis_level >= 2,
                needs_clarification=self.crisis_level == 1,
            )
        if schema_name == "TurnDispatchDecision":
            kwargs: dict[str, Any] = {
                "route": self.route,
                "active_flow_action": self.active_flow_action,
                "reasoning": f"scripted {self.route} route",
                "confidence": self.triage_confidence,
                "clarification_needed": self.clarification_needed,
                "clarification_kind": self.clarification_kind,
                "secondary_route": self.secondary_route,
                "intent_summary": self.intent_summary,
                "clarification_question": self.clarification_question,
                "no_clarification_reason": self.no_clarification_reason,
                "memory_reference_mode": self.memory_reference_mode,
                "therapeutic_approach": self.therapeutic_approach,
            }
            if self.therapeutic_response_style != "guided_exercise":
                kwargs["therapeutic_response_style"] = self.therapeutic_response_style
            if self.route == "grounded_lookup":
                kwargs["query"] = "grounded query"
            return response_schema(**kwargs)
        if schema_name == "DispatchDecision":
            return response_schema(
                response_style=self.therapeutic_response_style,
                therapeutic_approach=self.therapeutic_approach,
                exercise_start_basis=self.exercise_start_basis,
                reasoning="scripted therapeutic response style",
                confidence="high",
            )
        if schema_name == "ExerciseSelectionDecision":
            return response_schema(
                exercise_type=self.exercise_type,
                reasoning="scripted exercise selection",
                confidence="high",
            )
        if schema_name == "ExerciseStepDecision":
            return response_schema(
                step_state=self.exercise_step_state,
                reasoning="scripted exercise step state",
                confidence="high",
            )
        if schema_name == "PreferenceRuleDecision":
            return response_schema(
                rule_text="You prefer direct answers when you are spiraling.",
                reasoning="scripted preference rule",
                confidence="high",
            )
        if schema_name == "LookupPreflightDecision":
            return response_schema(
                status="search",
                search_query="grounded query",
                answer="",
                reasoning="scripted lookup preflight",
            )
        if schema_name == "GroundedLookupResult":
            return response_schema(
                status=self.grounded_status,
                answer=self.grounded_answer,
                sources=["Official source"],
                source_quality="official",
                reasoning="scripted grounded result",
            )
        if schema_name == "CrisisLocationDecision":
            return response_schema(
                status=self.crisis_location_status,
                location=(
                    self.crisis_location
                    if self.crisis_location_status == "provided"
                    else ""
                ),
                reasoning="scripted crisis location",
            )
        if schema_name == "CrisisResourceLookupResult":
            resources = []
            if self.crisis_resource_status == "found":
                resources = [
                    {
                        "name": "Samaritans of Singapore",
                        "phone": "1767",
                        "url": "https://www.sos.org.sg",
                        "region": self.crisis_location,
                    }
                ]
            return response_schema(
                status=self.crisis_resource_status,
                resources=resources,
                reasoning="scripted crisis resources",
            )
        return await super().generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )
