"""Shared OpenAI text-runtime test fakes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from tests.support.persistence import FakeCrossRestartLLM


class FakeOpenAISDKRunner:
    """Deterministic Agents SDK runner fake for text-runtime tests."""

    def __init__(self, final_output: str = "openai reply") -> None:
        self.final_output = final_output
        self.run_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def run(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
    ) -> SimpleNamespace:
        self.run_calls.append(
            {"agent": agent, "input_text": input_text, "context": context}
        )
        return SimpleNamespace(final_output=self.final_output)

    def run_streamed(
        self,
        *,
        agent: Any,
        input_text: str,
        context: Any,
    ) -> "FakeOpenAIStream":
        self.stream_calls.append(
            {"agent": agent, "input_text": input_text, "context": context}
        )
        return FakeOpenAIStream(self.final_output)


class FakeOpenAIStream:
    """Deterministic streaming result fake for the Agents SDK."""

    def __init__(self, final_output: str) -> None:
        self.final_output = final_output

    async def stream_events(self) -> AsyncIterator[SimpleNamespace]:
        yield SimpleNamespace(
            type="raw_response_event",
            data=SimpleNamespace(
                type="response.output_text.delta",
                delta=self.final_output,
            ),
        )


class ScriptedOpenAITextRouteLLM(FakeCrossRestartLLM):
    """Fake control LLM for exercising OpenAI text-runtime route slices."""

    def __init__(
        self,
        *,
        route: str,
        crisis_level: int = 0,
        memory_reference_mode: str = "none",
        memory_action_type: str = "status",
        active_flow_action: str = "none",
        enabled: bool = True,
        target_kind: str = "fact",
        target_index: int = 1,
        query: str = "saved memory",
        preference_text: str = "direct answers when I am spiraling",
        grounded_answer: str = "Official answer.\n\nSources:\n- Official source",
        grounded_status: str = "answered",
    ) -> None:
        super().__init__()
        self.route = route
        self.crisis_level = crisis_level
        self.memory_reference_mode = memory_reference_mode
        self.memory_action_type = memory_action_type
        self.active_flow_action = active_flow_action
        self.enabled = enabled
        self.target_kind = target_kind
        self.target_index = target_index
        self.query = query
        self.preference_text = preference_text
        self.grounded_answer = grounded_answer
        self.grounded_status = grounded_status

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        schema_name = response_schema.__name__
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
                "confidence": "high",
                "memory_reference_mode": self.memory_reference_mode,
            }
            if self.route == "memory_control":
                kwargs["memory_action_type"] = self.memory_action_type
                if self.memory_action_type == "set_recall":
                    kwargs["enabled"] = self.enabled
                if self.memory_action_type == "forget_by_index":
                    kwargs["target_kind"] = self.target_kind
                    kwargs["target_index"] = self.target_index
                if self.memory_action_type == "forget_by_query":
                    kwargs["query"] = self.query
                if self.memory_action_type == "save_preference":
                    kwargs["preference_text"] = self.preference_text
            if self.route == "grounded_lookup":
                kwargs["query"] = "grounded query"
            return response_schema(**kwargs)
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
        return await super().generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )
