"""LLM adapters for guided exercise response execution."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from agents import Agent
from llm.base import BaseLLMClient, StructuredResponseT

from agent.flows.guided_exercise.tool_instruction import (
    _guided_exercise_skill_tool_called,
    _replace_exercise_skill_context_with_tool_instruction,
)
from agent.runtime.context import OpenAITextRunContext
from agent.runtime.prompt_utils import (
    chunk_from_sdk_event,
    final_output_text,
)
from agent.runtime.session.history import (
    include_prompt_history,
    strip_recent_history_from_prompt,
)
from agent.runtime.services import TextRuntimeServices
from agent.runtime.workflow_context import WorkflowContext
from agent.specialists.guided_exercise import GUIDED_EXERCISE_AGENT_INSTRUCTIONS


class OpenAIGuidedExerciseResponseLLM(BaseLLMClient):
    """Response LLM client that routes exercise prose through Agents SDK."""

    def __init__(
        self,
        *,
        runner: Any,
        guided_exercise_agent: Any,
        run_context: OpenAITextRunContext,
        session: Any | None = None,
    ) -> None:
        self._runner = runner
        self._guided_exercise_agent = guided_exercise_agent
        self._run_context = run_context
        self._session = session
        self.last_duration_ms: float | None = None
        self.used_skill_tool_fallback = False

    @property
    def run_context(self) -> OpenAITextRunContext:
        return self._run_context

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        from agent.observability.timing import elapsed_ms

        del use_search
        if self._session is not None:
            prompt = strip_recent_history_from_prompt(prompt)
        original_prompt = prompt
        prompt, tool_request = _replace_exercise_skill_context_with_tool_instruction(
            prompt
        )
        tool_call_count = len(self._run_context.guided_exercise_skill_tool_calls)
        run_start = time.monotonic()
        result = await self._runner.run(
            agent=_build_guided_exercise_agent(
                self._guided_exercise_agent,
                system_instruction=system_instruction,
                runtime_instructions=GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
            ),
            input_text=prompt,
            context=self._run_context,
            session=self._session,
        )
        self.last_duration_ms = elapsed_ms(run_start)
        if tool_request is not None and not _guided_exercise_skill_tool_called(
            self._run_context,
            tool_call_count=tool_call_count,
        ):
            self.used_skill_tool_fallback = True
            result = await self._runner.run(
                agent=_build_guided_exercise_agent(
                    self._guided_exercise_agent,
                    system_instruction=system_instruction,
                    runtime_instructions=GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
                ),
                input_text=original_prompt,
                context=self._run_context,
                session=self._session,
            )
            self.last_duration_ms = elapsed_ms(run_start)
        return final_output_text(getattr(result, "final_output", None))

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        from agent.observability.timing import elapsed_ms

        if self._session is not None:
            prompt = strip_recent_history_from_prompt(prompt)
        original_prompt = prompt
        prompt, tool_request = _replace_exercise_skill_context_with_tool_instruction(
            prompt
        )
        tool_call_count = len(self._run_context.guided_exercise_skill_tool_calls)
        run_start = time.monotonic()
        stream = self._runner.run_streamed(
            agent=_build_guided_exercise_agent(
                self._guided_exercise_agent,
                system_instruction=system_instruction,
                runtime_instructions=GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
            ),
            input_text=prompt,
            context=self._run_context,
            session=self._session,
        )
        chunks: list[str] = []
        if tool_request is None:
            async for sdk_event in stream.stream_events():
                chunk = chunk_from_sdk_event(sdk_event)
                if chunk:
                    chunks.append(chunk)
                    yield chunk

            self.last_duration_ms = elapsed_ms(run_start)
            final_text = final_output_text(
                getattr(stream, "final_output", None),
                fallback="".join(chunks),
            )
            if final_text and not chunks:
                yield final_text
            return

        # Tool-forcing prompts need a post-stream fallback decision: if the
        # SDK did not call the requested skill tool, we retry non-streaming with
        # the original prompt. Keep that path buffered so we never emit partial
        # first-pass chunks and then replace them with fallback prose.
        async for sdk_event in stream.stream_events():
            chunk = chunk_from_sdk_event(sdk_event)
            if chunk:
                chunks.append(chunk)

        self.last_duration_ms = elapsed_ms(run_start)
        final_text = final_output_text(
            getattr(stream, "final_output", None),
            fallback="".join(chunks),
        )
        if not _guided_exercise_skill_tool_called(
            self._run_context,
            tool_call_count=tool_call_count,
        ):
            self.used_skill_tool_fallback = True
            result = await self._runner.run(
                agent=_build_guided_exercise_agent(
                    self._guided_exercise_agent,
                    system_instruction=system_instruction,
                    runtime_instructions=GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
                ),
                input_text=original_prompt,
                context=self._run_context,
                session=self._session,
            )
            self.last_duration_ms = elapsed_ms(run_start)
            final_text = final_output_text(getattr(result, "final_output", None))
            chunks = [final_text]

        for chunk in chunks:
            yield chunk
        if final_text and not chunks:
            yield final_text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        del prompt, response_schema, system_instruction, use_search
        raise RuntimeError("Guided exercise response LLM does not classify.")


class FallbackGuidedExerciseResponseLLM(BaseLLMClient):
    """Guided-exercise response LLM for explicit response overrides."""

    def __init__(
        self,
        *,
        fallback_llm: BaseLLMClient,
        run_context: OpenAITextRunContext,
        strip_recent_history: bool = False,
    ) -> None:
        self._fallback_llm = fallback_llm
        self._run_context = run_context
        self._strip_recent_history = strip_recent_history
        self.last_duration_ms: float | None = None
        self.used_skill_tool_fallback = False

    @property
    def run_context(self) -> OpenAITextRunContext:
        return self._run_context

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        from agent.observability.timing import elapsed_ms

        if self._strip_recent_history:
            prompt = strip_recent_history_from_prompt(prompt)
        run_start = time.monotonic()
        text = await self._fallback_llm.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            use_search=use_search,
        )
        self.last_duration_ms = elapsed_ms(run_start)
        return text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        from agent.observability.timing import elapsed_ms

        if self._strip_recent_history:
            prompt = strip_recent_history_from_prompt(prompt)
        run_start = time.monotonic()
        async for chunk in self._fallback_llm.generate_text_stream(
            prompt=prompt,
            system_instruction=system_instruction,
        ):
            yield chunk
        self.last_duration_ms = elapsed_ms(run_start)

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[StructuredResponseT],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> StructuredResponseT:
        return await self._fallback_llm.generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )


def guided_exercise_response_llm(
    services: TextRuntimeServices,
    state: Any,
    config: Any,
    context: WorkflowContext,
    *,
    session: Any | None = None,
) -> BaseLLMClient:
    run_context = services.build_run_context(state, config, context)
    if context.response_llm is not None:
        return FallbackGuidedExerciseResponseLLM(
            fallback_llm=context.response_llm,
            run_context=run_context,
            strip_recent_history=not include_prompt_history(session),
        )
    return OpenAIGuidedExerciseResponseLLM(
        runner=services.runner,
        guided_exercise_agent=services.roster.guided_exercise_agent,
        run_context=run_context,
        session=session,
    )


def _build_guided_exercise_agent(
    agent: Agent[OpenAITextRunContext],
    *,
    system_instruction: str | None,
    runtime_instructions: str,
) -> Agent[OpenAITextRunContext]:
    instructions = runtime_instructions
    if system_instruction:
        instructions = f"{instructions}\n\n{system_instruction}"
    return Agent[OpenAITextRunContext](
        name=agent.name,
        handoff_description=agent.handoff_description,
        tools=list(agent.tools),
        mcp_servers=list(agent.mcp_servers),
        mcp_config=agent.mcp_config,
        instructions=instructions,
        prompt=agent.prompt,
        handoffs=list(agent.handoffs),
        model=agent.model,
        model_settings=agent.model_settings,
        input_guardrails=list(agent.input_guardrails),
        output_guardrails=list(agent.output_guardrails),
        output_type=agent.output_type,
        hooks=agent.hooks,
        tool_use_behavior=agent.tool_use_behavior,
        reset_tool_choice=agent.reset_tool_choice,
    )


__all__ = [
    "FallbackGuidedExerciseResponseLLM",
    "OpenAIGuidedExerciseResponseLLM",
    "_build_guided_exercise_agent",
    "guided_exercise_response_llm",
]
