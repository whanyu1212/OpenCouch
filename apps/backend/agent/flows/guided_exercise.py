"""Guided exercise path helpers for the OpenAI text runtime."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from agents import Agent
from llm.base import BaseLLMClient, StructuredResponseT

from agent.runtime.context import OpenAITextRunContext
from agent.runtime.prompt_utils import (
    chunk_from_sdk_event,
    final_output_text,
    include_prompt_history,
    strip_recent_history_from_prompt,
)
from agent.runtime.state_ops import apply_state_delta
from agent.specialists.guided_exercise import (
    GUIDED_EXERCISE_AGENT_INSTRUCTIONS,
    GUIDED_EXERCISE_AGENT_NAME,
)
from agent.runtime.types import (
    TextRuntimeChunkEvent,
    TextRuntimeStateEvent,
    TextRuntimeStatusEvent,
    TextRuntimeStreamEvent,
)
from agent.runtime_context import WorkflowContext
from agent.skills.guided_exercises.lifecycle import GuidedExerciseSkillService
from agent.skills.guided_exercises.registry import (
    available_exercise_definitions,
    iter_exercise_selection_aliases,
)


def guided_exercise_selection_basis(state: Mapping[str, object]) -> str | None:
    exercise_state = state.get("exercise_state", {}) or {}
    active_exercise = (
        isinstance(exercise_state, Mapping)
        and exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    )
    message = str(state.get("message") or "")
    if active_exercise:
        if message_is_operational_side_request(message):
            return None
        if guided_exercise_runtime_action(state) == "preserve":
            return None
        return "active_exercise"
    if message_explicitly_requests_guided_exercise(state, message):
        return "explicit_user_request"
    return None


def guided_exercise_runtime_action(state: Mapping[str, object]) -> str:
    exercise_state = state.get("exercise_state", {}) or {}
    if (
        isinstance(exercise_state, Mapping)
        and exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    ):
        text = normalize_message_text(str(state.get("message") or ""))
        if any(phrase in text for phrase in ("resume", "return to", "back to")) and (
            "exercise" in text or "grounding" in text or "breathing" in text
        ):
            return "resume"
        if any(
            phrase in text
            for phrase in (
                "do you mean",
                "right now or",
                "right now, or",
                "or just around me",
                "what do you mean",
            )
        ):
            return "preserve"
        return "continue"
    return "start"


def message_is_operational_side_request(message: str) -> bool:
    text = normalize_message_text(message)
    if not text:
        return False
    memory_phrases = (
        "what do you remember",
        "what have you saved",
        "saved memory",
        "memory status",
        "forget",
        "delete memory",
        "delete that memory",
        "remove that memory",
        "turn proactive recall",
    )
    lookup_phrases = (
        "look this up",
        "look that up",
        "look up",
        "search",
        "source",
        "sources",
        "official",
        "current",
        "latest",
        "verify",
    )
    return any(phrase in text for phrase in (*memory_phrases, *lookup_phrases))


def message_explicitly_requests_guided_exercise(
    state: Mapping[str, object],
    message: str,
) -> bool:
    text = normalize_message_text(message)
    if not text:
        return False
    request_phrases = (
        "can we do",
        "could we do",
        "let's do",
        "lets do",
        "start",
        "walk me through",
        "guide me through",
        "take me through",
        "lead me through",
        "help me do",
        "help me with",
        "i need",
        "need a",
    )
    exercise_terms = (
        "exercise",
        "grounding",
        "breathing",
        "box breathing",
        "5 4 3 2 1",
        "5-4-3-2-1",
        "54321",
        "thought record",
        "values",
        "emotion regulation",
    )
    if any(phrase in text for phrase in request_phrases) and any(
        term in text for term in exercise_terms
    ):
        return True
    if "exercise" in text and any(
        phrase in text for phrase in ("do", "try", "start", "practice")
    ):
        return True
    aliases = available_exercise_aliases_for_state(state)
    return (
        bool(aliases)
        and any(phrase in text for phrase in request_phrases)
        and any(alias in text for alias in aliases)
    )


def available_exercise_aliases_for_state(
    state: Mapping[str, object],
) -> tuple[str, ...]:
    try:
        definitions = available_exercise_definitions(
            installed_skills=tuple(state.get("installed_skills") or ()),
            channel=str(state.get("channel") or "text"),
            therapeutic_approach=state.get("therapeutic_approach"),
        )
    except Exception:
        return ()
    aliases: set[str] = set()
    for definition in definitions:
        aliases.add(definition.id.replace("_", " "))
        aliases.add(definition.display_name)
    for alias, _definition in iter_exercise_selection_aliases(definitions=definitions):
        aliases.add(alias)
    return tuple(
        sorted(
            normalized
            for alias in aliases
            if (normalized := normalize_message_text(alias))
        )
    )


def normalize_message_text(message: str) -> str:
    return " ".join(message.casefold().replace("_", " ").split())


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
        async for sdk_event in stream.stream_events():
            chunk = chunk_from_sdk_event(sdk_event)
            if chunk:
                chunks.append(chunk)

        self.last_duration_ms = elapsed_ms(run_start)
        final_text = final_output_text(
            getattr(stream, "final_output", None),
            fallback="".join(chunks),
        )
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


@dataclass(frozen=True)
class _ExerciseSkillToolRequest:
    exercise_type: str
    runtime_action: str
    current_step_index: int | None


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
    runtime: Any,
    state: Any,
    config: Any,
    context: WorkflowContext,
    *,
    session: Any | None = None,
) -> BaseLLMClient:
    run_context = runtime._run_context_for_state(state, config, context)
    if context.response_llm is not None:
        return FallbackGuidedExerciseResponseLLM(
            fallback_llm=context.response_llm,
            run_context=run_context,
            strip_recent_history=not include_prompt_history(session),
        )
    return OpenAIGuidedExerciseResponseLLM(
        runner=runtime._runner,
        guided_exercise_agent=runtime._roster.guided_exercise_agent,
        run_context=run_context,
        session=session,
    )


def guided_exercise_skill_service(
    context: WorkflowContext,
    *,
    response_llm: BaseLLMClient,
    stream_writer_factory: Any | None = None,
) -> GuidedExerciseSkillService:
    kwargs: dict[str, Any] = {}
    if stream_writer_factory is not None:
        kwargs["stream_writer_factory"] = stream_writer_factory
    return GuidedExerciseSkillService(
        classifier_llm=context.llm_client,
        response_llm=response_llm,
        memory_store=context.memory_store,
        memory_mode=context.memory_mode,
        **kwargs,
    )


async def run_guided_exercise_turn(
    runtime: Any,
    state: Any,
    *,
    config: Any,
    context: WorkflowContext,
    streamed: bool,
    session: Any | None = None,
) -> Any:
    response_llm = guided_exercise_response_llm(
        runtime,
        state,
        config,
        context,
        session=session,
    )
    skill_service = guided_exercise_skill_service(
        context,
        response_llm=response_llm,
    )
    delta = await skill_service.run_turn(state)
    apply_state_delta(state, dict(delta))
    _apply_guided_exercise_tool_diagnostics(
        state,
        response_llm.run_context,
        fallback=response_llm.used_skill_tool_fallback,
    )
    response_text = str(state.get("response_text") or "")
    if not response_text:
        raise ValueError("guided_exercise returned an empty response.")
    return await runtime._finalize_openai_turn(
        state,
        response_text=response_text,
        config=config,
        runtime_mode="guided_exercise",
        response_style=str(state.get("response_style") or "guided_exercise"),
        selected_agent=GUIDED_EXERCISE_AGENT_NAME,
        sdk_duration_ms=response_llm.last_duration_ms,
        streamed=streamed,
    )


async def run_guided_exercise_turn_stream(
    runtime: Any,
    state: Any,
    *,
    config: Any,
    context: WorkflowContext,
    session: Any | None = None,
) -> AsyncIterator[TextRuntimeStreamEvent]:
    yield TextRuntimeStatusEvent(stage="guided_exercise")
    queue: asyncio.Queue[str] = asyncio.Queue()

    def writer_factory() -> Any:
        def writer(payload: dict[str, str]) -> None:
            if payload.get("type") == "chunk":
                queue.put_nowait(str(payload.get("text") or ""))

        return writer

    response_llm = guided_exercise_response_llm(
        runtime,
        state,
        config,
        context,
        session=session,
    )
    skill_service = guided_exercise_skill_service(
        context,
        response_llm=response_llm,
        stream_writer_factory=writer_factory,
    )
    task = asyncio.create_task(skill_service.run_turn(state))
    while not task.done() or not queue.empty():
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue
        if chunk:
            yield TextRuntimeChunkEvent(text=chunk)

    delta = await task
    apply_state_delta(state, dict(delta))
    _apply_guided_exercise_tool_diagnostics(
        state,
        response_llm.run_context,
        fallback=response_llm.used_skill_tool_fallback,
    )
    response_text = str(state.get("response_text") or "")
    if not response_text:
        raise ValueError("guided_exercise returned an empty response.")
    final_state = await runtime._finalize_openai_turn(
        state,
        response_text=response_text,
        config=config,
        runtime_mode="guided_exercise",
        response_style=str(state.get("response_style") or "guided_exercise"),
        selected_agent=GUIDED_EXERCISE_AGENT_NAME,
        sdk_duration_ms=response_llm.last_duration_ms,
        streamed=True,
    )
    yield TextRuntimeStatusEvent(stage="finalize", turn_finalized=True)
    yield TextRuntimeStateEvent(state=final_state)


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


def _replace_exercise_skill_context_with_tool_instruction(
    prompt: str,
) -> tuple[str, _ExerciseSkillToolRequest | None]:
    skill_start = prompt.find("Exercise skill:")
    runtime_task_marker = "\n\nRuntime task:"
    runtime_task_start = prompt.find(runtime_task_marker, skill_start)
    if skill_start == -1 or runtime_task_start == -1:
        return prompt, None

    skill_block = prompt[skill_start:runtime_task_start]
    exercise_type = _skill_block_value(skill_block, "skill_id")
    runtime_action = _skill_block_value(skill_block, "runtime_action")
    if not exercise_type or not runtime_action:
        return prompt, None

    current_step_index = _parse_optional_int(_skill_block_value(skill_block, "index"))
    arguments: dict[str, Any] = {
        "exercise_type": exercise_type,
        "runtime_action": runtime_action,
    }
    if current_step_index is not None:
        arguments["current_step_index"] = current_step_index

    replacement = (
        "Exercise skill:\n"
        "(skill context is owned by GuidedExerciseAgent tools)\n"
        "Required tool: load_guided_exercise_skill\n"
        f"Required tool arguments: {json.dumps(arguments, sort_keys=True)}\n"
        "Call the required tool exactly once before answering. Use only the "
        "returned skill_context plus the Runtime task below. Do not invent "
        "exercise steps, switch exercises, or offer a menu."
    )
    return (
        f"{prompt[:skill_start]}{replacement}{prompt[runtime_task_start:]}",
        _ExerciseSkillToolRequest(
            exercise_type=exercise_type,
            runtime_action=runtime_action,
            current_step_index=current_step_index,
        ),
    )


def _skill_block_value(block: str, key: str) -> str:
    prefix = f"- {key}:"
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).strip()
    return ""


def _parse_optional_int(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _guided_exercise_skill_tool_called(
    run_context: OpenAITextRunContext,
    *,
    tool_call_count: int,
) -> bool:
    return len(run_context.guided_exercise_skill_tool_calls) > tool_call_count


def _apply_guided_exercise_tool_diagnostics(
    state: Any,
    run_context: OpenAITextRunContext,
    *,
    fallback: bool,
) -> None:
    diagnostics = {
        **dict(state.get("diagnostics", {}) or {}),
        "openai_guided_exercise_tool_expected": "load_guided_exercise_skill",
        "openai_guided_exercise_tool_calls": [
            call.tool_name for call in run_context.guided_exercise_skill_tool_calls
        ],
        "openai_guided_exercise_tool_fallback": fallback,
    }
    latest = run_context.latest_guided_exercise_skill_tool_result()
    if latest is not None:
        diagnostics.update(
            {
                "openai_guided_exercise_tool_exercise_type": latest.exercise_type,
                "openai_guided_exercise_tool_runtime_action": latest.runtime_action,
                "openai_guided_exercise_tool_step": latest.current_step_index,
            }
        )
    apply_state_delta(state, {"diagnostics": diagnostics})


__all__ = [
    "available_exercise_aliases_for_state",
    "guided_exercise_response_llm",
    "guided_exercise_runtime_action",
    "guided_exercise_selection_basis",
    "guided_exercise_skill_service",
    "run_guided_exercise_turn",
    "run_guided_exercise_turn_stream",
]
