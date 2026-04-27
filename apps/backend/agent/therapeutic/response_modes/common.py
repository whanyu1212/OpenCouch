"""Shared streaming and response-delta helpers for therapeutic modes."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.models import ModeType, ResponseCategory
from agent.runtime_context import WorkflowContext
from agent.state import AgentState
from agent.therapeutic.prompts import build_therapeutic_response_prompt

StreamWriterFactory = Callable[[], Callable[[dict[str, str]], None]]
SystemPromptBuilder = Callable[[AgentState], str]
ResponsePostprocessor = Callable[[str], str]


def therapeutic_response_delta(*, mode: str, response_text: str) -> dict[str, Any]:
    """Build the fixed response delta emitted by simple therapeutic modes.

    Args:
        mode: Therapeutic response style name.
        response_text: Text to return to the user.

    Returns:
        Parent-graph response delta for a therapeutic turn.
    """

    return {
        "response_kind": ResponseCategory.THERAPEUTIC,
        "response_text": response_text,
        "response_style": mode,
        "response_style_source": "therapeutic_dispatch",
        "response_style_type": ModeType.THERAPEUTIC,
    }


async def run_streamed_mode_response(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
    *,
    mode: str,
    system_prompt_builder: SystemPromptBuilder,
    fallback_text: str,
    logger: logging.Logger,
    failure_message: str,
    postprocess: ResponsePostprocessor | None = None,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> dict[str, Any]:
    """Run a simple therapeutic response mode with streaming and fallback.

    Args:
        state: Current graph state for the turn.
        runtime: LangGraph runtime carrying configured dependencies.
        mode: Therapeutic response style name.
        system_prompt_builder: Function that builds the system prompt.
        fallback_text: Deterministic response used when no LLM is available
            or the LLM call fails.
        logger: Module logger for fallback warnings.
        failure_message: Warning message emitted when the LLM path fails.
        postprocess: Optional response-text postprocessor.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Parent-graph response delta for the therapeutic turn.
    """

    llm_client = runtime.context.response_llm or runtime.context.llm_client

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        mode=mode,
        system_prompt_builder=system_prompt_builder,
        fallback_text=fallback_text,
        logger=logger,
        failure_message=failure_message,
        postprocess=postprocess,
        stream_writer_factory=stream_writer_factory,
    )
    return therapeutic_response_delta(mode=mode, response_text=response_text)


async def generate_streamed_therapeutic_text(
    *,
    state: AgentState,
    llm_client: Any,
    mode: str,
    system_prompt_builder: SystemPromptBuilder,
    fallback_text: str,
    logger: logging.Logger,
    failure_message: str,
    step_directive: str | None = None,
    postprocess: ResponsePostprocessor | None = None,
    stream_writer_factory: StreamWriterFactory = get_stream_writer,
) -> str:
    """Generate therapeutic response text with streaming and fallback.

    Args:
        state: Current graph state for the turn.
        llm_client: Response LLM client, if configured.
        mode: Therapeutic response style name.
        system_prompt_builder: Function that builds the system prompt.
        fallback_text: Deterministic text used when no LLM is available
            or the LLM call fails.
        logger: Module logger for fallback warnings.
        failure_message: Warning message emitted when the LLM path fails.
        step_directive: Optional guided-exercise directive to pass into the
            shared therapeutic response prompt.
        postprocess: Optional response-text postprocessor.
        stream_writer_factory: Factory that returns the current LangGraph
            stream writer.

    Returns:
        Generated response text, or ``fallback_text`` when streaming is
        unavailable or fails.
    """

    response_text = fallback_text
    if llm_client is not None:
        try:
            writer = stream_writer_factory()
            chunks: list[str] = []
            async for chunk in llm_client.generate_text_stream(
                prompt=build_therapeutic_response_prompt(
                    state,
                    mode=mode,
                    step_directive=step_directive,
                ),
                system_instruction=system_prompt_builder(state),
            ):
                chunks.append(chunk)
                writer({"type": "chunk", "text": chunk})
            response_text = "".join(chunks)
        except Exception:
            logger.warning(failure_message, exc_info=True)

    if postprocess is not None:
        response_text = postprocess(response_text)

    return response_text
