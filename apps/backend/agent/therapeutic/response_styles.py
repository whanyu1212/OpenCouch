"""Shared streaming and response-delta helpers for therapeutic response styles."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.state import AgentState
from agent.therapeutic.prompts import build_therapeutic_response_prompt

StreamWriterFactory = Callable[[], Callable[[dict[str, str]], None]]
SystemPromptBuilder = Callable[[AgentState], str]


def _noop_stream_writer_factory() -> Callable[[dict[str, str]], None]:
    """Return a no-op stream writer for non-graph runtimes."""

    return lambda _payload: None


def therapeutic_response_delta(
    *,
    response_style: str,
    response_text: str,
) -> dict[str, Any]:
    """Build the fixed response delta emitted by therapeutic response styles.

    Args:
        response_style: Therapeutic response style name.
        response_text: Text to return to the user.

    Returns:
        Parent-graph response delta for a therapeutic turn.
    """

    return {
        "response_text": response_text,
        "response_style": response_style,
    }


async def run_streamed_response_style(
    state: AgentState,
    runtime: Any,
    *,
    response_style: str,
    system_prompt_builder: SystemPromptBuilder,
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> dict[str, Any]:
    """Run a therapeutic response style with streaming.

    Args:
        state: Current graph state for the turn.
        runtime: Runtime object carrying configured dependencies.
        response_style: Therapeutic response style name.
        system_prompt_builder: Function that builds the system prompt.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Parent-graph response delta for the therapeutic turn.
    """

    llm_client = runtime.context.response_llm or runtime.context.llm_client

    response_text = await generate_streamed_therapeutic_text(
        state=state,
        llm_client=llm_client,
        response_style=response_style,
        system_prompt_builder=system_prompt_builder,
        stream_writer_factory=stream_writer_factory,
    )
    return therapeutic_response_delta(
        response_style=response_style,
        response_text=response_text,
    )


async def generate_streamed_therapeutic_text(
    *,
    state: AgentState,
    llm_client: Any,
    response_style: str,
    system_prompt_builder: SystemPromptBuilder,
    step_directive: str | None = None,
    stream_writer_factory: StreamWriterFactory = _noop_stream_writer_factory,
) -> str:
    """Generate therapeutic response text with streaming.

    Args:
        state: Current graph state for the turn.
        llm_client: Response LLM client.
        response_style: Therapeutic response style name.
        system_prompt_builder: Function that builds the system prompt.
        step_directive: Optional guided-exercise directive to pass into the
            shared therapeutic response prompt.
        stream_writer_factory: Factory that returns the current stream writer.

    Returns:
        Generated response text.

    Raises:
        RuntimeError: If no LLM client is available.
    """

    if llm_client is None:
        raise RuntimeError(
            f"No LLM client available for {response_style} response generation."
        )

    writer = stream_writer_factory()
    chunks: list[str] = []
    async for chunk in llm_client.generate_text_stream(
        prompt=build_therapeutic_response_prompt(
            state,
            response_style=response_style,
            step_directive=step_directive,
        ),
        system_instruction=system_prompt_builder(state),
    ):
        chunks.append(chunk)
        writer({"type": "chunk", "text": chunk})
    return "".join(chunks)
