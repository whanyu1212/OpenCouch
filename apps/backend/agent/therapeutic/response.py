"""Generic therapeutic response node for non-exercise response styles."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState
from agent.therapeutic.prompts import (
    build_clarifying_system_prompt,
    build_closing_system_prompt,
    build_psychoeducation_system_prompt,
    build_reflective_system_prompt,
    build_supportive_system_prompt,
    build_technique_system_prompt,
)
from agent.therapeutic.response_styles import (
    SystemPromptBuilder,
    run_streamed_response_style,
)

_SYSTEM_PROMPT_BUILDERS: dict[str, SystemPromptBuilder] = {
    "supportive": build_supportive_system_prompt,
    "reflective": build_reflective_system_prompt,
    "clarifying": build_clarifying_system_prompt,
    "psychoeducation": build_psychoeducation_system_prompt,
    "closing": build_closing_system_prompt,
    "technique": build_technique_system_prompt,
}


def get_stream_writer() -> Any:
    """Return the default no-op stream writer for compatibility tests."""

    return lambda _payload: None


async def run_therapeutic_response_node(
    state: AgentState,
    runtime: Any,
) -> dict[str, Any]:
    """Generate a non-exercise therapeutic response from ``response_style``.

    Args:
        state: Current graph state for the turn.
        runtime: Runtime object carrying configured dependencies.

    Returns:
        Response delta for the parent graph.
    """

    response_style = state.get("response_style") or "supportive"
    system_prompt_builder = _SYSTEM_PROMPT_BUILDERS.get(response_style)
    if system_prompt_builder is None:
        raise ValueError(f"Unknown therapeutic response_style={response_style!r}.")

    return await run_streamed_response_style(
        state,
        runtime,
        response_style=response_style,
        system_prompt_builder=system_prompt_builder,
        stream_writer_factory=get_stream_writer,
    )
