"""Generic therapeutic response node for non-exercise response styles."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from agent.runtime_context import WorkflowContext
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
    ResponsePostprocessor,
    SystemPromptBuilder,
    run_streamed_response_style,
)

logger = logging.getLogger(__name__)

THERAPEUTIC_RESPONSE_NODE = "therapeutic_response_node"

FallbackBuilder = Callable[[AgentState], str]

_DEFAULT_SUPPORTIVE_REPLY = (
    "It sounds like there's a lot on your mind right now, "
    "and what you're feeling makes sense. "
    "Take your time — I'm here whenever you're ready to say more."
)
_DEFAULT_REFLECTIVE_REPLY = (
    "I notice you keep coming back to this — it sounds like there might be a "
    "pattern here that's worth looking at together. "
    "What do you think connects these moments for you?"
)
_REFLECTIVE_FALLBACK_QUESTION = (
    "What do you think keeps connecting these moments for you?"
)
_DEFAULT_CLARIFYING_REPLY = (
    "It sounds like something's on your mind. "
    "Can you help me understand a bit more about what brought this up today?"
)
_DEFAULT_SAFETY_CLARIFYING_REPLY = (
    "I'm really glad you said that. "
    "Are you feeling at risk of hurting yourself right now?"
)
_DEFAULT_PSYCHOEDUCATION_REPLY = (
    "Something about this reaction might be worth framing gently. "
    "I have a thought about what could be going on — but first, "
    "does it feel like the right moment to sit with this, "
    "or would something steadier help more?"
)
_PSYCHOEDUCATION_FALLBACK_QUESTION = "Does that fit what you notice in your body?"
_DEFAULT_CLOSING_REPLY = (
    "I'm glad you took the time to talk this through. "
    "Whenever you want to pick this back up, I'm here."
)
_DEFAULT_TECHNIQUE_REPLY = (
    "Let's stay with what you just described. "
    "What's the thought that shows up most in that moment?"
)
_ATTUNED_OPENING_TERMS = (
    "hard",
    "heavy",
    "painful",
    "sounds",
    "makes sense",
    "understandable",
    "a lot",
    "carrying",
    "tough",
)


def _ensure_reflective_question(response_text: str) -> str:
    """Normalize reflective replies without forcing a trailing question.

    Args:
        response_text: The generated reflective reply text.

    Returns:
        The original text when present, otherwise the default reflective reply.
    """

    stripped = response_text.strip()
    if not stripped:
        return _DEFAULT_REFLECTIVE_REPLY
    return stripped


def _needs_safety_clarification(state: AgentState) -> bool:
    """Return whether this clarifying turn is a safety check.

    Args:
        state: Current graph state for the turn.

    Returns:
        Whether the crisis gate marked this turn as level-1 ambiguous risk.
    """

    crisis = state.get("crisis")
    if crisis is None:
        return False
    if isinstance(crisis, dict):
        return bool(crisis.get("needs_clarification", False))
    return bool(getattr(crisis, "needs_clarification", False))


def _default_clarifying_reply(state: AgentState) -> str:
    """Return the deterministic clarifying fallback for this state.

    Args:
        state: Current graph state for the turn.

    Returns:
        Safety-check fallback for level-1 crisis ambiguity, otherwise the
        ordinary clarifying fallback.
    """

    if _needs_safety_clarification(state):
        return _DEFAULT_SAFETY_CLARIFYING_REPLY
    return _DEFAULT_CLARIFYING_REPLY


def _ensure_psychoeducation_question(response_text: str) -> str:
    """Normalize psychoeducation replies without forcing a stock check-in.

    Args:
        response_text: The generated psychoeducation reply text.

    Returns:
        The original text when present, otherwise the default
        psychoeducation reply.
    """

    stripped = response_text.strip()
    if not stripped:
        return _DEFAULT_PSYCHOEDUCATION_REPLY
    return stripped


def _first_sentence(response_text: str) -> str:
    """Return the first sentence-like span from a generated response.

    Args:
        response_text: The generated technique reply text.

    Returns:
        The first sentence-like span, or the stripped text when no sentence
        boundary is present.
    """

    match = re.search(r"(.+?[.!?])(?:\s|$)", response_text.strip())
    if match:
        return match.group(1).strip()
    return response_text.strip()


def _ensure_attuned_opening(response_text: str) -> str:
    """Normalize technique replies while keeping an attuned opening.

    Args:
        response_text: The generated technique reply text.

    Returns:
        The original reply when its first sentence is already attuned, or a
        lighter prefixed acknowledgment when the opening is too abrupt.
    """

    stripped = response_text.strip()
    if not stripped:
        return _DEFAULT_TECHNIQUE_REPLY

    first_sentence = _first_sentence(stripped).lower()
    if any(term in first_sentence for term in _ATTUNED_OPENING_TERMS):
        return stripped
    return f"Let's slow this down for a second. {stripped}"


@dataclass(frozen=True)
class TherapeuticResponseStyleConfig:
    """Configuration for a non-exercise therapeutic response style."""

    system_prompt_builder: SystemPromptBuilder
    fallback_builder: FallbackBuilder
    failure_message: str
    postprocess: ResponsePostprocessor | None = None


def _static_fallback(text: str) -> FallbackBuilder:
    """Return a fallback builder for static deterministic replies.

    Args:
        text: Static fallback response text.

    Returns:
        A fallback builder that ignores state and returns ``text``.
    """

    return lambda _state: text


_RESPONSE_STYLE_CONFIGS: dict[str, TherapeuticResponseStyleConfig] = {
    "supportive": TherapeuticResponseStyleConfig(
        system_prompt_builder=build_supportive_system_prompt,
        fallback_builder=_static_fallback(_DEFAULT_SUPPORTIVE_REPLY),
        failure_message=(
            "Supportive response LLM call failed; using deterministic fallback."
        ),
    ),
    "reflective": TherapeuticResponseStyleConfig(
        system_prompt_builder=build_reflective_system_prompt,
        fallback_builder=_static_fallback(_DEFAULT_REFLECTIVE_REPLY),
        failure_message=(
            "Reflective response LLM call failed; using deterministic fallback."
        ),
        postprocess=_ensure_reflective_question,
    ),
    "clarifying": TherapeuticResponseStyleConfig(
        system_prompt_builder=build_clarifying_system_prompt,
        fallback_builder=_default_clarifying_reply,
        failure_message=(
            "Clarifying response LLM call failed; using deterministic fallback."
        ),
    ),
    "psychoeducation": TherapeuticResponseStyleConfig(
        system_prompt_builder=build_psychoeducation_system_prompt,
        fallback_builder=_static_fallback(_DEFAULT_PSYCHOEDUCATION_REPLY),
        failure_message=(
            "Psychoeducation response LLM call failed; using deterministic fallback."
        ),
        postprocess=_ensure_psychoeducation_question,
    ),
    "closing": TherapeuticResponseStyleConfig(
        system_prompt_builder=build_closing_system_prompt,
        fallback_builder=_static_fallback(_DEFAULT_CLOSING_REPLY),
        failure_message=(
            "Closing response LLM call failed; using deterministic fallback."
        ),
    ),
    "technique": TherapeuticResponseStyleConfig(
        system_prompt_builder=build_technique_system_prompt,
        fallback_builder=_static_fallback(_DEFAULT_TECHNIQUE_REPLY),
        failure_message=(
            "Technique response LLM call failed; using deterministic fallback."
        ),
        postprocess=_ensure_attuned_opening,
    ),
}


async def run_therapeutic_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a non-exercise therapeutic response from ``response_style``.

    Args:
        state: Current graph state for the turn.
        runtime: LangGraph runtime carrying configured dependencies.

    Returns:
        Response delta for the parent graph.
    """

    response_style = state.get("response_style") or "supportive"
    config = _RESPONSE_STYLE_CONFIGS.get(response_style)
    if config is None:
        logger.warning(
            "Unknown therapeutic response_style=%s; falling back to supportive.",
            response_style,
        )
        response_style = "supportive"
        config = _RESPONSE_STYLE_CONFIGS[response_style]

    return await run_streamed_response_style(
        state,
        runtime,
        response_style=response_style,
        system_prompt_builder=config.system_prompt_builder,
        fallback_text=config.fallback_builder(state),
        logger=logger,
        failure_message=config.failure_message,
        postprocess=config.postprocess,
        stream_writer_factory=get_stream_writer,
    )
