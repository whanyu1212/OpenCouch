"""Runtime prompt products for therapeutic response generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent.runtime.session.history import state_without_prompt_history
from agent.runtime.session.state import format_recent_history
from agent.specialists.therapeutic_response.prompt_context import (
    _format_working_memory,
)
from agent.specialists.therapeutic_response.style_guidance import (
    render_therapeutic_response_style_guidance,
)
from agent.state import AgentState


@dataclass(frozen=True)
class TherapeuticResponseLLMRequest:
    prompt: str
    system_instruction: str


def response_style_from_state(state: Mapping[str, Any]) -> str:
    style = str(state.get("response_style") or "").strip()
    if style and style != "pending":
        return style
    return "supportive"


def therapeutic_approach_from_state(state: Mapping[str, Any]) -> str:
    approach = str(state.get("therapeutic_approach") or "").strip()
    return approach if approach else "none"


def therapeutic_system_prompt_for_state(state: AgentState) -> str:
    return render_therapeutic_response_style_guidance(
        state,
        response_style=response_style_from_state(state),
        therapeutic_approach=therapeutic_approach_from_state(state),
    )


def response_llm_prompt_for_state(
    state: AgentState,
    *,
    include_recent_history: bool = True,
) -> str:
    """Build the plain response-writer prompt for response LLM overrides."""

    prompt_state = (
        state if include_recent_history else state_without_prompt_history(state)
    )
    memory_block = _format_working_memory(prompt_state)
    return (
        "Write the next assistant message for a mental health support "
        "conversation.\n\n"
        "You are writing final user-facing text only. You do not have access "
        "to tools in this response-writing path. Do not emit tool calls, "
        "function names, JSON arguments, XML tags, internal style names, or "
        "implementation traces. Use any private context silently.\n\n"
        f"Recent conversation:\n{format_recent_history(prompt_state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {prompt_state['message']}"
    )


def build_therapeutic_response_llm_request(
    state: AgentState,
    *,
    include_recent_history: bool,
) -> TherapeuticResponseLLMRequest:
    return TherapeuticResponseLLMRequest(
        prompt=response_llm_prompt_for_state(
            state,
            include_recent_history=include_recent_history,
        ),
        system_instruction=therapeutic_system_prompt_for_state(state),
    )


def therapeutic_agent_prompt_for_state(state: AgentState) -> str:
    memory_block = _format_working_memory(state)
    style = response_style_from_state(state)
    approach = therapeutic_approach_from_state(state)
    style_guidance = render_therapeutic_response_style_guidance(
        state,
        response_style=style,
        therapeutic_approach=approach,
    )
    return (
        "Write the next assistant message for a mental health support "
        "conversation.\n\n"
        "Use the runtime-provided therapeutic response guidance below as "
        "private drafting guidance. Do not expose internal style names unless "
        "the user asks how the system works. Do not start or continue guided "
        "exercises here; the runtime routes those turns to GuidedExerciseAgent.\n\n"
        f"{style_guidance}\n\n"
        f"Recent conversation:\n{format_recent_history(state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {state['message']}"
    )


def operational_context_for_prompt(state: AgentState) -> str:
    lines = [
        "Operational context:",
        "- The current turn has already passed the app-owned crisis gate.",
        "- For ordinary therapeutic replies, apply the runtime-provided "
        "therapeutic response guidance already included in this prompt.",
        "- Use memory or grounded lookup tools instead when the user explicitly "
        "asks for saved-memory management or grounded lookup.",
    ]
    memory_control = state.get("memory_control", {}) or {}
    pending_action = (
        memory_control.get("pending_action")
        if isinstance(memory_control, Mapping)
        else None
    )
    if isinstance(pending_action, Mapping):
        preview = ""
        target = pending_action.get("target")
        if isinstance(target, Mapping):
            preview = str(target.get("preview") or "").strip()
        pending_line = (
            "- Pending memory deletion exists. Call confirm_memory_deletion only "
            "if the user clearly confirms; call cancel_memory_deletion only if "
            "the user clearly declines."
        )
        if preview:
            pending_line = f"{pending_line} Target preview: {preview}"
        lines.append(pending_line)

    turn_lifecycle = state.get("turn_lifecycle", {}) or {}
    if (
        isinstance(turn_lifecycle, Mapping)
        and turn_lifecycle.get("triage_confidence") == "low"
    ):
        tentative_route = str(turn_lifecycle.get("tentative_route") or "").strip()
        if tentative_route:
            lines.append(
                "- The user's intent is ambiguous. Triage tentatively suggested "
                f"'{tentative_route}'. Clarify whether the user wants to proceed "
                "with that intent or continue the current flow before taking "
                "route-specific action."
            )

    memory_reference = state.get("memory_reference", {}) or {}
    if (
        isinstance(memory_reference, Mapping)
        and memory_reference.get("mode") == "explicit"
    ):
        lines.append(
            "- The user explicitly asked to use prior conversation context; use "
            "retrieved memory context when it is available."
        )

    return "\n".join(lines)


def build_therapeutic_agent_input(
    state: AgentState,
    *,
    include_recent_history: bool,
) -> str:
    prompt_state = (
        state if include_recent_history else state_without_prompt_history(state)
    )
    prompt = therapeutic_agent_prompt_for_state(prompt_state)
    operational_context = operational_context_for_prompt(state)
    if not operational_context:
        return prompt
    return f"{prompt}\n\n{operational_context}"
