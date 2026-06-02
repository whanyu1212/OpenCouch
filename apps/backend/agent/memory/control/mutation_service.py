"""Mutation memory-control service helpers."""

from __future__ import annotations

from typing import Any

from agent.memory.control.actions import SavePreferenceAction, SetRecallAction
from agent.memory.control.operations import save_preference_rule, set_memory_recall
from agent.memory.control.types import (
    MemoryControlRequest,
    MemoryControlServiceResult,
    PreferenceRuleDecision,
)
from agent.runtime_context import WorkflowContext


def _build_preference_rule_prompt(
    *,
    current_user_message: str,
    preference_text: str,
) -> str:
    """Build the focused rule-writing prompt for explicit preferences."""

    return (
        "Convert the explicit user preference into one durable procedural rule "
        "for how the assistant should respond or use memory.\n\n"
        "Rules:\n"
        "- Write exactly one grammatical second-person rule.\n"
        "- The rule must start with 'You prefer', 'You do not want', 'Do not', "
        "'When', or 'If'.\n"
        "- Preserve the user's intent; do not add therapeutic advice or new facts.\n"
        "- Use natural grammar. Do not write fragments such as 'You prefer give...'.\n"
        "- Keep it under 24 words.\n\n"
        "Examples:\n"
        "preference_text: direct answers when I am spiraling\n"
        "rule_text: You prefer direct answers when you are spiraling.\n\n"
        "preference_text: don't suggest journaling\n"
        "rule_text: Do not suggest journaling.\n\n"
        "preference_text: ask fewer questions\n"
        "rule_text: You prefer fewer questions.\n\n"
        f'Current user message: "{current_user_message}"\n'
        f'preference_text: "{preference_text}"'
    )


def _build_preference_rule_system_prompt() -> str:
    """Return the system instruction for preference-rule writing."""

    return (
        "You write concise, durable procedural memory rules for one user. "
        "Return only the structured rule decision."
    )


async def _write_preference_rule(
    *,
    action: SavePreferenceAction,
    context: WorkflowContext,
    current_user_message: str,
) -> str:
    """Generate the final procedural rule for an explicit preference."""

    if context.llm_client is None:
        raise RuntimeError("save_preference requires an LLM client.")

    decision = await context.llm_client.generate_structured(
        prompt=_build_preference_rule_prompt(
            current_user_message=current_user_message,
            preference_text=action.preference_text,
        ),
        response_schema=PreferenceRuleDecision,
        system_instruction=_build_preference_rule_system_prompt(),
    )
    return decision.rule_text


async def handle_set_recall(
    *,
    action: SetRecallAction,
    store: Any,
    owner_id: str,
) -> MemoryControlServiceResult:
    """Enable or disable proactive recall for one owner."""

    await set_memory_recall(store, owner_id=owner_id, enabled=action.enabled)
    state_text = "on" if action.enabled else "off"
    reply = (
        f"I turned proactive recall {state_text}. "
        "Style preferences can still shape how I respond, but I "
        f"{'may' if action.enabled else 'will not'} proactively bring up past sessions."
    )
    return MemoryControlServiceResult(
        response_text=reply,
        memory_control={"pending_action": None},
        procedural_profile={"proactive_recall_enabled": action.enabled},
    )


async def handle_save_preference(
    *,
    action: SavePreferenceAction,
    context: WorkflowContext,
    owner_id: str,
    request: MemoryControlRequest,
) -> MemoryControlServiceResult:
    """Persist one explicit response-style or memory-use preference."""

    if context.llm_client is None:
        raise RuntimeError("save_preference requires an LLM client.")

    rule_text = await _write_preference_rule(
        action=action,
        context=context,
        current_user_message=request.current_user_message,
    )
    saved_rule = await save_preference_rule(
        context.memory_store,
        owner_id=owner_id,
        rule_text=rule_text,
        evidence=request.current_user_message,
        llm_client=context.llm_client,
    )
    return MemoryControlServiceResult(
        response_text=f"Saved: {saved_rule}",
        memory_control={"pending_action": None},
    )
