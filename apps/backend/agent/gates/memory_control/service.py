"""Service logic for executing explicit memory-management actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.gates.memory_control.operations import (
    MemoryControlTarget,
    delete_memory_target,
    find_memory_target_by_index,
    find_memory_targets,
    list_memory_for_owner,
    save_preference_rule,
    set_memory_recall,
)
from agent.memory.modes import MemoryMode
from agent.memory.procedural_profile import aget_procedural_profile
from agent.gates.memory_control.actions import (
    CancelPendingAction,
    ConfirmPendingAction,
    ForgetByIndexAction,
    ForgetByQueryAction,
    ListAction,
    SavePreferenceAction,
    SetRecallAction,
    StatusAction,
    parse_memory_control_action,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id


class PreferenceRuleDecision(BaseModel):
    """Structured rule generated from an explicit user preference."""

    rule_text: str = Field(
        min_length=1,
        max_length=280,
        description="One grammatical second-person procedural rule to persist.",
    )
    reasoning: str = Field(min_length=1, max_length=240)
    confidence: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class MemoryControlServiceResult:
    """Result of executing one memory-management action."""

    response_text: str
    memory_control: dict[str, Any]
    procedural_profile: dict[str, Any] | None = None


def _empty_memory_reply() -> str:
    """Return a concise reply for an empty memory store.

    Returns:
        str: User-facing empty-memory reply.
    """

    return (
        "I don't have any saved facts, session summaries, or style rules for you "
        "right now."
    )


def _format_memory_overview(previews: dict[str, list[str]]) -> str:
    """Render memory previews into a short user-facing list.

    Args:
        previews (dict[str, list[str]]): Memory preview rows grouped by facts,
            sessions, and rules.

    Returns:
        str: User-facing memory overview.
    """

    lines: list[str] = []
    labels = {
        "facts": "Saved facts",
        "sessions": "Session summaries",
        "rules": "Style preferences",
    }
    for key in ("facts", "sessions", "rules"):
        items = previews.get(key, [])
        if not items:
            continue
        lines.append(f"{labels[key]}:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(items, start=1))

    if not lines:
        return _empty_memory_reply()
    return "Here's what I currently have saved:\n\n" + "\n".join(lines)


def _pending_delete_reply(target: MemoryControlTarget) -> str:
    """Return confirmation wording for a pending memory deletion.

    Args:
        target (MemoryControlTarget): Memory target selected for deletion.

    Returns:
        str: User-facing deletion confirmation prompt.
    """

    return (
        f"I found this saved {target['kind']}:\n\n"
        f'"{target["preview"]}"\n\n'
        "Do you want me to delete it?"
    )


def _multiple_matches_reply(targets: list[MemoryControlTarget]) -> str:
    """Return disambiguation wording for multiple deletion matches.

    Args:
        targets (list[MemoryControlTarget]): Candidate memory targets matching
            the deletion query.

    Returns:
        str: User-facing disambiguation prompt.
    """

    lines = [
        "I found multiple saved memories that might match. Which one should I delete?"
    ]
    lines.extend(
        f"{index}. {target['kind']}: {target['preview']}"
        for index, target in enumerate(targets, start=1)
    )
    return "\n\n".join([lines[0], "\n".join(lines[1:])])


def _incognito_reply() -> str:
    """Return the no-op reply for incognito memory mode.

    Returns:
        str: User-facing no-persistent-memory reply.
    """

    return (
        "You're in guest mode, so I don't have persistent memory to show or edit "
        "for this session."
    )


async def _owner_or_reply(state: AgentState) -> tuple[str | None, str | None]:
    """Resolve the memory owner or return a user-facing failure reply.

    Args:
        state (AgentState): Current graph state containing user/session identity.

    Returns:
        tuple[str | None, str | None]: ``(owner_id, None)`` on success,
            otherwise ``(None, reply)``.
    """

    try:
        return resolve_owner_id(state), None
    except ValueError:
        return (
            None,
            "I don't have a stable memory owner for this conversation, so I can't "
            "show or edit saved memory here.",
        )


async def _handle_status(
    *,
    owner_id: str,
    context: WorkflowContext,
) -> str:
    """Return memory status text for one owner.

    Args:
        owner_id (str): Owner whose memory status should be loaded.
        context (WorkflowContext): Workflow context carrying memory dependencies.

    Returns:
        str: User-facing memory status text.
    """

    store = context.memory_store
    profile = await aget_procedural_profile(store, user_id=owner_id)
    fact_count = await store.arecord_count((owner_id, "semantic"))
    session_count = await store.arecord_count((owner_id, "episodic"))
    return (
        "Memory status:\n\n"
        f"Saved facts: {fact_count}\n"
        f"Session summaries: {session_count}\n"
        f"Style preferences: {len(profile.rules)}\n"
        f"Proactive recall: {'on' if profile.proactive_recall_enabled else 'off'}"
    )


def _build_preference_rule_prompt(
    *,
    state: AgentState,
    preference_text: str,
) -> str:
    """Build the focused rule-writing prompt for explicit preferences.

    Args:
        state (AgentState): Current graph state.
        preference_text (str): Preference phrase selected by turn dispatch.

    Returns:
        str: Prompt requesting one durable procedural rule.
    """

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
        f'Current user message: "{state.get("message", "")}"\n'
        f'preference_text: "{preference_text}"'
    )


def _build_preference_rule_system_prompt() -> str:
    """Return the system instruction for preference-rule writing.

    Returns:
        str: System instruction.
    """

    return (
        "You write concise, durable procedural memory rules for one user. "
        "Return only the structured rule decision."
    )


async def _write_preference_rule(
    *,
    action: SavePreferenceAction,
    context: WorkflowContext,
    state: AgentState,
) -> str:
    """Generate the final procedural rule for an explicit preference.

    Args:
        action (SavePreferenceAction): Save-preference action payload.
        context (WorkflowContext): Runtime context containing the control LLM.
        state (AgentState): Current graph state for evidence.

    Returns:
        str: Final rule text to persist.

    Raises:
        RuntimeError: If no control LLM is configured.
    """

    if context.llm_client is None:
        raise RuntimeError("save_preference requires an LLM client.")

    decision = await context.llm_client.generate_structured(
        prompt=_build_preference_rule_prompt(
            state=state,
            preference_text=action.preference_text,
        ),
        response_schema=PreferenceRuleDecision,
        system_instruction=_build_preference_rule_system_prompt(),
    )
    return decision.rule_text


async def _handle_set_recall(
    *, action: SetRecallAction, store: Any, owner_id: str
) -> MemoryControlServiceResult:
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


async def _handle_save_preference(
    *,
    action: SavePreferenceAction,
    context: WorkflowContext,
    owner_id: str,
    state: AgentState,
) -> MemoryControlServiceResult:
    if context.llm_client is None:
        raise RuntimeError("save_preference requires an LLM client.")

    rule_text = await _write_preference_rule(
        action=action,
        context=context,
        state=state,
    )
    saved_rule = await save_preference_rule(
        context.memory_store,
        owner_id=owner_id,
        rule_text=rule_text,
        evidence=state.get("message", ""),
        llm_client=context.llm_client,
    )
    return MemoryControlServiceResult(
        response_text=f"Saved: {saved_rule}",
        memory_control={"pending_action": None},
    )


async def _handle_forget_by_index(
    *, action: ForgetByIndexAction, store: Any, owner_id: str
) -> MemoryControlServiceResult:
    target = await find_memory_target_by_index(
        store,
        owner_id=owner_id,
        kind=action.target_kind,
        index_1based=action.target_index,
    )
    if target is None:
        return MemoryControlServiceResult(
            response_text=(
                f"I couldn't find saved {action.target_kind} #{action.target_index}."
            ),
            memory_control={"pending_action": None},
        )
    return MemoryControlServiceResult(
        response_text=_pending_delete_reply(target),
        memory_control={"pending_action": {"type": "delete", "target": target}},
    )


async def _handle_forget_by_query(
    *, action: ForgetByQueryAction, store: Any, owner_id: str
) -> MemoryControlServiceResult:
    targets = await find_memory_targets(store, owner_id=owner_id, query=action.query)
    if not targets:
        return MemoryControlServiceResult(
            response_text="I couldn't find a saved memory matching that.",
            memory_control={"pending_action": None},
        )
    if len(targets) > 1:
        return MemoryControlServiceResult(
            response_text=_multiple_matches_reply(targets),
            memory_control={"pending_action": None},
        )
    target = targets[0]
    return MemoryControlServiceResult(
        response_text=_pending_delete_reply(target),
        memory_control={"pending_action": {"type": "delete", "target": target}},
    )


async def _handle_confirm_pending(
    *, store: Any, owner_id: str, state: AgentState
) -> MemoryControlServiceResult:
    pending = (state.get("memory_control", {}) or {}).get("pending_action") or {}
    target = pending.get("target")
    if not isinstance(target, dict):
        return MemoryControlServiceResult(
            response_text="There isn't a pending memory change to confirm.",
            memory_control={"pending_action": None},
        )
    deleted = await delete_memory_target(
        store,
        owner_id=owner_id,
        target=target,  # type: ignore[arg-type]
    )
    kind = target.get("kind", "memory")
    reply = (
        f"Deleted that saved {kind}."
        if deleted
        else "I couldn't delete that memory because it was already gone."
    )
    return MemoryControlServiceResult(
        response_text=reply,
        memory_control={"pending_action": None},
    )


async def execute_memory_control_action(
    state: AgentState,
    context: WorkflowContext,
) -> MemoryControlServiceResult:
    """Execute an explicit memory-management action.

    Args:
        state (AgentState): Current graph state with ``memory_control.action`` set
            by the gate.
        context (WorkflowContext): Workflow context carrying memory dependencies.

    Returns:
        MemoryControlServiceResult: User-facing reply plus memory-management state
            updates.
    """

    if context.memory_mode == MemoryMode.INCOGNITO:
        return MemoryControlServiceResult(
            response_text=_incognito_reply(),
            memory_control={"pending_action": None},
        )

    owner_id, failure_reply = await _owner_or_reply(state)
    if owner_id is None:
        return MemoryControlServiceResult(
            response_text=failure_reply or _empty_memory_reply(),
            memory_control={"pending_action": None},
        )

    raw_action = (state.get("memory_control", {}) or {}).get("action", {}) or {}
    if not isinstance(raw_action, dict):
        raise ValueError("memory_control.action must be a mapping.")
    action = parse_memory_control_action(raw_action)

    store = context.memory_store

    match action:
        case ListAction():
            previews = await list_memory_for_owner(store, owner_id=owner_id)
            return MemoryControlServiceResult(
                response_text=_format_memory_overview(previews),
                memory_control={"pending_action": None},
            )
        case StatusAction():
            return MemoryControlServiceResult(
                response_text=await _handle_status(owner_id=owner_id, context=context),
                memory_control={"pending_action": None},
            )
        case SetRecallAction():
            return await _handle_set_recall(
                action=action, store=store, owner_id=owner_id
            )
        case SavePreferenceAction():
            return await _handle_save_preference(
                action=action, context=context, owner_id=owner_id, state=state
            )
        case ForgetByIndexAction():
            return await _handle_forget_by_index(
                action=action, store=store, owner_id=owner_id
            )
        case ForgetByQueryAction():
            return await _handle_forget_by_query(
                action=action, store=store, owner_id=owner_id
            )
        case ConfirmPendingAction():
            return await _handle_confirm_pending(
                store=store, owner_id=owner_id, state=state
            )
        case CancelPendingAction():
            return MemoryControlServiceResult(
                response_text="Cancelled. I didn't change your memory.",
                memory_control={"pending_action": None},
            )

    raise RuntimeError(f"Unhandled memory-control action: {action!r}")
