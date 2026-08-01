"""Service logic for executing explicit memory-management actions."""

from __future__ import annotations

from collections.abc import Mapping

from agent.memory.control.actions import (
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
from agent.memory.control.deletion_service import (
    cancel_pending_result,
    handle_confirm_pending,
    handle_forget_by_index,
    handle_forget_by_query,
)
from agent.memory.control.mutation_service import (
    handle_save_preference,
    handle_set_recall,
)
from agent.memory.control.read_service import (
    incognito_memory_control_result,
    handle_memory_list,
    handle_memory_status,
    owner_or_failure_result,
)
from agent.memory.control.types import (
    MemoryControlDependencies,
    MemoryControlRequest,
    MemoryControlServiceResult,
    PreferenceRuleDecision,
)
from agent.memory.modes import MemoryMode

__all__ = [
    "MemoryControlDependencies",
    "MemoryControlRequest",
    "MemoryControlServiceResult",
    "PreferenceRuleDecision",
    "execute_memory_control_request",
]


async def execute_memory_control_request(
    request: MemoryControlRequest,
    dependencies: MemoryControlDependencies,
) -> MemoryControlServiceResult:
    """Execute an explicit memory-management action from neutral input.

    Args:
        request: Framework-neutral memory action request.
        dependencies: Explicit memory-control dependencies.

    Returns:
        MemoryControlServiceResult: User-facing reply plus memory-management state
            updates.
    """

    if dependencies.memory_mode == MemoryMode.INCOGNITO:
        return incognito_memory_control_result()

    owner_id, failure_result = owner_or_failure_result(request.owner_id)
    if owner_id is None:
        if failure_result is None:
            raise RuntimeError("owner resolution failed without a failure result.")
        return failure_result

    raw_action = request.action or {}
    if not isinstance(raw_action, Mapping):
        raise ValueError("Memory action must be a mapping.")
    action = parse_memory_control_action(dict(raw_action))

    store = dependencies.memory_store

    match action:
        case ListAction():
            return await handle_memory_list(store=store, owner_id=owner_id)
        case StatusAction():
            return await handle_memory_status(owner_id=owner_id, store=store)
        case SetRecallAction():
            return await handle_set_recall(
                action=action,
                store=store,
                owner_id=owner_id,
            )
        case SavePreferenceAction():
            return await handle_save_preference(
                action=action,
                store=store,
                llm_client=dependencies.llm_client,
                owner_id=owner_id,
                request=request,
            )
        case ForgetByIndexAction():
            return await handle_forget_by_index(
                action=action,
                store=store,
                owner_id=owner_id,
            )
        case ForgetByQueryAction():
            return await handle_forget_by_query(
                action=action,
                store=store,
                owner_id=owner_id,
                pending_action=request.pending_action,
            )
        case ConfirmPendingAction():
            return await handle_confirm_pending(
                store=store,
                owner_id=owner_id,
                pending_action=request.pending_action,
            )
        case CancelPendingAction():
            return cancel_pending_result()

    raise RuntimeError(f"Unhandled memory-control action: {action!r}")
