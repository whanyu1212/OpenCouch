"""User-directed memory-management helpers.

This package owns the application service used by SDK memory tools. Routing may
classify a turn as memory management, but executable memory changes enter here
through explicit tool requests rather than through runtime state actions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.memory.control.operations import (
    MemoryControlTarget,
    delete_memory_target,
    find_memory_target_by_index,
    find_memory_targets,
    list_memory_for_owner,
    save_preference_rule,
    set_memory_recall,
)

if TYPE_CHECKING:
    from agent.memory.control.actions import (
        CancelPendingAction,
        ConfirmPendingAction,
        ForgetByIndexAction,
        ForgetByQueryAction,
        ListAction,
        SavePreferenceAction,
        SetRecallAction,
        StatusAction,
        TypedMemoryAction,
        parse_memory_control_action,
    )
    from agent.memory.control.service import (
        MemoryControlRequest,
        MemoryControlServiceResult,
        execute_memory_control_request,
    )


_LAZY_ACTION_SYMBOLS = frozenset(
    {
        "CancelPendingAction",
        "ConfirmPendingAction",
        "ForgetByIndexAction",
        "ForgetByQueryAction",
        "ListAction",
        "SavePreferenceAction",
        "SetRecallAction",
        "StatusAction",
        "TypedMemoryAction",
        "parse_memory_control_action",
    }
)

_LAZY_SERVICE_SYMBOLS = frozenset(
    {
        "MemoryControlRequest",
        "MemoryControlServiceResult",
        "execute_memory_control_request",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily resolve action/service symbols on first access."""

    if name in _LAZY_ACTION_SYMBOLS:
        from agent.memory.control import actions as _actions

        value = getattr(_actions, name)
        globals()[name] = value
        return value

    if name in _LAZY_SERVICE_SYMBOLS:
        from agent.memory.control import service as _service

        value = getattr(_service, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy symbols in ``dir()`` so introspection still works."""

    return sorted(set(globals()) | _LAZY_ACTION_SYMBOLS | _LAZY_SERVICE_SYMBOLS)


__all__ = [
    "CancelPendingAction",
    "ConfirmPendingAction",
    "ForgetByIndexAction",
    "ForgetByQueryAction",
    "ListAction",
    "MemoryControlRequest",
    "MemoryControlServiceResult",
    "MemoryControlTarget",
    "SavePreferenceAction",
    "SetRecallAction",
    "StatusAction",
    "TypedMemoryAction",
    "delete_memory_target",
    "execute_memory_control_request",
    "find_memory_target_by_index",
    "find_memory_targets",
    "list_memory_for_owner",
    "parse_memory_control_action",
    "save_preference_rule",
    "set_memory_recall",
]
