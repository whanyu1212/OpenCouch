"""User-directed memory-management helpers.

Public entry points for memory actions and memory-store operations. External
callers should import from this package; submodules are internal organization
and may be reorganized.

Symbols defined in :mod:`actions` and :mod:`service` are loaded lazily via
:pep:`562` ``__getattr__`` so lightweight callers that only need
``MemoryControlTarget`` do not pay the import cost of workflow-context
machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.gates.memory_control.operations import (
    MemoryControlTarget,
    delete_memory_target,
    find_memory_target_by_index,
    find_memory_targets,
    list_memory_for_owner,
    save_preference_rule,
    set_memory_recall,
)

if TYPE_CHECKING:
    from agent.gates.memory_control.actions import (
        CancelPendingAction,
        ConfirmPendingAction,
        ForgetByIndexAction,
        ForgetByQueryAction,
        ListAction,
        MemoryControlAction,
        SavePreferenceAction,
        SetRecallAction,
        StatusAction,
        TypedMemoryAction,
        parse_memory_control_action,
    )
    from agent.gates.memory_control.service import (
        MemoryControlRequest,
        MemoryControlServiceResult,
        execute_memory_control_action,
        execute_memory_control_request,
        memory_control_request_from_state,
    )


_LAZY_ACTION_SYMBOLS = frozenset(
    {
        "CancelPendingAction",
        "ConfirmPendingAction",
        "ForgetByIndexAction",
        "ForgetByQueryAction",
        "ListAction",
        "MemoryControlAction",
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
        "execute_memory_control_action",
        "execute_memory_control_request",
        "memory_control_request_from_state",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily resolve action/service symbols on first access."""

    if name in _LAZY_ACTION_SYMBOLS:
        from agent.gates.memory_control import actions as _actions

        value = getattr(_actions, name)
        globals()[name] = value
        return value

    if name in _LAZY_SERVICE_SYMBOLS:
        from agent.gates.memory_control import service as _service

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
    "MemoryControlAction",
    "MemoryControlRequest",
    "MemoryControlServiceResult",
    "MemoryControlTarget",
    "SavePreferenceAction",
    "SetRecallAction",
    "StatusAction",
    "TypedMemoryAction",
    "delete_memory_target",
    "execute_memory_control_action",
    "execute_memory_control_request",
    "find_memory_target_by_index",
    "find_memory_targets",
    "list_memory_for_owner",
    "memory_control_request_from_state",
    "parse_memory_control_action",
    "save_preference_rule",
    "set_memory_recall",
]
