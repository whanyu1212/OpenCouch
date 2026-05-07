"""User-directed memory-management routing helpers.

Public entry points for explicit memory commands ("show my memory",
"forget X", "save preference Y", "turn proactive recall off"). External
callers should import from this package — submodules (router, service,
operations) are internal organization and may be reorganized.

Symbols defined in :mod:`router` and :mod:`service` are loaded lazily
via :pep:`562` ``__getattr__`` so that lightweight callers (e.g. a tool
that only needs ``MemoryControlTarget``) do not pay the import cost of
the LLM-routing and workflow-context machinery.
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
    from agent.gates.memory_control.router import (
        CancelPendingAction,
        ConfirmPendingAction,
        ForgetByIndexAction,
        ForgetByQueryAction,
        ListAction,
        MemoryControlAction,
        MemoryControlDecision,
        MemoryControlRoute,
        SavePreferenceAction,
        SetRecallAction,
        StatusAction,
        TypedMemoryAction,
        detect_memory_control_action,
        is_pending_cancellation,
        is_pending_confirmation,
        parse_memory_control_action,
        resolve_memory_control_action,
    )
    from agent.gates.memory_control.service import (
        MemoryControlServiceResult,
        execute_memory_control_action,
    )


_LAZY_ROUTER_SYMBOLS = frozenset(
    {
        "CancelPendingAction",
        "ConfirmPendingAction",
        "ForgetByIndexAction",
        "ForgetByQueryAction",
        "ListAction",
        "MemoryControlAction",
        "MemoryControlDecision",
        "MemoryControlRoute",
        "SavePreferenceAction",
        "SetRecallAction",
        "StatusAction",
        "TypedMemoryAction",
        "detect_memory_control_action",
        "is_pending_cancellation",
        "is_pending_confirmation",
        "parse_memory_control_action",
        "resolve_memory_control_action",
    }
)

_LAZY_SERVICE_SYMBOLS = frozenset(
    {
        "MemoryControlServiceResult",
        "execute_memory_control_action",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily resolve router/service symbols on first access (PEP 562)."""

    if name in _LAZY_ROUTER_SYMBOLS:
        from agent.gates.memory_control import router as _router

        value = getattr(_router, name)
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

    return sorted(set(globals()) | _LAZY_ROUTER_SYMBOLS | _LAZY_SERVICE_SYMBOLS)


__all__ = [
    "CancelPendingAction",
    "ConfirmPendingAction",
    "ForgetByIndexAction",
    "ForgetByQueryAction",
    "ListAction",
    "MemoryControlAction",
    "MemoryControlDecision",
    "MemoryControlRoute",
    "MemoryControlServiceResult",
    "MemoryControlTarget",
    "SavePreferenceAction",
    "SetRecallAction",
    "StatusAction",
    "TypedMemoryAction",
    "delete_memory_target",
    "detect_memory_control_action",
    "execute_memory_control_action",
    "find_memory_target_by_index",
    "find_memory_targets",
    "is_pending_cancellation",
    "is_pending_confirmation",
    "list_memory_for_owner",
    "parse_memory_control_action",
    "resolve_memory_control_action",
    "save_preference_rule",
    "set_memory_recall",
]
