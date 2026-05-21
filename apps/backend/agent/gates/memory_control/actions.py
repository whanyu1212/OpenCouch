"""Typed memory-control action payloads.

The runtime stores memory-control actions as plain dicts in app-owned state.
These models give service and routing code one validated shape to produce and
consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError


class _ActionBase(BaseModel):
    """Base for typed memory-control actions."""

    model_config = {"frozen": True, "extra": "ignore"}


class ListAction(_ActionBase):
    type: Literal["list"] = "list"


class StatusAction(_ActionBase):
    type: Literal["status"] = "status"


class SetRecallAction(_ActionBase):
    type: Literal["set_recall"] = "set_recall"
    enabled: bool


class SavePreferenceAction(_ActionBase):
    type: Literal["save_preference"] = "save_preference"
    preference_text: str = Field(min_length=1)


class ForgetByIndexAction(_ActionBase):
    type: Literal["forget_by_index"] = "forget_by_index"
    target_kind: Literal["fact", "session", "rule"]
    target_index: int = Field(ge=1)


class ForgetByQueryAction(_ActionBase):
    type: Literal["forget_by_query"] = "forget_by_query"
    query: str = Field(min_length=1)


class ConfirmPendingAction(_ActionBase):
    type: Literal["confirm_pending"] = "confirm_pending"


class CancelPendingAction(_ActionBase):
    type: Literal["cancel_pending"] = "cancel_pending"


TypedMemoryAction = Annotated[
    Union[
        ListAction,
        StatusAction,
        SetRecallAction,
        SavePreferenceAction,
        ForgetByIndexAction,
        ForgetByQueryAction,
        ConfirmPendingAction,
        CancelPendingAction,
    ],
    Field(discriminator="type"),
]

_TYPED_ACTION_ADAPTER: TypeAdapter[TypedMemoryAction] = TypeAdapter(TypedMemoryAction)


def parse_memory_control_action(payload: dict[str, Any]) -> TypedMemoryAction:
    """Parse a runtime-state action dict into a typed memory-control action.

    Args:
        payload (dict[str, Any]): Action dict carried on runtime state.

    Returns:
        TypedMemoryAction: Parsed action.

    Raises:
        ValueError: If the payload is missing a discriminator or fails validation.
    """

    if not payload or "type" not in payload:
        raise ValueError("memory_control.action requires a type.")
    try:
        return _TYPED_ACTION_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ValueError("Invalid memory_control.action payload.") from exc


@dataclass(frozen=True)
class MemoryControlAction:
    """Resolved memory-management action."""

    payload: dict[str, Any]

    def to_state_action(self) -> dict[str, Any]:
        """Return a serializable action for runtime state updates.

        Returns:
            dict[str, Any]: Serializable memory-management action payload.
        """

        return dict(self.payload)

    def parsed(self) -> TypedMemoryAction:
        """Return the typed action model.

        Returns:
            TypedMemoryAction: Parsed action.
        """

        return parse_memory_control_action(self.payload)


__all__ = [
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
]
