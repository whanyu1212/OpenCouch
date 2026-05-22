"""Typed memory-control action payloads for SDK tool requests."""

from __future__ import annotations

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
    """Parse an explicit SDK tool payload into a typed memory-control action.

    Args:
        payload (dict[str, Any]): Action payload requested by a memory tool.

    Returns:
        TypedMemoryAction: Parsed action.

    Raises:
        ValueError: If the payload is missing a discriminator or fails validation.
    """

    if not payload or "type" not in payload:
        raise ValueError("Memory action payload requires a type.")
    try:
        return _TYPED_ACTION_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise ValueError("Invalid memory action payload.") from exc


__all__ = [
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
]
