"""Typed memory-control action payloads.

The graph stores memory-control actions as plain dicts for LangGraph state
serialization. These models give service and routing code one validated shape
to produce and consume.
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
    rule_text: str = Field(min_length=1)


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


def parse_memory_control_action(payload: dict[str, Any]) -> TypedMemoryAction | None:
    """Parse a graph-state action dict into a typed memory-control action.

    Args:
        payload (dict[str, Any]): Action dict carried on graph state.

    Returns:
        TypedMemoryAction | None: Parsed action, or ``None`` when the payload is
            missing a known discriminator or fails validation.
    """

    if not payload or "type" not in payload:
        return None
    try:
        return _TYPED_ACTION_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


@dataclass(frozen=True)
class MemoryControlAction:
    """Resolved memory-management action."""

    payload: dict[str, Any]

    def to_state_action(self) -> dict[str, Any]:
        """Return a serializable action for graph state updates.

        Returns:
            dict[str, Any]: Serializable memory-management action payload.
        """

        return dict(self.payload)

    def parsed(self) -> TypedMemoryAction | None:
        """Return the typed action model.

        Returns:
            TypedMemoryAction | None: Parsed action, or ``None`` when invalid.
        """

        return parse_memory_control_action(self.payload)


def normalize_preference_rule(rule_text: str) -> str:
    """Normalize an LLM-proposed preference rule for persistence.

    Args:
        rule_text (str): Raw model-proposed procedural rule.

    Returns:
        str: A trimmed, sentence-like rule.
    """

    normalized = " ".join(rule_text.strip().split()).strip("\"'")
    if not normalized:
        return ""
    if not normalized.endswith((".", "!", "?")):
        normalized = f"{normalized}."
    if normalized.lower().startswith(("you ", "your ", "when ", "if ")):
        return normalized
    return f"You prefer {normalized[0].lower()}{normalized[1:]}"


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
    "normalize_preference_rule",
    "parse_memory_control_action",
]
