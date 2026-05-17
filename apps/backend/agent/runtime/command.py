"""Small routing-command value used by compatibility node adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

NextNodeT = TypeVar("NextNodeT")


@dataclass(frozen=True)
class RuntimeCommand(Generic[NextNodeT]):
    """State update plus the next node identifier."""

    update: dict[str, Any]
    goto: NextNodeT


__all__ = ["RuntimeCommand"]
