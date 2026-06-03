"""Shared helpers for AgentState diagnostics mappings."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

DIAGNOSTICS_KEY = "diagnostics"


def normalize_diagnostics(value: Any) -> dict[str, Any]:
    """Return a mutable diagnostics mapping from an arbitrary value."""

    if not isinstance(value, Mapping):
        return {}
    return dict(value)


def diagnostics_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable copy of ``state["diagnostics"]`` when present."""

    return normalize_diagnostics(state.get(DIAGNOSTICS_KEY))


def merge_diagnostics(
    existing: Mapping[str, Any] | None,
    updates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return diagnostics with ``updates`` shallow-merged over ``existing``."""

    diagnostics = normalize_diagnostics(existing)
    if isinstance(updates, Mapping):
        diagnostics.update(dict(updates))
    return diagnostics


def merge_state_diagnostics(
    state: MutableMapping[str, Any],
    updates: Mapping[str, Any] | None,
) -> None:
    """Merge diagnostics updates into state in place."""

    state[DIAGNOSTICS_KEY] = merge_diagnostics(
        normalize_diagnostics(state.get(DIAGNOSTICS_KEY)),
        updates,
    )


def replace_state_diagnostics(
    state: MutableMapping[str, Any],
    diagnostics: Mapping[str, Any] | None,
) -> None:
    """Replace state diagnostics with a normalized diagnostics mapping."""

    state[DIAGNOSTICS_KEY] = normalize_diagnostics(diagnostics)
