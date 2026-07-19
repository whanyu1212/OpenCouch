"""Tests for diagnostics merge helpers."""

from __future__ import annotations

from agent.observability.diagnostics import (
    diagnostics_from_state,
    merge_diagnostics,
    merge_state_diagnostics,
    normalize_diagnostics,
    replace_state_diagnostics,
)


def test_normalize_diagnostics_returns_copy_for_mappings() -> None:
    source = {"a": 1}

    diagnostics = normalize_diagnostics(source)

    assert diagnostics == {"a": 1}
    assert diagnostics is not source


def test_normalize_diagnostics_ignores_non_mappings() -> None:
    assert normalize_diagnostics(None) == {}
    assert normalize_diagnostics(["not", "a", "mapping"]) == {}


def test_diagnostics_from_state_handles_missing_or_invalid_values() -> None:
    assert diagnostics_from_state({}) == {}
    assert diagnostics_from_state({"diagnostics": None}) == {}
    assert diagnostics_from_state({"diagnostics": {"a": 1}}) == {"a": 1}


def test_merge_diagnostics_preserves_existing_and_overrides_updates() -> None:
    existing = {"a": 1, "b": 2}
    updates = {"b": 3, "c": 4}

    assert merge_diagnostics(existing, updates) == {"a": 1, "b": 3, "c": 4}
    assert existing == {"a": 1, "b": 2}


def test_merge_state_diagnostics_initializes_and_merges_in_place() -> None:
    state = {"diagnostics": {"a": 1}}

    merge_state_diagnostics(state, {"b": 2})

    assert state["diagnostics"] == {"a": 1, "b": 2}


def test_merge_state_diagnostics_recovers_from_invalid_existing_value() -> None:
    state = {"diagnostics": ["bad"]}

    merge_state_diagnostics(state, {"ok": True})

    assert state["diagnostics"] == {"ok": True}


def test_replace_state_diagnostics_drops_stale_existing_keys() -> None:
    state = {"diagnostics": {"stale": True, "keep": False}}

    replace_state_diagnostics(state, {"keep": True})

    assert state["diagnostics"] == {"keep": True}
