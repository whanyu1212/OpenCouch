"""Evaluate standalone memory-control node contracts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)
from eval.runners.memory_control_common import (
    EvalRuntime,
    ScriptedTurnDispatchLLM,
    build_eval_state,
    expect_equal,
    grade_store_expectations,
    grade_text_expectations,
    jsonify,
    memory_snapshot,
    optional_mapping,
    seed_memory_store,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "memory_control" / "node_contract_v1.json"
)


@dataclass(frozen=True)
class MemoryControlNodeEvalCase:
    """Parsed standalone memory-control node eval case."""

    id: str
    message: str
    description: str = ""
    memory_mode: str = "local"
    owner_id: str | None = None
    action: dict[str, Any] = field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    memory_seed: dict[str, Any] = field(default_factory=dict)
    scripted: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


class MemoryControlNodeEvaluator(BaseEvaluator[MemoryControlNodeEvalCase]):
    """Run standalone memory-control node contract checks."""

    def __init__(self, *, dataset_path: str | Path) -> None:
        super().__init__(dataset_path=dataset_path, name="memory_control_node")

    def parse_case(self, raw_case: Any) -> MemoryControlNodeEvalCase:
        """Parse one raw eval case."""

        return _parse_case(raw_case)

    def case_id(self, case: MemoryControlNodeEvalCase, index: int) -> str:
        """Return the stable case identifier."""

        return case.id

    async def run_case(self, case: MemoryControlNodeEvalCase) -> EvalResult:
        """Run and grade one standalone node case."""

        artifact = await _invoke_case(case)
        failures = _grade_case(case, artifact)
        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=1.0 if not failures else 0.0,
            details={
                "description": case.description,
                "failures": failures,
                "artifact": artifact,
            },
        )


async def _invoke_case(case: MemoryControlNodeEvalCase) -> dict[str, Any]:
    from agent.memory.modes import MemoryMode
    from agent.memory.store import OpenCouchMemoryStore
    from agent.nodes.memory_control import run_memory_control_node

    owner_id = case.owner_id
    if owner_id is None and not case.expected.get("missing_owner"):
        owner_id = f"eval-user-{case.id}"

    store = OpenCouchMemoryStore()
    if owner_id is not None:
        await seed_memory_store(store, owner_id=owner_id, seed=case.memory_seed)
    state = build_eval_state(
        message=case.message,
        case_id=case.id,
        owner_id=owner_id,
        state_patch=case.state,
    )
    memory_control = dict(state.get("memory_control") or {})
    memory_control["action"] = dict(case.action)
    if case.pending_action is not None:
        memory_control["pending_action"] = dict(case.pending_action)
    state["memory_control"] = memory_control

    mode = MemoryMode.INCOGNITO if case.memory_mode == "incognito" else MemoryMode.LOCAL
    llm_client = _llm_for_case(case)
    snapshot_before = (
        await memory_snapshot(store, owner_id=owner_id) if owner_id is not None else {}
    )
    delta = await run_memory_control_node(
        state,
        EvalRuntime(  # type: ignore[arg-type]
            store=store,
            llm_client=llm_client,
            memory_mode=mode,
        ),
    )
    snapshot_after = (
        await memory_snapshot(store, owner_id=owner_id) if owner_id is not None else {}
    )
    return {
        "delta": jsonify(delta),
        "store_before": snapshot_before,
        "store_after": snapshot_after,
    }


def _llm_for_case(case: MemoryControlNodeEvalCase) -> Any | None:
    if case.action.get("type") != "save_preference":
        return None
    preference_rule_text = case.scripted.get("preference_rule_text")
    if preference_rule_text is None:
        return None
    return ScriptedTurnDispatchLLM(
        {},
        preference_rule_text=str(preference_rule_text),
    )


def _grade_case(
    case: MemoryControlNodeEvalCase,
    artifact: Mapping[str, Any],
) -> list[str]:
    expected = case.expected
    failures: list[str] = []
    delta = artifact.get("delta")
    if not isinstance(delta, Mapping):
        return ["delta is not a mapping"]

    expect_equal(failures, name="route", actual=delta.get("route"), expected=expected)
    expect_equal(
        failures,
        name="response_style",
        actual=delta.get("response_style"),
        expected=expected,
    )
    response_text = str(delta.get("response_text", ""))
    if expected.get("response_text_non_empty") and not response_text.strip():
        failures.append("response_text is empty")
    grade_text_expectations(failures, text=response_text, expected=expected)
    _grade_pending_action(failures, delta=delta, expected=expected)
    _grade_procedural_profile(failures, delta=delta, expected=expected)

    diagnostics = delta.get("diagnostics")
    if expected.get("diagnostics_memory_control_ms") and (
        not isinstance(diagnostics, Mapping) or "memory_control_ms" not in diagnostics
    ):
        failures.append("diagnostics.memory_control_ms is missing")

    store_after = artifact.get("store_after")
    if isinstance(store_after, Mapping):
        grade_store_expectations(
            failures,
            snapshot=store_after,
            expected=expected,
            prefix="store_after",
        )
    return failures


def _grade_pending_action(
    failures: list[str],
    *,
    delta: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    memory_control = delta.get("memory_control")
    if not isinstance(memory_control, Mapping):
        if "pending_action_is_null" in expected:
            failures.append("memory_control is not a mapping")
        return

    pending = memory_control.get("pending_action")
    if "pending_action_is_null" in expected:
        expected_null = bool(expected["pending_action_is_null"])
        if expected_null and pending is not None:
            failures.append(f"pending_action: expected null, got {pending!r}")
        if not expected_null and pending is None:
            failures.append("pending_action unexpectedly null")

    if "pending_action_type" in expected:
        actual_type = pending.get("type") if isinstance(pending, Mapping) else None
        if actual_type != expected["pending_action_type"]:
            failures.append(
                "pending_action.type: expected "
                f"{expected['pending_action_type']!r}, got {actual_type!r}"
            )

    target = pending.get("target") if isinstance(pending, Mapping) else None
    if "pending_target_kind" in expected:
        actual_kind = target.get("kind") if isinstance(target, Mapping) else None
        if actual_kind != expected["pending_target_kind"]:
            failures.append(
                "pending_action.target.kind: expected "
                f"{expected['pending_target_kind']!r}, got {actual_kind!r}"
            )
    if "pending_target_preview_contains" in expected:
        preview = str(target.get("preview", "")) if isinstance(target, Mapping) else ""
        grade_text_expectations(
            failures,
            text=preview,
            expected={
                "response_text_contains": expected["pending_target_preview_contains"]
            },
        )


def _grade_procedural_profile(
    failures: list[str],
    *,
    delta: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    profile = delta.get("procedural_profile")
    if "procedural_profile_proactive_recall_enabled" not in expected:
        return
    actual = (
        profile.get("proactive_recall_enabled")
        if isinstance(profile, Mapping)
        else None
    )
    expected_value = expected["procedural_profile_proactive_recall_enabled"]
    if actual != expected_value:
        failures.append(
            "procedural_profile.proactive_recall_enabled: expected "
            f"{expected_value!r}, got {actual!r}"
        )


def _parse_case(raw_case: Any) -> MemoryControlNodeEvalCase:
    if not isinstance(raw_case, Mapping):
        raise TypeError("Memory-control node cases must be JSON objects.")
    pending = raw_case.get("pending_action")
    if pending is not None and not isinstance(pending, Mapping):
        raise TypeError("pending_action must be an object when provided.")
    return MemoryControlNodeEvalCase(
        id=str(raw_case["id"]),
        description=str(raw_case.get("description", "")),
        message=str(raw_case["message"]),
        memory_mode=str(raw_case.get("memory_mode", "local")),
        owner_id=(
            str(raw_case["owner_id"]) if raw_case.get("owner_id") is not None else None
        ),
        action=dict(optional_mapping(raw_case, "action")),
        pending_action=dict(pending) if isinstance(pending, Mapping) else None,
        memory_seed=dict(optional_mapping(raw_case, "memory_seed")),
        scripted=dict(optional_mapping(raw_case, "scripted")),
        state=dict(optional_mapping(raw_case, "state")),
        expected=dict(optional_mapping(raw_case, "expected")),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate standalone memory-control node behavior.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    return parser


def main() -> int:
    """Run the standalone memory-control node evaluator CLI."""

    return run_evaluator_cli(
        lambda args: MemoryControlNodeEvaluator(dataset_path=args.dataset),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
