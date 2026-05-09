"""Evaluate multi-turn memory-control conversation trajectories."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.judges.rubric import RubricDimension, RubricJudgeArtifact, RubricLLMJudge
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
    list_of_mappings,
    memory_snapshot,
    optional_mapping,
    seed_memory_store,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "memory_control" / "trajectory_v1.json"
)
_SHALLOW_DICT_CHANNELS = {
    "crisis",
    "crisis_audit",
    "diagnostics",
    "exercise_state",
    "grounded_lookup",
    "memory_control",
    "procedural_profile",
    "session_memory",
}


@dataclass(frozen=True)
class MemoryControlTrajectoryTurn:
    """One turn in a memory-control trajectory case."""

    message: str
    description: str = ""
    scripted: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryControlTrajectoryCase:
    """Parsed multi-turn memory-control trajectory case."""

    id: str
    description: str = ""
    memory_seed: dict[str, Any] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    turns: list[MemoryControlTrajectoryTurn] = field(default_factory=list)
    rubric: dict[str, Any] = field(default_factory=dict)


class MemoryControlTrajectoryEvaluator(BaseEvaluator[MemoryControlTrajectoryCase]):
    """Run multi-turn memory-control routing and state checks."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        mode: str,
        judge_mode: str,
        min_judge_score: float | None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"memory_control_trajectory_{mode}_{judge_mode}",
        )
        self.mode = mode
        self.judge_mode = judge_mode
        self.min_judge_score = min_judge_score
        self._live_llm: Any | None = None

    def parse_case(self, raw_case: Any) -> MemoryControlTrajectoryCase:
        """Parse one raw trajectory case."""

        return _parse_case(raw_case)

    def case_id(self, case: MemoryControlTrajectoryCase, index: int) -> str:
        """Return the stable case identifier."""

        return case.id

    async def run_case(self, case: MemoryControlTrajectoryCase) -> EvalResult:
        """Run and grade one multi-turn trajectory."""

        artifact = await self._run_trajectory(case)
        hard_failures = _grade_trajectory(case, artifact)
        failures = list(hard_failures)
        score = 1.0 if not failures else 0.0
        judge_details: dict[str, Any] | None = None

        if self.judge_mode == "live":
            judge_outcome = await _judge_trajectory(
                case,
                artifact,
                hard_failures=hard_failures,
                min_score=self._min_score_for_case(case),
            )
            judge_details = judge_outcome.to_dict()
            failures = judge_outcome.failures
            score = judge_outcome.score

        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=score,
            details={
                "description": case.description,
                "mode": self.mode,
                "judge_mode": self.judge_mode,
                "failures": failures,
                "judge": judge_details,
                "artifact": artifact,
            },
        )

    async def _run_trajectory(
        self,
        case: MemoryControlTrajectoryCase,
    ) -> dict[str, Any]:
        from agent.memory.store import OpenCouchMemoryStore

        owner_id = f"eval-user-{case.id}"
        store = OpenCouchMemoryStore()
        await seed_memory_store(store, owner_id=owner_id, seed=case.memory_seed)
        initial_store = await memory_snapshot(store, owner_id=owner_id)
        carried_state = dict(case.initial_state)
        history: list[dict[str, str]] = []
        turns: list[dict[str, Any]] = []

        for index, turn in enumerate(case.turns):
            state = build_eval_state(
                message=turn.message,
                case_id=case.id,
                owner_id=owner_id,
                history=history,
                state_patch=turn.state,
            )
            _merge_graph_delta(state, carried_state)
            turn_artifact = await self._run_turn(
                turn,
                state=state,
                store=store,
                turn_index=index + 1,
            )
            _carry_forward(carried_state, turn_artifact["state_after"])
            turns.append(turn_artifact)
            _append_history(history, turn, turn_artifact)

        return {
            "case_id": case.id,
            "description": case.description,
            "initial_store": initial_store,
            "turns": turns,
            "final_store": await memory_snapshot(store, owner_id=owner_id),
        }

    async def _run_turn(
        self,
        turn: MemoryControlTrajectoryTurn,
        *,
        state: dict[str, Any],
        store: Any,
        turn_index: int,
    ) -> dict[str, Any]:
        from agent.graph_constants import LOAD_MEMORY_NODE, MEMORY_CONTROL_NODE
        from agent.nodes.finalize_turn import run_finalize_turn_node
        from agent.nodes.load_memory import run_load_memory_node
        from agent.nodes.memory_control import run_memory_control_node
        from agent.nodes.turn_dispatch import run_turn_dispatch_node

        llm = self._llm_for_turn(turn)
        runtime = EvalRuntime(store=store, llm_client=llm)
        dispatch_command = await run_turn_dispatch_node(
            state,
            runtime,  # type: ignore[arg-type]
        )
        dispatch_delta = dict(dispatch_command.update or {})
        _merge_graph_delta(state, dispatch_delta)

        branch_delta: dict[str, Any] = {}
        finalize_delta: dict[str, Any] = {}
        if dispatch_command.goto == MEMORY_CONTROL_NODE:
            branch_delta = dict(
                await run_memory_control_node(
                    state,
                    runtime,  # type: ignore[arg-type]
                )
            )
            _merge_graph_delta(state, branch_delta)
            finalize_delta = dict(
                await run_finalize_turn_node(
                    state,
                    runtime,  # type: ignore[arg-type]
                )
            )
            _merge_graph_delta(state, finalize_delta)
        elif dispatch_command.goto == LOAD_MEMORY_NODE:
            branch_delta = dict(
                await run_load_memory_node(
                    state,
                    runtime,  # type: ignore[arg-type]
                )
            )
            _merge_graph_delta(state, branch_delta)

        owner_id = str(state.get("user_id") or state.get("session_id"))
        return {
            "turn_index": turn_index,
            "user_message": turn.message,
            "goto": dispatch_command.goto,
            "dispatch_delta": jsonify(dispatch_delta),
            "branch_delta": jsonify(branch_delta),
            "finalize_delta": jsonify(finalize_delta),
            "response_text": str(state.get("response_text", "")),
            "response_style": state.get("response_style"),
            "state_after": jsonify(state),
            "store_after": await memory_snapshot(store, owner_id=owner_id),
            "structured_calls": getattr(llm, "structured_calls", {}),
        }

    def _llm_for_turn(self, turn: MemoryControlTrajectoryTurn) -> Any:
        if self.mode == "live":
            if self._live_llm is None:
                from config import create_configured_control_llm_client

                self._live_llm = create_configured_control_llm_client()
            return self._live_llm

        decision = optional_mapping(turn.scripted, "decision")
        if not decision:
            raise ValueError("Scripted trajectory turns need scripted.decision.")
        preference_rule_text = turn.scripted.get("preference_rule_text")
        return ScriptedTurnDispatchLLM(
            decision,
            preference_rule_text=(
                str(preference_rule_text) if preference_rule_text is not None else None
            ),
        )

    def _min_score_for_case(self, case: MemoryControlTrajectoryCase) -> float:
        if self.min_judge_score is not None:
            return self.min_judge_score
        return float(case.rubric.get("min_judge_score", 0.8))


def _grade_trajectory(
    case: MemoryControlTrajectoryCase,
    artifact: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    turns = artifact.get("turns")
    if not isinstance(turns, list):
        return ["trajectory turns is not a list"]

    for index, turn_case in enumerate(case.turns):
        if index >= len(turns) or not isinstance(turns[index], Mapping):
            failures.append(f"turn {index + 1}: missing artifact")
            continue
        _grade_turn(failures, turn_case=turn_case, artifact=turns[index])

    final_store = artifact.get("final_store")
    if isinstance(final_store, Mapping):
        grade_store_expectations(
            failures,
            snapshot=final_store,
            expected=case.expected,
            prefix="final_store",
        )
    return failures


def _grade_turn(
    failures: list[str],
    *,
    turn_case: MemoryControlTrajectoryTurn,
    artifact: Mapping[str, Any],
) -> None:
    expected = turn_case.expected
    label = f"turn {artifact.get('turn_index')}"
    expect_equal(failures, name="goto", actual=artifact.get("goto"), expected=expected)
    expect_equal(
        failures,
        name="response_style",
        actual=artifact.get("response_style"),
        expected=expected,
    )
    response_text = str(artifact.get("response_text", ""))
    if expected.get("response_text_non_empty") and not response_text.strip():
        failures.append(f"{label}: response_text is empty")
    grade_text_expectations(failures, text=response_text, expected=expected)

    dispatch_delta = artifact.get("dispatch_delta")
    if isinstance(dispatch_delta, Mapping):
        expect_equal(
            failures,
            name="route",
            actual=dispatch_delta.get("route"),
            expected=expected,
        )
        _grade_memory_action(failures, dispatch_delta=dispatch_delta, expected=expected)
        _grade_dispatch_pending_action(
            failures,
            dispatch_delta=dispatch_delta,
            expected=expected,
        )

    branch_delta = artifact.get("branch_delta")
    if isinstance(branch_delta, Mapping):
        _grade_pending_action(failures, delta=branch_delta, expected=expected)
        _grade_load_memory(failures, delta=branch_delta, expected=expected)

    store_after = artifact.get("store_after")
    if isinstance(store_after, Mapping):
        grade_store_expectations(
            failures,
            snapshot=store_after,
            expected=expected,
            prefix="store_after",
        )


def _grade_memory_action(
    failures: list[str],
    *,
    dispatch_delta: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    memory_control = dispatch_delta.get("memory_control")
    if not isinstance(memory_control, Mapping):
        if "memory_action_type" in expected:
            failures.append("dispatch_delta.memory_control is not a mapping")
        return
    action = memory_control.get("action")
    actual_type = action.get("type") if isinstance(action, Mapping) else None
    if (
        "memory_action_type" in expected
        and actual_type != expected["memory_action_type"]
    ):
        failures.append(
            "memory_action.type: expected "
            f"{expected['memory_action_type']!r}, got {actual_type!r}"
        )
    for field_name in ("query", "preference_text"):
        key = f"memory_action_{field_name}_contains"
        if key not in expected:
            continue
        actual = str(action.get(field_name, "")) if isinstance(action, Mapping) else ""
        required = expected[key] if isinstance(expected[key], list) else [expected[key]]
        for phrase in required:
            if str(phrase).casefold() not in actual.casefold():
                failures.append(f"memory_action.{field_name} missing {phrase!r}")


def _grade_dispatch_pending_action(
    failures: list[str],
    *,
    dispatch_delta: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if "dispatch_pending_action_is_null" not in expected:
        return
    memory_control = dispatch_delta.get("memory_control")
    pending = (
        memory_control.get("pending_action")
        if isinstance(memory_control, Mapping)
        else "__missing__"
    )
    expected_null = bool(expected["dispatch_pending_action_is_null"])
    if expected_null and pending is not None:
        failures.append(f"dispatch pending_action: expected null, got {pending!r}")
    if not expected_null and pending is None:
        failures.append("dispatch pending_action unexpectedly null")


def _grade_pending_action(
    failures: list[str],
    *,
    delta: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    memory_control = delta.get("memory_control")
    if not isinstance(memory_control, Mapping):
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


def _grade_load_memory(
    failures: list[str],
    *,
    delta: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    profile = delta.get("procedural_profile")
    if "load_memory_proactive_recall_enabled" in expected:
        actual = (
            profile.get("proactive_recall_enabled")
            if isinstance(profile, Mapping)
            else None
        )
        if actual != expected["load_memory_proactive_recall_enabled"]:
            failures.append(
                "load_memory.proactive_recall_enabled: expected "
                f"{expected['load_memory_proactive_recall_enabled']!r}, got {actual!r}"
            )
    if "load_memory_rules_contain" in expected:
        rules = profile.get("procedural_rules") if isinstance(profile, Mapping) else []
        haystack = "\n".join(str(rule) for rule in (rules or [])).casefold()
        required = expected["load_memory_rules_contain"]
        phrases = required if isinstance(required, list) else [required]
        for phrase in phrases:
            if str(phrase).casefold() not in haystack:
                failures.append(f"load_memory.procedural_rules missing {phrase!r}")


def _carry_forward(carried_state: dict[str, Any], state: Mapping[str, Any]) -> None:
    for key in (
        "memory_control",
        "procedural_profile",
        "session_memory",
        "exercise_state",
        "grounded_lookup",
    ):
        value = state.get(key)
        if isinstance(value, Mapping):
            carried_state[key] = dict(value)


def _merge_graph_delta(state: dict[str, Any], delta: Mapping[str, Any]) -> None:
    for key, value in delta.items():
        if key == "transcript" and isinstance(value, list):
            existing = state.get("transcript")
            prefix = list(existing) if isinstance(existing, list) else []
            state[key] = [*prefix, *value]
            continue

        if key in _SHALLOW_DICT_CHANNELS and isinstance(value, Mapping):
            existing = state.get(key)
            base = dict(existing) if isinstance(existing, Mapping) else {}
            state[key] = {**base, **dict(value)}
            continue

        state[key] = value


def _append_history(
    history: list[dict[str, str]],
    turn: MemoryControlTrajectoryTurn,
    artifact: Mapping[str, Any],
) -> None:
    history.append({"role": "user", "content": turn.message})
    assistant_text = str(
        artifact.get("response_text")
        or turn.expected.get("assistant_history_text")
        or "I hear you. Let's stay with that."
    )
    history.append(
        {
            "role": "assistant",
            "content": assistant_text,
            "response_style": str(artifact.get("response_style") or ""),
        }
    )


async def _judge_trajectory(
    case: MemoryControlTrajectoryCase,
    artifact: Mapping[str, Any],
    *,
    hard_failures: list[str],
    min_score: float,
) -> Any:
    from config import create_configured_control_llm_client

    judge = RubricLLMJudge(llm_client=create_configured_control_llm_client())
    verdict = await judge.judge(
        RubricJudgeArtifact(
            task=(
                "Judge memory-control conversation quality. The workflow should "
                "inspect or modify saved assistant memory only when the user "
                "explicitly asks for memory administration."
            ),
            input={
                "case_id": case.id,
                "description": case.description,
                "expected": case.expected,
                "hard_check_note": (
                    "Hard checks already verify route, state, and store "
                    "invariants. Judge qualitative clarity and memory-control "
                    "behavior from the summarized output only."
                ),
            },
            output=_judge_output(
                case,
                artifact,
                hard_failures=hard_failures,
            ),
            rubric=_rubric_dimensions(case),
            hard_failures=hard_failures,
        )
    )
    return judge.combine(
        verdict=verdict,
        hard_failures=hard_failures,
        min_score=min_score,
    )


def _judge_output(
    case: MemoryControlTrajectoryCase,
    artifact: Mapping[str, Any],
    *,
    hard_failures: list[str],
) -> dict[str, Any]:
    turns = artifact.get("turns")
    turn_artifacts = turns if isinstance(turns, list) else []
    return {
        "case_id": artifact.get("case_id"),
        "description": artifact.get("description"),
        "hard_checks": {"passed": not hard_failures, "failures": hard_failures},
        "notes": [
            "Therapeutic turns in this eval stop after load_memory_node; "
            "assistant response quality is not evaluated for those turns.",
            "pending_action_after_turn is the authoritative pending memory-control "
            "state after each turn.",
        ],
        "turns": [
            _judge_turn(
                case_turn=case.turns[index] if index < len(case.turns) else None,
                artifact=turn_artifact,
            )
            for index, turn_artifact in enumerate(turn_artifacts)
            if isinstance(turn_artifact, Mapping)
        ],
        "final_store": artifact.get("final_store"),
    }


def _judge_turn(
    *,
    case_turn: MemoryControlTrajectoryTurn | None,
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    dispatch_delta = artifact.get("dispatch_delta")
    dispatch = dispatch_delta if isinstance(dispatch_delta, Mapping) else {}
    dispatch_memory = dispatch.get("memory_control")
    dispatch_memory_control = (
        dispatch_memory if isinstance(dispatch_memory, Mapping) else {}
    )
    branch_delta = artifact.get("branch_delta")
    branch = branch_delta if isinstance(branch_delta, Mapping) else {}
    branch_memory = branch.get("memory_control")
    branch_memory_control = branch_memory if isinstance(branch_memory, Mapping) else {}
    state_after = artifact.get("state_after")
    state = state_after if isinstance(state_after, Mapping) else {}
    state_memory = state.get("memory_control")
    state_memory_control = state_memory if isinstance(state_memory, Mapping) else {}
    response_style = artifact.get("response_style")

    return {
        "turn_index": artifact.get("turn_index"),
        "description": case_turn.description if case_turn is not None else "",
        "user_message": artifact.get("user_message"),
        "expected": case_turn.expected if case_turn is not None else {},
        "goto": artifact.get("goto"),
        "route": dispatch.get("route"),
        "dispatch_memory_action": dispatch_memory_control.get("action"),
        "dispatch_pending_action": dispatch_memory_control.get("pending_action"),
        "branch_pending_action": branch_memory_control.get("pending_action"),
        "pending_action_after_turn": state_memory_control.get("pending_action"),
        "response_style": response_style,
        "memory_control_response": (
            artifact.get("response_text")
            if response_style == "memory_control"
            else None
        ),
        "load_memory": _judge_load_memory(branch),
        "store_after": artifact.get("store_after"),
    }


def _judge_load_memory(branch_delta: Mapping[str, Any]) -> dict[str, Any] | None:
    profile = branch_delta.get("procedural_profile")
    session_memory = branch_delta.get("session_memory")
    if not isinstance(profile, Mapping) and not isinstance(session_memory, Mapping):
        return None

    working_memory = (
        session_memory.get("working_memory")
        if isinstance(session_memory, Mapping)
        else []
    )
    rules = profile.get("procedural_rules") if isinstance(profile, Mapping) else []
    return {
        "response_not_evaluated": True,
        "working_memory_count": len(working_memory)
        if isinstance(working_memory, list)
        else 0,
        "procedural_profile": {
            "proactive_recall_enabled": profile.get("proactive_recall_enabled")
            if isinstance(profile, Mapping)
            else None,
            "procedural_rules": rules if isinstance(rules, list) else [],
        },
    }


def _rubric_dimensions(
    case: MemoryControlTrajectoryCase,
) -> list[RubricDimension]:
    raw_dimensions = case.rubric.get("dimensions")
    if isinstance(raw_dimensions, list) and raw_dimensions:
        return [RubricDimension.model_validate(item) for item in raw_dimensions]
    dimensions = [
        RubricDimension(
            name="memory_boundary",
            question=(
                "Does the trajectory keep ordinary support turns out of memory "
                "administration while routing explicit memory commands correctly? "
                "Treat the hard route checks as authoritative."
            ),
        ),
        RubricDimension(
            name="no_overclaiming",
            question=(
                "Do responses avoid claiming saved memories that are absent from "
                "the seeded store?"
            ),
        ),
        RubricDimension(
            name="operational_clarity",
            question=(
                "Are memory-control responses concise and clear about what changed?"
            ),
        ),
    ]
    if _needs_destructive_consent_dimension(case):
        dimensions.insert(
            1,
            RubricDimension(
                name="destructive_action_consent",
                question=(
                    "Does deletion require clear confirmation before any saved "
                    "memory is removed? Use pending_action_after_turn and store "
                    "snapshots instead of inferring from omitted raw graph fields."
                ),
            ),
        )
    return dimensions


def _needs_destructive_consent_dimension(case: MemoryControlTrajectoryCase) -> bool:
    destructive_action_types = {
        "forget_by_index",
        "forget_by_query",
        "confirm_pending",
        "cancel_pending",
    }
    for turn in case.turns:
        if turn.expected.get("pending_action_type") == "delete":
            return True
        action_type = turn.expected.get("memory_action_type")
        if action_type in destructive_action_types:
            return True
        decision = optional_mapping(turn.scripted, "decision")
        if decision.get("memory_action_type") in destructive_action_types:
            return True
    return False


def _parse_case(raw_case: Any) -> MemoryControlTrajectoryCase:
    if not isinstance(raw_case, Mapping):
        raise TypeError("Memory-control trajectory cases must be JSON objects.")
    return MemoryControlTrajectoryCase(
        id=str(raw_case["id"]),
        description=str(raw_case.get("description", "")),
        memory_seed=dict(optional_mapping(raw_case, "memory_seed")),
        initial_state=dict(optional_mapping(raw_case, "initial_state")),
        expected=dict(optional_mapping(raw_case, "expected")),
        rubric=dict(optional_mapping(raw_case, "rubric")),
        turns=[
            _parse_turn(item)
            for item in list_of_mappings(raw_case.get("turns", []), "turns")
        ],
    )


def _parse_turn(raw_turn: Mapping[str, Any]) -> MemoryControlTrajectoryTurn:
    return MemoryControlTrajectoryTurn(
        message=str(raw_turn["message"]),
        description=str(raw_turn.get("description", "")),
        scripted=dict(optional_mapping(raw_turn, "scripted")),
        state=dict(optional_mapping(raw_turn, "state")),
        expected=dict(optional_mapping(raw_turn, "expected")),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate memory-control multi-turn trajectories.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="scripted uses fixture routing decisions; live uses configured LLM.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("off", "live"),
        default="off",
        help="off uses hard checks only; live adds LLM-as-judge scoring.",
    )
    parser.add_argument(
        "--min-judge-score",
        type=float,
        default=None,
        help="Override the minimum LLM judge score.",
    )
    return parser


def main() -> int:
    """Run the memory-control trajectory evaluator CLI."""

    return run_evaluator_cli(
        lambda args: MemoryControlTrajectoryEvaluator(
            dataset_path=args.dataset,
            mode=args.mode,
            judge_mode=args.judge_mode,
            min_judge_score=args.min_judge_score,
        ),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
