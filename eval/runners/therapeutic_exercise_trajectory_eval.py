"""Evaluate multi-turn guided-exercise trajectories."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.judges import (
    ExerciseTrajectoryArtifact,
    ExerciseTrajectoryJudge,
    ExerciseTrajectoryTurnArtifact,
)
from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)
from eval.runners.therapeutic_common import (
    ScriptedTherapeuticLLM,
    TherapeuticEvalCase,
    build_live_therapeutic_llms,
    deep_update,
    grade_therapeutic_output,
    last_routing_decision,
    parse_therapeutic_case,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "therapeutic" / "exercise_trajectory_v1.json"
)
_MENU_PHRASES = (
    "which would you like",
    "choose one",
    "option 1",
    "option 2",
)


@dataclass(frozen=True)
class ExerciseTrajectoryTurnCase:
    """One user turn in a trajectory eval case."""

    message: str
    case: TherapeuticEvalCase
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExerciseTrajectoryCase:
    """Parsed multi-turn exercise trajectory case."""

    id: str
    description: str = ""
    initial_state: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    turns: list[ExerciseTrajectoryTurnCase] = field(default_factory=list)


class TherapeuticExerciseTrajectoryEvaluator(BaseEvaluator[ExerciseTrajectoryCase]):
    """Run guided-exercise trajectory checks and optional LLM judging."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        mode: str,
        judge_mode: str,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"therapeutic_exercise_trajectory_{mode}_{judge_mode}",
        )
        self.mode = mode
        self.judge_mode = judge_mode
        self._live_llms: tuple[Any, Any] | None = None

    def parse_case(self, raw_case: Any) -> ExerciseTrajectoryCase:
        """Parse one exercise trajectory case.

        Args:
            raw_case (Any): Raw JSON case.

        Returns:
            ExerciseTrajectoryCase: Parsed trajectory case.
        """

        if not isinstance(raw_case, Mapping):
            raise TypeError("Exercise trajectory cases must be JSON objects.")
        raw_turns = raw_case.get("turns")
        if not isinstance(raw_turns, list):
            raise TypeError("'turns' must be a list.")

        case_id = str(raw_case["id"])
        turns: list[ExerciseTrajectoryTurnCase] = []
        for index, raw_turn in enumerate(raw_turns):
            if not isinstance(raw_turn, Mapping):
                raise TypeError("'turns' entries must be objects.")
            expected = _optional_mapping(raw_turn, "expected")
            therapeutic_case = parse_therapeutic_case(
                {
                    "id": f"{case_id}:turn-{index + 1}",
                    "message": raw_turn["message"],
                    "scripted": raw_turn.get("scripted"),
                    "expected": expected,
                }
            )
            turns.append(
                ExerciseTrajectoryTurnCase(
                    message=str(raw_turn["message"]),
                    case=therapeutic_case,
                    expected=dict(expected),
                )
            )

        return ExerciseTrajectoryCase(
            id=case_id,
            description=str(raw_case.get("description", "")),
            initial_state=dict(_optional_mapping(raw_case, "initial_state")),
            expected=dict(_optional_mapping(raw_case, "expected")),
            turns=turns,
        )

    def case_id(self, case: ExerciseTrajectoryCase, index: int) -> str:
        """Return the trajectory case id.

        Args:
            case (ExerciseTrajectoryCase): Parsed case.
            index (int): Zero-based position.

        Returns:
            str: Stable case id.
        """

        return case.id

    async def run_case(self, case: ExerciseTrajectoryCase) -> EvalResult:
        """Run and grade one trajectory case.

        Args:
            case (ExerciseTrajectoryCase): Parsed case.

        Returns:
            EvalResult: Case result.
        """

        trajectory = await self._run_trajectory(case)
        hard_failures = _grade_trajectory(case, trajectory)
        artifact = _build_judge_artifact(
            case=case,
            trajectory=trajectory,
            hard_failures=hard_failures,
        )

        judge_details: dict[str, Any] | None = None
        score = 1.0 if not hard_failures else 0.0
        passed = not hard_failures
        failures = list(hard_failures)

        if self.judge_mode == "live":
            judge_llm = self._get_live_llms()[0]
            judge = ExerciseTrajectoryJudge(llm_client=judge_llm)
            verdict = await judge.judge(artifact)
            outcome = judge.combine(
                verdict=verdict,
                hard_failures=hard_failures,
                min_score=float(case.expected.get("min_judge_score", 0.75)),
            )
            passed = outcome.passed
            score = outcome.score
            failures = outcome.failures
            judge_details = outcome.to_dict()

        return EvalResult(
            case_id=case.id,
            passed=passed,
            score=score,
            details={
                "description": case.description,
                "mode": self.mode,
                "judge_mode": self.judge_mode,
                "failures": failures,
                "judge": judge_details,
                "trajectory": trajectory,
            },
        )

    async def _run_trajectory(
        self,
        case: ExerciseTrajectoryCase,
    ) -> list[dict[str, Any]]:
        from agent.audit.crisis_log import InMemoryCrisisLogBackend
        from agent.graph import build_initial_state
        from agent.memory.modes import MemoryMode
        from agent.memory.store import OpenCouchMemoryStore
        from agent.models import AgentInput, Message
        from agent.runtime_context import WorkflowContext
        from agent.therapeutic.graph import build_therapeutic_subgraph

        subgraph = build_therapeutic_subgraph()
        memory_store = OpenCouchMemoryStore()
        history: list[dict[str, str]] = []
        carried_state = dict(case.initial_state)
        observed: list[dict[str, Any]] = []

        for index, turn in enumerate(case.turns):
            llm_client, response_llm = self._turn_llms(turn)
            state = dict(
                build_initial_state(
                    AgentInput(
                        message=turn.message,
                        history=[Message.model_validate(item) for item in history],
                        session_id=case.id,
                    ),
                    include_input_history=True,
                )
            )
            deep_update(state, carried_state)
            exercise_state_before = dict(state.get("exercise_state") or {})
            output = dict(
                await subgraph.ainvoke(
                    state,
                    context=WorkflowContext(
                        llm_client=llm_client,
                        response_llm=response_llm,
                        memory_store=memory_store,
                        crisis_log_backend=InMemoryCrisisLogBackend(),
                        memory_mode=MemoryMode.INCOGNITO,
                    ),
                )
            )

            _carry_forward(carried_state, output)
            turn_case = TherapeuticEvalCase(
                id=turn.case.id,
                message=turn.message,
                expected=turn.expected,
            )
            turn_failures = [
                f"turn {index + 1}: {failure}"
                for failure in grade_therapeutic_output(turn_case, output)
            ]
            observed.append(
                {
                    "turn_index": index + 1,
                    "user_message": turn.message,
                    "assistant_response": str(output.get("response_text", "")),
                    "response_style": output.get("response_style"),
                    "therapeutic_approach": output.get("therapeutic_approach"),
                    "routing_decision": last_routing_decision(output),
                    "exercise_state_before": exercise_state_before,
                    "exercise_state_after": dict(output.get("exercise_state") or {}),
                    "hard_failures": turn_failures,
                }
            )

            history.append({"role": "user", "content": turn.message})
            history.append(
                {
                    "role": "assistant",
                    "content": str(output.get("response_text", "")) or " ",
                    "response_style": str(output.get("response_style") or ""),
                }
            )

        return observed

    def _turn_llms(
        self,
        turn: ExerciseTrajectoryTurnCase,
    ) -> tuple[Any, Any]:
        if self.mode == "scripted":
            if turn.case.scripted is None:
                raise ValueError(
                    f"Turn {turn.case.id!r} needs scripted LLM outputs in scripted mode."
                )
            llm = ScriptedTherapeuticLLM(turn.case.scripted)
            return llm, llm
        return self._get_live_llms()

    def _get_live_llms(self) -> tuple[Any, Any]:
        if self._live_llms is None:
            self._live_llms = build_live_therapeutic_llms()
        return self._live_llms


def _grade_trajectory(
    case: ExerciseTrajectoryCase,
    trajectory: list[dict[str, Any]],
) -> list[str]:
    failures = [
        failure for turn in trajectory for failure in turn.get("hard_failures", [])
    ]
    expected = case.expected
    if expected.get("must_not_offer_menu", True):
        for turn in trajectory:
            text = str(turn.get("assistant_response", "")).casefold()
            found = [phrase for phrase in _MENU_PHRASES if phrase in text]
            if found:
                failures.append(
                    f"turn {turn['turn_index']}: response looks like a menu {found}"
                )

    expected_final_state = expected.get("final_exercise_state")
    if isinstance(expected_final_state, Mapping):
        actual_final_state = trajectory[-1].get("exercise_state_after", {})
        for key, expected_value in expected_final_state.items():
            actual_value = actual_final_state.get(key)
            if actual_value != expected_value:
                failures.append(
                    "final_exercise_state."
                    f"{key}: expected {expected_value!r}, got {actual_value!r}"
                )
    return failures


def _build_judge_artifact(
    *,
    case: ExerciseTrajectoryCase,
    trajectory: list[dict[str, Any]],
    hard_failures: list[str],
) -> ExerciseTrajectoryArtifact:
    return ExerciseTrajectoryArtifact(
        case_id=case.id,
        description=case.description,
        expected_exercise=_exercise_summary(case.expected.get("exercise_type")),
        expected_trajectory={
            key: value
            for key, value in case.expected.items()
            if key not in {"min_judge_score"}
        },
        turns=[
            ExerciseTrajectoryTurnArtifact(
                turn_index=int(turn["turn_index"]),
                user_message=str(turn["user_message"]),
                assistant_response=str(turn["assistant_response"]),
                response_style=(
                    str(turn["response_style"])
                    if turn.get("response_style") is not None
                    else None
                ),
                therapeutic_approach=(
                    str(turn["therapeutic_approach"])
                    if turn.get("therapeutic_approach") is not None
                    else None
                ),
                routing_decision=(
                    str(turn["routing_decision"])
                    if turn.get("routing_decision") is not None
                    else None
                ),
                exercise_state_before=dict(turn.get("exercise_state_before") or {}),
                exercise_state_after=dict(turn.get("exercise_state_after") or {}),
                hard_failures=list(turn.get("hard_failures") or []),
            )
            for turn in trajectory
        ],
        hard_failures=hard_failures,
    )


def _exercise_summary(exercise_type: Any) -> dict[str, Any]:
    if not exercise_type:
        return {}

    from agent.therapeutic.exercises.registry import get_exercise_definition

    definition = get_exercise_definition(str(exercise_type))
    if definition is None:
        return {"id": str(exercise_type), "found": False}
    return {
        "id": definition.id,
        "display_name": definition.display_name,
        "category": definition.category,
        "selection_use_case": definition.selection_use_case,
        "steps": [
            {
                "index": index,
                "id": step.id,
                "instruction": step.instruction,
                "completion_mode": step.completion_mode,
                "completion_criteria": step.completion_criteria,
            }
            for index, step in enumerate(definition.steps)
        ],
    }


def _carry_forward(carried_state: dict[str, Any], output: Mapping[str, Any]) -> None:
    for key in ("exercise_state", "therapeutic_approach"):
        if key in output:
            value = output[key]
            if isinstance(value, Mapping):
                carried_state[key] = {**(carried_state.get(key) or {}), **value}
            else:
                carried_state[key] = value


def _optional_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key!r} must be an object when provided.")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate guided-exercise trajectories.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="scripted avoids provider calls; live uses configured LLM clients.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("off", "live"),
        default="off",
        help="off runs only hard checks; live applies the LLM trajectory judge.",
    )
    return parser


def main() -> int:
    """Run the guided-exercise trajectory evaluator CLI.

    Returns:
        int: Shell exit code.
    """

    return run_evaluator_cli(
        lambda args: TherapeuticExerciseTrajectoryEvaluator(
            dataset_path=args.dataset,
            mode=args.mode,
            judge_mode=args.judge_mode,
        ),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
