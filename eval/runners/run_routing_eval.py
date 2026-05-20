"""Run deterministic routing and behavior evals through OpenAITextRuntime.

This runner verifies that representative inputs route to the expected
specialist, route, and runtime mode without requiring a live provider.

It supports both single-turn cases and multiturn sequences:

    apps/backend/.venv/bin/python eval/runners/run_routing_eval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.audit.crisis_log import InMemoryCrisisLogBackend  # noqa: E402
from agent.memory.modes import MemoryMode  # noqa: E402
from agent.memory.store import OpenCouchMemoryStore  # noqa: E402
from agent.models import AgentInput  # noqa: E402
from agent.runtime import OpenAITextRuntime, build_initial_state  # noqa: E402
from agent.runtime_context import WorkflowContext  # noqa: E402
from tests.support.openai_text import (  # noqa: E402
    FakeOpenAISDKRunner,
    ScriptedOpenAITextRouteLLM,
)
from tests.support.persistence import FakeCrossRestartLLM  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "eval" / "datasets" / "routing_matrix.jsonl"


@dataclass(slots=True)
class EvalTurn:
    """One turn inside a routing eval case."""

    message: str
    prior_state: dict[str, Any] | None
    llm: dict[str, Any] | None
    runner: dict[str, Any]
    expected: dict[str, Any]
    memory_seed: list[dict[str, Any]] | None


@dataclass(slots=True)
class EvalCase:
    """One routing eval case loaded from JSONL."""

    id: str
    turns: list[EvalTurn]


@dataclass(slots=True)
class EvalResult:
    """Serializable result for one routing eval case."""

    id: str
    passed: bool
    checks: list[str]
    failures: list[str]
    output: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to routing JSONL dataset.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=None,
        help="Run only the given case id. Can be provided multiple times.",
    )
    return parser.parse_args()


def _load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if isinstance(raw.get("turns"), list):
                turns = [_load_turn(turn) for turn in raw["turns"]]
            else:
                turns = [_load_turn(raw)]
            cases.append(
                EvalCase(
                    id=str(raw["id"]),
                    turns=turns,
                )
            )
    return cases


def _load_turn(raw: dict[str, Any]) -> EvalTurn:
    return EvalTurn(
        message=str(raw["message"]),
        prior_state=(
            dict(raw["prior_state"])
            if isinstance(raw.get("prior_state"), dict)
            else None
        ),
        llm=dict(raw["llm"]) if isinstance(raw.get("llm"), dict) else None,
        runner=dict(raw["runner"]),
        expected=dict(raw["expected"]),
        memory_seed=(
            [dict(record) for record in raw["memory_seed"]]
            if isinstance(raw.get("memory_seed"), list)
            else None
        ),
    )


def _select_cases(
    cases: list[EvalCase],
    *,
    case_ids: list[str] | None,
) -> list[EvalCase]:
    if not case_ids:
        return cases
    allowed = set(case_ids)
    return [case for case in cases if case.id in allowed]


def _initial_state(case_id: str, turn_index: int, turn: EvalTurn) -> dict[str, Any]:
    return dict(
        build_initial_state(
            AgentInput(
                message=turn.message,
                user_id="eval-user-1",
                session_id=f"eval-session-{case_id}-turn-{turn_index}",
            )
        )
    )


def _llm(turn: EvalTurn) -> Any:
    if turn.llm is None:
        return FakeCrossRestartLLM()
    return ScriptedOpenAITextRouteLLM(**turn.llm)


def _context(turn: EvalTurn) -> WorkflowContext:
    return WorkflowContext(
        llm_client=_llm(turn),
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
    )


async def _seed_memory_store(context: WorkflowContext, turn: EvalTurn) -> None:
    for record in turn.memory_seed or []:
        namespace = tuple(record["namespace"])
        await context.memory_store.aput(
            namespace,
            str(record["key"]),
            dict(record["value"]),
        )


def _runner(turn: EvalTurn) -> FakeOpenAISDKRunner:
    tool_calls = [
        (str(tool_name), dict(arguments))
        for tool_name, arguments in turn.runner.get("tool_calls", [])
    ]
    return FakeOpenAISDKRunner(
        final_output=str(turn.runner.get("final_output", "openai reply")),
        tool_calls=tool_calls,
        tool_response_as_final=bool(turn.runner.get("tool_response_as_final", False)),
    )


async def _run_case(case: EvalCase) -> EvalResult:
    runtime = OpenAITextRuntime(
        runner=FakeOpenAISDKRunner(),
        model="gpt-test",
    )
    failures: list[str] = []
    checks: list[str] = []
    outputs: list[dict[str, Any]] = []

    prior_state: dict[str, Any] | None = None
    for index, turn in enumerate(case.turns, start=1):
        context = _context(turn)
        await _seed_memory_store(context, turn)
        runner = _runner(turn)
        runtime._runner = runner  # noqa: SLF001 - eval-only runner injection

        effective_prior_state = (
            turn.prior_state if turn.prior_state is not None else prior_state
        )
        try:
            result = await runtime.run_turn(
                _initial_state(case.id, index, turn),
                config={
                    "configurable": {"thread_id": f"eval-thread-{case.id}-{index}"}
                },
                context=context,
                prior_state=effective_prior_state,
            )
        except Exception as exc:
            output = {"turn": index, "exception": repr(exc)}
            failures.append(f"turn {index}: raised exception {exc!r}")
            outputs.append(output)
            return EvalResult(
                id=case.id,
                passed=False,
                checks=checks,
                failures=failures,
                output={"turns": outputs},
            )

        output = _turn_output(result, runner, index)
        outputs.append(output)
        _score_expected(
            turn.expected,
            result=result,
            output=output,
            checks=checks,
            failures=failures,
            label_prefix=f"turn {index}",
        )
        prior_state = result

    return EvalResult(
        id=case.id,
        passed=not failures,
        checks=checks,
        failures=failures,
        output={"turns": outputs},
    )


def _turn_output(
    result: dict[str, Any], runner: FakeOpenAISDKRunner, turn_index: int
) -> dict[str, Any]:
    diagnostics = dict(result.get("diagnostics", {}) or {})
    return {
        "turn": turn_index,
        "selected_agent": diagnostics.get("openai_selected_agent"),
        "route": result.get("route"),
        "runtime_mode": diagnostics.get("openai_text_runtime_mode"),
        "response_style": result.get("response_style"),
        "response_text": result.get("response_text", ""),
        "diagnostics": diagnostics,
        "run_call_count": len(runner.run_calls),
        "stream_call_count": len(runner.stream_calls),
    }


def _score_expected(
    expected: dict[str, Any],
    *,
    result: dict[str, Any],
    output: dict[str, Any],
    checks: list[str],
    failures: list[str],
    label_prefix: str,
) -> None:
    for label in ("selected_agent", "route", "runtime_mode"):
        if label not in expected:
            continue
        _check_equal(
            f"{label_prefix} {label}",
            actual=output.get(label),
            expected=expected[label],
            checks=checks,
            failures=failures,
        )

    expected_state = expected.get("state")
    if isinstance(expected_state, dict):
        for path, expected_value in expected_state.items():
            _check_equal(
                f"{label_prefix} state.{path}",
                actual=_dotted_get(result, str(path)),
                expected=expected_value,
                checks=checks,
                failures=failures,
            )

    expected_diagnostics = expected.get("diagnostics")
    if isinstance(expected_diagnostics, dict):
        for path, expected_value in expected_diagnostics.items():
            _check_equal(
                f"{label_prefix} diagnostics.{path}",
                actual=_dotted_get(output.get("diagnostics", {}), str(path)),
                expected=expected_value,
                checks=checks,
                failures=failures,
            )

    response_text = str(output.get("response_text", ""))
    for needle in expected.get("must_include", []):
        if str(needle) in response_text:
            checks.append(f"{label_prefix} included {needle!r}")
        else:
            failures.append(f"{label_prefix} missing required text {needle!r}")

    for needle in expected.get("must_not_include", []):
        if str(needle) in response_text:
            failures.append(f"{label_prefix} contained forbidden text {needle!r}")
        else:
            checks.append(f"{label_prefix} did not include forbidden text {needle!r}")


def _dotted_get(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
            continue
        return None
    return current


def _check_equal(
    label: str,
    *,
    actual: Any,
    expected: Any,
    checks: list[str],
    failures: list[str],
) -> None:
    if actual == expected:
        checks.append(f"{label} matched {expected!r}")
    else:
        failures.append(f"{label} expected {expected!r}, got {actual!r}")


async def _amain() -> int:
    args = _parse_args()
    cases = _select_cases(_load_cases(args.dataset), case_ids=args.case_id)
    if not cases:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error": "No eval cases selected.",
                    "dataset": str(args.dataset),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    results = [await _run_case(case) for case in cases]
    passed = sum(1 for result in results if result.passed)
    summary = {
        "passed": passed == len(results),
        "passed_count": passed,
        "failed_count": len(results) - passed,
        "total_count": len(results),
        "results": [
            {
                "id": result.id,
                "passed": result.passed,
                "checks": result.checks,
                "failures": result.failures,
                "output": result.output,
            }
            for result in results
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
