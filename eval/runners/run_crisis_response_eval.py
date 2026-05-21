"""Run end-to-end crisis response evals through OpenAITextRuntime.

This runner exercises the full runtime crisis path, including:
- safety-gate crisis routing
- crisis specialist agent selection
- crisis-resource lookup tool execution
- deterministic crisis template loading
- crisis audit log side effects

Default mode is deterministic and does not call a live provider:

    apps/backend/.venv/bin/python eval/runners/run_crisis_response_eval.py
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

DEFAULT_DATASET = REPO_ROOT / "eval" / "datasets" / "crisis_response_events.jsonl"


@dataclass(slots=True)
class EvalCase:
    """One end-to-end crisis response eval case loaded from JSONL."""

    id: str
    message: str
    risk_level: int
    template_risk_level: str
    crisis_location_status: str
    crisis_location: str
    crisis_resource_status: str
    simulate_search_error: bool
    expected: dict[str, Any]


@dataclass(slots=True)
class EvalResult:
    """Serializable result for one end-to-end crisis response eval case."""

    id: str
    passed: bool
    checks: list[str]
    failures: list[str]
    output: dict[str, Any]


class EvalRouteLLM(ScriptedOpenAITextRouteLLM):
    """Scripted route LLM with optional crisis-resource lookup failure injection."""

    def __init__(self, *, simulate_search_error: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.simulate_search_error = simulate_search_error

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        if (
            response_schema.__name__ == "CrisisResourceLookupResult"
            and self.simulate_search_error
        ):
            raise RuntimeError("search unavailable")
        return await super().generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to crisis response JSONL dataset.",
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
            cases.append(
                EvalCase(
                    id=str(raw["id"]),
                    message=str(raw["message"]),
                    risk_level=int(raw["risk_level"]),
                    template_risk_level=str(raw["template_risk_level"]),
                    crisis_location_status=str(raw["crisis_location_status"]),
                    crisis_location=str(raw.get("crisis_location", "")),
                    crisis_resource_status=str(raw["crisis_resource_status"]),
                    simulate_search_error=bool(raw.get("simulate_search_error", False)),
                    expected=dict(raw.get("expected", {})),
                )
            )
    return cases


def _select_cases(
    cases: list[EvalCase],
    *,
    case_ids: list[str] | None,
) -> list[EvalCase]:
    if not case_ids:
        return cases
    allowed = set(case_ids)
    return [case for case in cases if case.id in allowed]


def _initial_state(case: EvalCase) -> dict[str, Any]:
    return dict(
        build_initial_state(
            AgentInput(
                message=case.message,
                user_id="eval-user-1",
                session_id=f"eval-session-{case.id}",
            )
        )
    )


def _context(case: EvalCase) -> WorkflowContext:
    llm = EvalRouteLLM(
        route="therapeutic",
        crisis_level=case.risk_level,
        crisis_location_status=case.crisis_location_status,
        crisis_location=case.crisis_location,
        crisis_resource_status=case.crisis_resource_status,
        simulate_search_error=case.simulate_search_error,
    )
    return WorkflowContext(
        llm_client=llm,
        memory_store=OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.LOCAL,
    )


def _runner(case: EvalCase) -> FakeOpenAISDKRunner:
    return FakeOpenAISDKRunner(
        tool_calls=[
            ("lookup_crisis_resources", {}),
            (
                "get_crisis_support_template",
                {"risk_level": case.template_risk_level},
            ),
        ],
        tool_response_as_final=True,
    )


async def _run_case(case: EvalCase) -> EvalResult:
    context = _context(case)
    runner = _runner(case)
    runtime = OpenAITextRuntime(
        runner=runner,
        model="gpt-test",
    )
    failures: list[str] = []
    checks: list[str] = []

    try:
        result = await runtime.run_turn(
            _initial_state(case),
            config={"configurable": {"thread_id": f"eval-thread-{case.id}"}},
            context=context,
        )
    except Exception as exc:
        output = {"exception": repr(exc)}
        failures.append(f"raised exception: {exc!r}")
        return EvalResult(
            id=case.id,
            passed=False,
            checks=checks,
            failures=failures,
            output=output,
        )

    output = {
        "route": result.get("route"),
        "response_style": result.get("response_style"),
        "response_text": result.get("response_text", ""),
        "crisis_level": getattr(result.get("crisis"), "level", None),
        "resource_lookup_status": result.get("resource_lookup_status"),
        "inferred_location": result.get("inferred_location", ""),
        "found_resources": result.get("found_resources", []),
        "selected_agent": result.get("diagnostics", {}).get("openai_selected_agent"),
        "runtime_mode": result.get("diagnostics", {}).get("openai_text_runtime_mode"),
        "tool_calls": result.get("diagnostics", {}).get("openai_crisis_tool_calls", []),
        "tool_fallback": result.get("diagnostics", {}).get(
            "openai_crisis_tool_fallback"
        ),
        "crisis_log_count": await context.crisis_log_backend.arecord_count(),
        "run_call_count": len(runner.run_calls),
    }

    _score_expected(case.expected, output, checks, failures)
    return EvalResult(
        id=case.id,
        passed=not failures,
        checks=checks,
        failures=failures,
        output=output,
    )


def _score_expected(
    expected: dict[str, Any],
    output: dict[str, Any],
    checks: list[str],
    failures: list[str],
) -> None:
    for label in (
        "route",
        "response_style",
        "selected_agent",
        "runtime_mode",
        "crisis_level",
        "resource_lookup_status",
        "inferred_location",
        "crisis_log_count",
        "run_call_count",
        "tool_fallback",
    ):
        expected_key = f"expected_{label}"
        if label in expected:
            _check_equal(
                label,
                actual=output[label],
                expected=expected[label],
                checks=checks,
                failures=failures,
            )
        elif expected_key in expected:
            _check_equal(
                label,
                actual=output[label],
                expected=expected[expected_key],
                checks=checks,
                failures=failures,
            )

    response_text = str(output.get("response_text", ""))
    user_facing_text = response_text.split("\n\nAvoid:\n", maxsplit=1)[0]
    resources = list(output.get("found_resources", []))

    for needle in expected.get("must_include", []):
        if str(needle) in user_facing_text:
            checks.append(f"included {needle!r}")
        else:
            failures.append(f"missing required text {needle!r}")

    for needle in expected.get("must_not_include", []):
        if str(needle) in user_facing_text:
            failures.append(f"contained forbidden text {needle!r}")
        else:
            checks.append(f"did not include forbidden text {needle!r}")

    if expected.get("requires_resource"):
        if resources:
            checks.append("returned at least one resource")
        else:
            failures.append("expected at least one resource")

    if expected.get("requires_no_resources"):
        if not resources:
            checks.append("returned no resources")
        else:
            failures.append(f"expected no resources, got {resources!r}")

    if phone := expected.get("must_preserve_phone"):
        phones = [str(resource.get("phone", "")) for resource in resources]
        if str(phone) in phones:
            checks.append(f"preserved phone {phone!r}")
        else:
            failures.append(f"expected phone {phone!r} in resources, got {phones!r}")

    if "expected_tool_calls" in expected:
        _check_equal(
            "tool_calls",
            actual=output.get("tool_calls"),
            expected=expected["expected_tool_calls"],
            checks=checks,
            failures=failures,
        )


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
