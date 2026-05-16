"""Evaluate crisis-gate topology and branch isolation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch

from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)
from eval.runners.crisis_common import (
    CrisisEvalCase,
    ScriptedCrisisLLM,
    build_graph_state,
    optional_mapping,
    parse_crisis_case,
    routing_decision,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "eval" / "datasets" / "crisis" / "topology_v1.json"


class CrisisTopologyEvaluator(BaseEvaluator[CrisisEvalCase]):
    """Run CI-safe topology checks for the crisis gate."""

    def __init__(self, *, dataset_path: str | Path) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name="crisis_topology",
        )

    def parse_case(self, raw_case: Any) -> CrisisEvalCase:
        """Parse one crisis topology case.

        Args:
            raw_case (Any): Raw JSON case object.

        Returns:
            CrisisEvalCase: Parsed case.
        """

        return parse_crisis_case(raw_case)

    def case_id(self, case: CrisisEvalCase, index: int) -> str:
        """Return the stable case identifier."""

        return case.id

    async def run_case(self, case: CrisisEvalCase) -> EvalResult:
        """Run and grade one topology case."""

        expected_error = case.expected.get("error_contains")
        try:
            output, llm, tool_calls, crisis_log_count = await _invoke_parent_graph(case)
        except Exception as exc:  # noqa: BLE001 - expected errors are eval results
            if expected_error and str(expected_error).casefold() in str(exc).casefold():
                return EvalResult(
                    case_id=case.id,
                    passed=True,
                    score=1.0,
                    details={
                        "description": case.description,
                        "expected_error": str(expected_error),
                        "actual_error": f"{type(exc).__name__}: {exc}",
                    },
                )
            raise

        failures = _grade_case(
            case,
            output=output,
            llm=llm,
            tool_calls=tool_calls,
            crisis_log_count=crisis_log_count,
        )
        if expected_error:
            failures.append(f"expected error containing {expected_error!r}")
        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=1.0 if not failures else 0.0,
            details={
                "description": case.description,
                "failures": failures,
                "output": _summarize_output(output),
                "tool_calls": tool_calls,
                "structured_calls": llm.structured_calls if llm else {},
                "text_stream_calls": llm.text_stream_calls if llm else 0,
                "crisis_log_count": crisis_log_count,
            },
        )


async def _invoke_parent_graph(
    case: CrisisEvalCase,
) -> tuple[dict[str, Any], ScriptedCrisisLLM | None, dict[str, list[Any]], int]:
    from agent.audit.crisis_log import InMemoryCrisisLogBackend
    from agent.graph import build_agent_workflow
    from agent.memory.modes import MemoryMode
    from agent.memory.store import OpenCouchMemoryStore
    from agent.runtime_context import WorkflowContext

    state = build_graph_state(case)
    llm = None if case.scripted.get("no_llm") else ScriptedCrisisLLM(case)
    tool_calls: dict[str, list[Any]] = {
        "factual_lookup": [],
        "crisis_resources": [],
    }

    async def fake_factual_lookup(
        state: dict[str, Any],
        *,
        llm_client: Any,
        query: str,
    ) -> tuple[str, str]:
        tool_calls["factual_lookup"].append(
            {"message": state.get("message"), "query": query}
        )
        return (
            "verified factual answer\n\nSources:\n- Official source",
            "answered",
        )

    async def fake_crisis_resources(
        state: dict[str, Any],
        *,
        llm_client: Any,
    ) -> tuple[str, list[dict[str, str]], str]:
        scripted = optional_mapping(case.scripted, "crisis_resources")
        location = str(scripted.get("location", ""))
        status = str(scripted.get("status", "no_location"))
        resources = _resource_rows(scripted.get("resources", []))
        tool_calls["crisis_resources"].append(
            {
                "message": state.get("message"),
                "location": location,
                "status": status,
                "resources": resources,
            }
        )
        return location, resources, status

    crisis_log = InMemoryCrisisLogBackend()
    workflow = build_agent_workflow()
    with (
        patch("agent.turn_branches.answer_factual_lookup", new=fake_factual_lookup),
        patch(
            "agent.nodes.crisis_resource_lookup.find_crisis_resources",
            new=fake_crisis_resources,
        ),
    ):
        output = await workflow.ainvoke(
            state,
            context=WorkflowContext(
                llm_client=llm,
                response_llm=llm,
                memory_store=OpenCouchMemoryStore(),
                crisis_log_backend=crisis_log,
                memory_mode=MemoryMode.LOCAL,
            ),
        )
    return dict(output), llm, tool_calls, await crisis_log.arecord_count()


def _grade_case(
    case: CrisisEvalCase,
    *,
    output: dict[str, Any],
    llm: ScriptedCrisisLLM | None,
    tool_calls: dict[str, list[Any]],
    crisis_log_count: int,
) -> list[str]:
    expected = case.expected
    failures: list[str] = []
    crisis = output.get("crisis")

    _expect_equal(failures, "response_style", output.get("response_style"), expected)
    _expect_equal(failures, "crisis_level", getattr(crisis, "level", None), expected)
    _expect_equal(
        failures,
        "crisis_needs_response",
        getattr(crisis, "needs_crisis_response", None),
        expected,
    )
    _expect_equal(
        failures,
        "crisis_needs_clarification",
        getattr(crisis, "needs_clarification", None),
        expected,
    )
    _expect_equal(
        failures,
        "crisis_resource_tool_calls",
        len(tool_calls["crisis_resources"]),
        expected,
    )
    _expect_equal(
        failures,
        "factual_tool_calls",
        len(tool_calls["factual_lookup"]),
        expected,
    )
    _expect_equal(failures, "crisis_log_count", crisis_log_count, expected)

    structured_calls = llm.structured_calls if llm else {}
    _expect_equal(
        failures,
        "crisis_classifier_calls",
        structured_calls.get("CrisisAssessmentSchema", 0),
        expected,
    )
    _expect_equal(
        failures,
        "turn_dispatch_calls",
        structured_calls.get("TurnDispatchDecision", 0),
        expected,
    )
    _expect_equal(
        failures,
        "therapeutic_dispatch_calls",
        structured_calls.get("DispatchDecision", 0),
        expected,
    )

    if "safety_decision" in expected:
        _expect_equal(
            failures,
            "safety_decision",
            routing_decision(output, stage="safety"),
            expected,
        )
    if "turn_dispatch_decision" in expected:
        _expect_equal(
            failures,
            "turn_dispatch_decision",
            routing_decision(output, stage="turn_dispatch"),
            expected,
        )
    if "resource_lookup_status" in expected:
        actual_status = (
            tool_calls["crisis_resources"][0]["status"]
            if tool_calls["crisis_resources"]
            else None
        )
        _expect_equal(failures, "resource_lookup_status", actual_status, expected)

    response_text = str(output.get("response_text", ""))
    for needle in expected.get("response_text_contains", []):
        if str(needle).casefold() not in response_text.casefold():
            failures.append(f"response_text missing {needle!r}")

    return failures


def _expect_equal(
    failures: list[str],
    name: str,
    actual: Any,
    expected: Mapping[str, Any],
) -> None:
    if name not in expected:
        return
    expected_value = expected[name]
    if actual != expected_value:
        failures.append(f"{name}: expected {expected_value!r}, got {actual!r}")


def _summarize_output(output: Mapping[str, Any]) -> dict[str, Any]:
    crisis = output.get("crisis")
    return {
        "response_style": output.get("response_style"),
        "response_text": output.get("response_text"),
        "crisis": (
            crisis.model_dump(mode="json") if hasattr(crisis, "model_dump") else crisis
        ),
        "routing_trace": (output.get("diagnostics") or {}).get("routing_trace"),
    }


def _resource_rows(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError("resources must be a list.")
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("resource entries must be objects.")
        rows.append({str(key): str(val) for key, val in item.items()})
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate crisis-gate topology.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    return parser


def main() -> int:
    """Run the crisis topology evaluator CLI."""

    return run_evaluator_cli(
        lambda args: CrisisTopologyEvaluator(dataset_path=args.dataset),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
