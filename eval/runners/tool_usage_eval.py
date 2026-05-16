"""Evaluate parent-graph dispatch and grounded-search tool usage."""

from __future__ import annotations

import argparse
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)
from eval.runners.therapeutic_common import deep_update

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "tool_usage" / "turn_dispatch_v1.json"
)


@dataclass(frozen=True)
class ToolUsageCase:
    """Parsed tool-usage eval case."""

    id: str
    message: str
    description: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


class ScriptedGraphLLM:
    """Scripted LLM client for parent-graph dispatch evals."""

    def __init__(self, case: ToolUsageCase) -> None:
        self.case = case
        self.structured_calls: dict[str, int] = {}
        self.text_stream_calls = 0
        self.text_calls: list[dict[str, Any]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.text_calls.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction,
                "use_search": use_search,
            }
        )
        return str(self.case.scripted.get("text_response", "scripted text response"))

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        self.text_stream_calls += 1
        yield str(
            self.case.scripted.get(
                "response_text",
                "scripted streamed response",
            )
        )

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
    ) -> Any:
        schema_name = response_schema.__name__
        self.structured_calls[schema_name] = (
            self.structured_calls.get(schema_name, 0) + 1
        )

        if schema_name == "CrisisAssessmentSchema":
            crisis = _required_mapping(self.case.scripted, "crisis")
            return response_schema(**crisis)

        if schema_name == "TurnDispatchDecision":
            turn_dispatch = _required_mapping(self.case.scripted, "turn_dispatch")
            return response_schema(**turn_dispatch)

        if schema_name == "DispatchDecision":
            therapeutic_dispatch = _required_mapping(
                self.case.scripted,
                "therapeutic_dispatch",
            )
            return response_schema(**therapeutic_dispatch)

        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")


class ToolUsageEvaluator(BaseEvaluator[ToolUsageCase]):
    """Run parent-graph routing and tool-usage checks."""

    def __init__(self, *, dataset_path: str | Path) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name="tool_usage_turn_dispatch",
        )

    def parse_case(self, raw_case: Any) -> ToolUsageCase:
        """Parse one raw tool-usage case.

        Args:
            raw_case (Any): Raw JSON case object.

        Returns:
            ToolUsageCase: Parsed case.
        """

        if not isinstance(raw_case, Mapping):
            raise TypeError("Tool usage eval cases must be JSON objects.")
        return ToolUsageCase(
            id=str(raw_case["id"]),
            description=str(raw_case.get("description", "")),
            message=str(raw_case["message"]),
            history=_list_of_mappings(raw_case.get("history", []), "history"),
            state=dict(_optional_mapping(raw_case, "state")),
            scripted=dict(_optional_mapping(raw_case, "scripted")),
            expected=dict(_optional_mapping(raw_case, "expected")),
        )

    def case_id(self, case: ToolUsageCase, index: int) -> str:
        """Return the stable dataset id for one case.

        Args:
            case (ToolUsageCase): Parsed case.
            index (int): Zero-based case index.

        Returns:
            str: Stable case id.
        """

        return case.id

    async def run_case(self, case: ToolUsageCase) -> EvalResult:
        """Run and grade one tool-usage eval case.

        Args:
            case (ToolUsageCase): Parsed case.

        Returns:
            EvalResult: Case result.
        """

        output, llm, tool_calls = await _invoke_parent_graph(case)
        failures = _grade_case(case, output, llm, tool_calls)
        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=1.0 if not failures else 0.0,
            details={
                "description": case.description,
                "failures": failures,
                "output": _summarize_output(output),
                "tool_calls": tool_calls,
                "structured_calls": llm.structured_calls,
                "text_stream_calls": llm.text_stream_calls,
            },
        )


async def _invoke_parent_graph(
    case: ToolUsageCase,
) -> tuple[dict[str, Any], ScriptedGraphLLM, dict[str, list[dict[str, Any]]]]:
    from agent.audit.crisis_log import InMemoryCrisisLogBackend
    from agent.graph import build_agent_workflow, build_initial_state
    from agent.memory.modes import MemoryMode
    from agent.memory.store import OpenCouchMemoryStore
    from agent.models import AgentInput, Message
    from agent.runtime_context import WorkflowContext

    history = [Message.model_validate(item) for item in case.history]
    state = dict(
        build_initial_state(
            AgentInput(
                message=case.message,
                user_id="eval-user",
                session_id=case.id,
                history=history,
            ),
            include_input_history=True,
        )
    )
    deep_update(state, case.state)

    llm = ScriptedGraphLLM(case)
    tool_calls: dict[str, list[dict[str, Any]]] = {
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
        tool_calls["crisis_resources"].append({"message": state.get("message")})
        return (
            "Singapore",
            [
                {
                    "name": "Samaritans of Singapore",
                    "phone": "1767",
                    "website": "https://www.sos.org.sg",
                    "region": "Singapore",
                }
            ],
            "found",
        )

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
                crisis_log_backend=InMemoryCrisisLogBackend(),
                memory_mode=MemoryMode.LOCAL,
            ),
        )
    return dict(output), llm, tool_calls


def _grade_case(
    case: ToolUsageCase,
    output: dict[str, Any],
    llm: ScriptedGraphLLM,
    tool_calls: dict[str, list[dict[str, Any]]],
) -> list[str]:
    expected = case.expected
    failures: list[str] = []

    _expect_equal(
        failures,
        "response_style",
        output.get("response_style"),
        expected.get("response_style"),
    )

    response_text = str(output.get("response_text", ""))
    for needle in expected.get("response_text_contains", []):
        if str(needle).casefold() not in response_text.casefold():
            failures.append(f"response_text missing {needle!r}")

    factual_calls = tool_calls["factual_lookup"]
    crisis_calls = tool_calls["crisis_resources"]
    _expect_equal(
        failures,
        "factual_tool_calls",
        len(factual_calls),
        expected.get("factual_tool_calls"),
    )
    _expect_equal(
        failures,
        "crisis_resource_tool_calls",
        len(crisis_calls),
        expected.get("crisis_resource_tool_calls"),
    )
    if "factual_tool_query" in expected:
        actual_query = factual_calls[0]["query"] if factual_calls else None
        _expect_equal(
            failures,
            "factual_tool_query",
            actual_query,
            expected.get("factual_tool_query"),
        )

    _expect_equal(
        failures,
        "turn_dispatch_calls",
        llm.structured_calls.get("TurnDispatchDecision", 0),
        expected.get("turn_dispatch_calls"),
    )
    _expect_equal(
        failures,
        "therapeutic_dispatch_calls",
        llm.structured_calls.get("DispatchDecision", 0),
        expected.get("therapeutic_dispatch_calls"),
    )

    if "turn_dispatch_decision" in expected:
        actual_decision = _routing_decision(output, stage="turn_dispatch")
        _expect_equal(
            failures,
            "turn_dispatch_decision",
            actual_decision,
            expected.get("turn_dispatch_decision"),
        )

    return failures


def _routing_decision(output: dict[str, Any], *, stage: str) -> str | None:
    diagnostics = output.get("diagnostics") or {}
    if not isinstance(diagnostics, Mapping):
        return None
    trace = diagnostics.get("routing_trace") or []
    if not isinstance(trace, list):
        return None
    for item in reversed(trace):
        if isinstance(item, Mapping) and item.get("stage") == stage:
            decision = item.get("decision")
            return str(decision) if decision is not None else None
    return None


def _summarize_output(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "response_style": output.get("response_style"),
        "response_type": str(output.get("response_type", "")),
        "response_text": output.get("response_text"),
        "crisis": (
            output.get("crisis").model_dump(mode="json")
            if hasattr(output.get("crisis"), "model_dump")
            else output.get("crisis")
        ),
        "routing_trace": (output.get("diagnostics") or {}).get("routing_trace"),
    }


def _expect_equal(
    failures: list[str],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    if expected is None:
        return
    if actual != expected:
        failures.append(f"{name}: expected {expected!r}, got {actual!r}")


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object.")
    return value


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Case is missing scripted {key!r} object.")
    return value


def _list_of_mappings(value: Any, field_name: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list.")
    items: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name} entries must be objects.")
        items.append({str(key): str(val) for key, val in item.items()})
    return items


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate parent-graph tool usage.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    return parser


def main() -> int:
    """Run the tool-usage evaluator CLI.

    Returns:
        int: Shell exit code.
    """

    return run_evaluator_cli(
        lambda args: ToolUsageEvaluator(dataset_path=args.dataset),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
