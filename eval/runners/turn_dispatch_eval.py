"""Evaluate standalone turn-dispatch node contracts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    _REPO_ROOT / "eval" / "datasets" / "turn_dispatch" / "standalone_v1.json"
)


@dataclass(frozen=True)
class TurnDispatchEvalCase:
    """Parsed standalone turn-dispatch eval case."""

    id: str
    message: str
    description: str = ""
    modes: tuple[str, ...] = ("scripted", "live")
    history: list[dict[str, str]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


class _Runtime:
    """Small runtime-shaped object exposing ``.context`` for node calls."""

    def __init__(self, *, llm_client: Any | None) -> None:
        from agent.audit.crisis_log import InMemoryCrisisLogBackend
        from agent.memory.modes import MemoryMode
        from agent.memory.store import OpenCouchMemoryStore
        from agent.runtime_context import WorkflowContext

        self.context = WorkflowContext(
            llm_client=llm_client,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.LOCAL,
        )


class ScriptedTurnDispatchLLM:
    """LLM-shaped fake for standalone turn-dispatch evals."""

    def __init__(self, case: TurnDispatchEvalCase) -> None:
        self.case = case
        self.structured_calls: dict[str, int] = {}
        self.text_stream_calls = 0
        self.text_calls = 0

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.text_calls += 1
        return str(self.case.scripted.get("text_response", "scripted response"))

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        self.text_stream_calls += 1
        yield str(self.case.scripted.get("response_text", "scripted response"))

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        schema_name = response_schema.__name__
        self.structured_calls[schema_name] = (
            self.structured_calls.get(schema_name, 0) + 1
        )
        if self.case.scripted.get("raise_on_schema") == schema_name:
            raise RuntimeError(str(self.case.scripted.get("error", "scripted error")))
        if schema_name == "TurnDispatchDecision":
            return response_schema(**required_mapping(self.case.scripted, "decision"))
        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")


class _CountingLLM:
    """Delegate LLM calls while recording structured call counts."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.structured_calls: dict[str, int] = {}

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return await self.delegate.generate_text(
            prompt=prompt,
            system_instruction=system_instruction,
            use_search=use_search,
        )

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self.delegate.generate_text_stream(
            prompt=prompt,
            system_instruction=system_instruction,
        ):
            yield chunk

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> Any:
        schema_name = response_schema.__name__
        self.structured_calls[schema_name] = (
            self.structured_calls.get(schema_name, 0) + 1
        )
        return await self.delegate.generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )


class TurnDispatchEvaluator(BaseEvaluator[TurnDispatchEvalCase]):
    """Run standalone turn-dispatch contract and routing checks."""

    def __init__(self, *, dataset_path: str | Path, mode: str) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"turn_dispatch_{mode}",
        )
        self.mode = mode

    def load_cases(self) -> list[TurnDispatchEvalCase]:
        """Load cases applicable to the selected mode."""

        return [case for case in super().load_cases() if self.mode in case.modes]

    def parse_case(self, raw_case: Any) -> TurnDispatchEvalCase:
        """Parse one standalone turn-dispatch case."""

        return parse_case(raw_case)

    def case_id(self, case: TurnDispatchEvalCase, index: int) -> str:
        """Return the stable case identifier."""

        return case.id

    async def run_case(self, case: TurnDispatchEvalCase) -> EvalResult:
        """Run and grade one standalone turn-dispatch case."""

        expected_error = case.expected.get("error_contains")
        try:
            artifact = await invoke_case(case, mode=self.mode)
        except Exception as exc:  # noqa: BLE001 - expected failures are eval data
            if expected_error and str(expected_error).casefold() in str(exc).casefold():
                return EvalResult(
                    case_id=case.id,
                    passed=True,
                    score=1.0,
                    details={
                        "description": case.description,
                        "mode": self.mode,
                        "expected_error": str(expected_error),
                        "actual_error": f"{type(exc).__name__}: {exc}",
                    },
                )
            raise

        failures = grade_case(case, artifact)
        if expected_error:
            failures.append(f"expected error containing {expected_error!r}")

        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=1.0 if not failures else 0.0,
            details={
                "description": case.description,
                "mode": self.mode,
                "failures": failures,
                "artifact": artifact,
            },
        )


async def invoke_case(
    case: TurnDispatchEvalCase,
    *,
    mode: str,
) -> dict[str, Any]:
    """Invoke ``run_turn_dispatch_node`` for one eval case."""

    from agent.nodes.turn_dispatch import run_turn_dispatch_node

    llm = llm_for_case(case, mode=mode)
    command = await run_turn_dispatch_node(
        build_state(case),
        _Runtime(llm_client=llm),  # type: ignore[arg-type]
    )
    return {
        "goto": command.goto,
        "delta": jsonify(command.update),
        "structured_calls": getattr(llm, "structured_calls", {}),
    }


def llm_for_case(case: TurnDispatchEvalCase, *, mode: str) -> Any | None:
    """Return the LLM client for one eval case."""

    if case.scripted.get("no_llm"):
        return None
    if mode == "live":
        from config import create_configured_control_llm_client

        return _CountingLLM(create_configured_control_llm_client())
    return ScriptedTurnDispatchLLM(case)


def build_state(case: TurnDispatchEvalCase) -> dict[str, Any]:
    """Build graph state for a standalone turn-dispatch eval case."""

    from agent.graph import build_initial_state
    from agent.models import AgentInput, Message

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
    return state


def grade_case(
    case: TurnDispatchEvalCase,
    artifact: dict[str, Any],
) -> list[str]:
    """Return grading failures for one eval artifact."""

    expected = case.expected
    failures: list[str] = []

    expect_equal(failures, "goto", artifact.get("goto"), expected)
    expect_equal(
        failures,
        "structured_calls",
        artifact.get("structured_calls"),
        expected,
    )

    delta = artifact.get("delta")
    if not isinstance(delta, Mapping):
        failures.append("delta is not a mapping")
        return failures

    expect_equal(failures, "route", delta.get("route"), expected)
    expect_equal(
        failures,
        "grounded_lookup_status",
        nested_get(delta, "grounded_lookup", "status"),
        expected,
    )
    expect_equal(
        failures,
        "grounded_lookup_query",
        nested_get(delta, "grounded_lookup", "query"),
        expected,
    )
    required_query_phrases = expected.get("grounded_lookup_query_contains")
    if required_query_phrases is not None:
        query = str(nested_get(delta, "grounded_lookup", "query") or "")
        require_phrases(
            failures,
            "grounded_lookup.query",
            query,
            required_query_phrases,
        )

    memory_control = delta.get("memory_control")
    if isinstance(memory_control, Mapping):
        grade_memory_control(failures, memory_control, expected)

    diagnostics = delta.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        grade_diagnostics(failures, diagnostics, expected)

    return failures


def grade_memory_control(
    failures: list[str],
    memory_control: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Grade memory-control fields emitted by turn dispatch."""

    action = memory_control.get("action")
    if "memory_action" in expected:
        expected_action = expected["memory_action"]
        if action != expected_action:
            failures.append(
                f"memory_action: expected {expected_action!r}, got {action!r}"
            )

    if "memory_action_subset" in expected:
        if not isinstance(action, Mapping):
            failures.append("memory_action is not a mapping")
        else:
            expect_subset(
                failures,
                "memory_action",
                action,
                expected["memory_action_subset"],
            )

    if "memory_action_type" in expected:
        actual_type = action.get("type") if isinstance(action, Mapping) else None
        expect_equal(failures, "memory_action_type", actual_type, expected)

    for field_name in ("query", "preference_text"):
        expected_key = f"memory_action_{field_name}_contains"
        if expected_key not in expected:
            continue
        actual = action.get(field_name) if isinstance(action, Mapping) else ""
        require_phrases(
            failures,
            f"memory_action.{field_name}",
            str(actual or ""),
            expected[expected_key],
        )

    if "pending_action_is_null" in expected:
        actual = memory_control.get("pending_action", "__missing__")
        expected_null = bool(expected["pending_action_is_null"])
        if expected_null and actual is not None:
            failures.append(
                f"memory_control.pending_action: expected null, got {actual!r}"
            )
        if not expected_null and actual is None:
            failures.append("memory_control.pending_action unexpectedly null")


def grade_diagnostics(
    failures: list[str],
    diagnostics: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    """Grade routing diagnostics emitted by turn dispatch."""

    expect_equal(
        failures,
        "turn_dispatch_classifier_path",
        diagnostics.get("turn_dispatch_classifier_path"),
        expected,
    )
    expect_equal(
        failures,
        "turn_dispatch_llm_failure_occurred",
        diagnostics.get("turn_dispatch_llm_failure_occurred"),
        expected,
    )
    if "turn_dispatch_decision" in expected:
        decision = routing_trace_field(diagnostics, "decision")
        expect_equal(failures, "turn_dispatch_decision", decision, expected)
    if "turn_dispatch_source" in expected:
        source = routing_trace_field(diagnostics, "source")
        expect_equal(failures, "turn_dispatch_source", source, expected)
    if "active_flow" in expected:
        active_flow = nested_get(
            diagnostics,
            "turn_dispatch_active_flow",
            "active_flow",
        )
        expect_equal(failures, "active_flow", active_flow, expected)
        trace_flow = routing_trace_field(diagnostics, "active_flow")
        if trace_flow != expected["active_flow"]:
            failures.append(
                "active_flow_trace: "
                f"expected {expected['active_flow']!r}, got {trace_flow!r}"
            )
    if "active_flow_action" in expected:
        active_flow_action = nested_get(
            diagnostics,
            "turn_dispatch_active_flow",
            "action",
        )
        expect_equal(failures, "active_flow_action", active_flow_action, expected)
        trace_action = routing_trace_field(diagnostics, "active_flow_action")
        if trace_action != expected["active_flow_action"]:
            failures.append(
                "active_flow_action_trace: "
                f"expected {expected['active_flow_action']!r}, got {trace_action!r}"
            )


def routing_trace_field(
    diagnostics: Mapping[str, Any],
    field_name: str,
) -> str | None:
    """Return the latest turn-dispatch routing-trace field."""

    trace = diagnostics.get("routing_trace") or []
    if not isinstance(trace, list):
        return None
    for item in reversed(trace):
        if isinstance(item, Mapping) and item.get("stage") == "turn_dispatch":
            value = item.get(field_name)
            return str(value) if value is not None else None
    return None


def parse_case(raw_case: Any) -> TurnDispatchEvalCase:
    """Parse one raw JSON case."""

    if not isinstance(raw_case, Mapping):
        raise TypeError("Turn-dispatch eval cases must be JSON objects.")
    return TurnDispatchEvalCase(
        id=str(raw_case["id"]),
        description=str(raw_case.get("description", "")),
        modes=tuple(str(item) for item in raw_case.get("modes", ["scripted", "live"])),
        message=str(raw_case["message"]),
        history=list_of_mappings(raw_case.get("history", []), "history"),
        state=dict(optional_mapping(raw_case, "state")),
        scripted=dict(optional_mapping(raw_case, "scripted")),
        expected=dict(optional_mapping(raw_case, "expected")),
    )


def jsonify(value: Any) -> Any:
    """Convert Pydantic/dict/list values into JSON-compatible data."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonify(item) for item in value]
    return value


def nested_get(mapping: Mapping[str, Any], key: str, nested_key: str) -> Any:
    """Return a nested mapping value if present."""

    value = mapping.get(key)
    if not isinstance(value, Mapping):
        return None
    return value.get(nested_key)


def expect_equal(
    failures: list[str],
    name: str,
    actual: Any,
    expected: Mapping[str, Any],
) -> None:
    """Append a failure when an expected exact value does not match."""

    if name not in expected:
        return
    expected_value = expected[name]
    if actual != expected_value:
        failures.append(f"{name}: expected {expected_value!r}, got {actual!r}")


def expect_subset(
    failures: list[str],
    name: str,
    actual: Mapping[str, Any],
    expected: Any,
) -> None:
    """Append failures for expected key/value pairs missing from a mapping."""

    if not isinstance(expected, Mapping):
        failures.append(f"{name} expected subset is not a mapping")
        return
    for key, expected_value in expected.items():
        actual_value = actual.get(str(key))
        if actual_value != expected_value:
            failures.append(
                f"{name}.{key}: expected {expected_value!r}, got {actual_value!r}"
            )


def require_phrases(
    failures: list[str],
    name: str,
    actual: str,
    required: Any,
) -> None:
    """Append failures for required case-insensitive substrings."""

    phrases = required if isinstance(required, list) else [required]
    normalized = actual.casefold()
    for phrase in phrases:
        if str(phrase).casefold() not in normalized:
            failures.append(f"{name} missing {str(phrase)!r}")


def optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return an optional mapping field."""

    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object.")
    return value


def required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required mapping field."""

    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Case is missing scripted {key!r} object.")
    return value


def list_of_mappings(value: Any, field_name: str) -> list[dict[str, str]]:
    """Parse a list of JSON objects into string dictionaries."""

    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list.")
    items: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name} entries must be objects.")
        items.append({str(key): str(val) for key, val in item.items()})
    return items


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate standalone turn-dispatch contracts.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="scripted uses canned decisions; live uses configured control LLM.",
    )
    return parser


def main() -> int:
    """Run the standalone turn-dispatch evaluator CLI."""

    return run_evaluator_cli(
        lambda args: TurnDispatchEvaluator(
            dataset_path=args.dataset,
            mode=args.mode,
        ),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
