"""Shared therapeutic eval helpers."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

ALLOWED_THERAPEUTIC_OUTPUT_KEYS = {
    "response_text",
    "response_style",
    "therapeutic_approach",
    "exercise_state",
    "diagnostics",
}


@dataclass(frozen=True)
class ScriptedDispatch:
    """Scripted dispatch decision returned by the fake control LLM."""

    response_style: str
    therapeutic_approach: str = "none"
    reasoning: str = "scripted therapeutic eval decision"
    confidence: str = "high"


@dataclass(frozen=True)
class ScriptedLLMConfig:
    """Scripted LLM outputs for one therapeutic eval case."""

    dispatch: ScriptedDispatch
    response_text: str
    exercise_selection: str | None = None
    step_state: str | None = None


@dataclass(frozen=True)
class TherapeuticEvalCase:
    """Parsed therapeutic eval case."""

    id: str
    message: str
    description: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    scripted: ScriptedLLMConfig | None = None
    expected: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] = field(default_factory=dict)


class ScriptedTherapeuticLLM:
    """LLM-shaped scripted client for deterministic therapeutic evals."""

    def __init__(self, config: ScriptedLLMConfig) -> None:
        self.config = config

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return self.config.response_text

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        yield self.config.response_text

    async def generate_structured(
        self,
        *,
        prompt: str,
        response_schema: type[Any],
        system_instruction: str | None = None,
    ) -> Any:
        schema_name = response_schema.__name__
        if schema_name == "DispatchDecision":
            dispatch = self.config.dispatch
            return response_schema(
                response_style=dispatch.response_style,
                therapeutic_approach=dispatch.therapeutic_approach,
                reasoning=dispatch.reasoning,
                confidence=dispatch.confidence,
            )
        if schema_name == "ExerciseSelectionDecision":
            if self.config.exercise_selection is None:
                raise RuntimeError("Case did not script exercise_selection.")
            return response_schema(
                exercise_type=self.config.exercise_selection,
                reasoning="scripted exercise selection",
                confidence="high",
            )
        if schema_name == "ExerciseStepDecision":
            if self.config.step_state is None:
                raise RuntimeError("Case did not script step_state.")
            return response_schema(
                step_state=self.config.step_state,
                reasoning="scripted exercise step decision",
                confidence="high",
            )
        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")


def parse_therapeutic_case(raw_case: Any) -> TherapeuticEvalCase:
    """Parse one therapeutic eval case.

    Args:
        raw_case (Any): Raw JSON case object.

    Returns:
        TherapeuticEvalCase: Parsed case.
    """

    if not isinstance(raw_case, Mapping):
        raise TypeError("Therapeutic eval cases must be JSON objects.")

    scripted = _parse_scripted(raw_case.get("scripted"))
    return TherapeuticEvalCase(
        id=str(raw_case["id"]),
        description=str(raw_case.get("description", "")),
        message=str(raw_case["message"]),
        history=_list_of_mappings(raw_case.get("history", []), field_name="history"),
        state=dict(_optional_mapping(raw_case, "state")),
        scripted=scripted,
        expected=dict(_optional_mapping(raw_case, "expected")),
        rubric=dict(_optional_mapping(raw_case, "rubric")),
    )


def build_scripted_llm(case: TherapeuticEvalCase) -> ScriptedTherapeuticLLM:
    """Build the scripted LLM for a case.

    Args:
        case (TherapeuticEvalCase): Parsed case with scripted LLM outputs.

    Returns:
        ScriptedTherapeuticLLM: Scripted LLM client.
    """

    if case.scripted is None:
        raise ValueError(f"Case {case.id!r} does not define scripted LLM outputs.")
    return ScriptedTherapeuticLLM(case.scripted)


async def invoke_therapeutic_subgraph(
    case: TherapeuticEvalCase,
    *,
    llm_client: Any | None,
    response_llm: Any | None = None,
) -> dict[str, Any]:
    """Invoke the real therapeutic subgraph for a case.

    Args:
        case (TherapeuticEvalCase): Parsed eval case.
        llm_client (Any | None): Control-plane LLM client.
        response_llm (Any | None): Response LLM client. Defaults to llm_client.

    Returns:
        dict[str, Any]: Parent-visible subgraph output.
    """

    from agent.audit.crisis_log import InMemoryCrisisLogBackend
    from agent.graph import build_initial_state
    from agent.memory.modes import MemoryMode
    from agent.memory.store import OpenCouchMemoryStore
    from agent.models import AgentInput, Message
    from agent.runtime_context import WorkflowContext
    from agent.therapeutic.graph import build_therapeutic_subgraph

    history = [Message.model_validate(item) for item in case.history]
    state = dict(
        build_initial_state(
            AgentInput(
                message=case.message,
                history=history,
                session_id=case.id,
            ),
            include_input_history=True,
        )
    )
    deep_update(state, case.state)

    subgraph = build_therapeutic_subgraph()
    raw_output = await subgraph.ainvoke(
        state,
        context=WorkflowContext(
            llm_client=llm_client,
            response_llm=response_llm or llm_client,
            memory_store=OpenCouchMemoryStore(),
            crisis_log_backend=InMemoryCrisisLogBackend(),
            memory_mode=MemoryMode.INCOGNITO,
        ),
    )
    return dict(raw_output)


def grade_therapeutic_output(
    case: TherapeuticEvalCase,
    output: dict[str, Any],
) -> list[str]:
    """Grade common therapeutic output expectations.

    Args:
        case (TherapeuticEvalCase): Parsed case.
        output (dict[str, Any]): Subgraph output.

    Returns:
        list[str]: Failure messages.
    """

    failures: list[str] = []
    unexpected_keys = set(output) - ALLOWED_THERAPEUTIC_OUTPUT_KEYS
    if unexpected_keys:
        failures.append(f"unexpected output keys: {sorted(unexpected_keys)}")

    expected = case.expected
    _expect_equal_or_any_of(
        failures,
        "response_style",
        output.get("response_style"),
        exact=expected.get("response_style"),
        any_of=expected.get("response_style_any_of")
        or expected.get("acceptable_response_styles"),
    )
    if "therapeutic_approach" in expected or "therapeutic_approach_any_of" in expected:
        _expect_equal_or_any_of(
            failures,
            "therapeutic_approach",
            output.get("therapeutic_approach"),
            exact=expected.get("therapeutic_approach"),
            any_of=expected.get("therapeutic_approach_any_of")
            or expected.get("acceptable_therapeutic_approaches"),
        )

    response_text = str(output.get("response_text", ""))
    if expected.get("response_text_non_empty") and not response_text.strip():
        failures.append("response_text is empty")
    for needle in expected.get("response_text_contains", []):
        if str(needle).casefold() not in response_text.casefold():
            failures.append(f"response_text missing {needle!r}")
    for needle in expected.get("response_text_not_contains", []):
        if str(needle).casefold() in response_text.casefold():
            failures.append(f"response_text contains forbidden text {needle!r}")

    expected_exercise_state = expected.get("exercise_state")
    if isinstance(expected_exercise_state, Mapping):
        actual_exercise_state = output.get("exercise_state") or {}
        for key, expected_value in expected_exercise_state.items():
            actual_value = actual_exercise_state.get(key)
            if actual_value != expected_value:
                failures.append(
                    "exercise_state."
                    f"{key}: expected {expected_value!r}, got {actual_value!r}"
                )

    actual_decision = last_routing_decision(output)
    expected_routing_decision = expected.get("routing_decision")
    if (
        expected_routing_decision is not None
        and actual_decision != expected_routing_decision
    ):
        failures.append(
            "routing_decision: "
            f"expected {expected_routing_decision!r}, got {actual_decision!r}"
        )
    expected_routing_any_of = expected.get("routing_decision_any_of")
    if expected_routing_any_of is not None:
        if not isinstance(expected_routing_any_of, list):
            failures.append("routing_decision_any_of must be a list.")
        elif actual_decision not in expected_routing_any_of:
            failures.append(
                "routing_decision: expected one of "
                f"{expected_routing_any_of!r}, got {actual_decision!r}"
            )

    return failures


def last_routing_decision(output: Mapping[str, Any]) -> str | None:
    """Return the last dispatch routing decision from diagnostics.

    Args:
        output (Mapping[str, Any]): Subgraph output.

    Returns:
        str | None: Last routing decision when present.
    """

    diagnostics = output.get("diagnostics") or {}
    if not isinstance(diagnostics, Mapping):
        return None
    trace = diagnostics.get("routing_trace") or []
    if not trace:
        return None
    last = trace[-1]
    if not isinstance(last, Mapping):
        return None
    decision = last.get("decision")
    return str(decision) if decision is not None else None


def build_live_therapeutic_llms() -> tuple[Any, Any]:
    """Build configured live control and response LLM clients.

    Returns:
        tuple[Any, Any]: Control-plane and response LLM clients.
    """

    from config import (
        create_configured_control_llm_client,
        create_configured_response_llm_client,
    )

    return (
        create_configured_control_llm_client(),
        create_configured_response_llm_client("fast"),
    )


def deep_update(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    """Recursively apply a JSON-style object patch.

    Args:
        target (dict[str, Any]): Mutable target mapping.
        patch (Mapping[str, Any]): Patch values.

    Returns:
        None.
    """

    for key, value in patch.items():
        current = target.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            deep_update(current, value)
        else:
            target[key] = value


def _parse_scripted(value: Any) -> ScriptedLLMConfig | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("'scripted' must be an object when provided.")
    dispatch = _require_mapping(value, "dispatch")
    return ScriptedLLMConfig(
        dispatch=ScriptedDispatch(
            response_style=str(dispatch["response_style"]),
            therapeutic_approach=str(dispatch.get("therapeutic_approach", "none")),
            reasoning=str(
                dispatch.get("reasoning", "scripted therapeutic eval decision")
            ),
            confidence=str(dispatch.get("confidence", "high")),
        ),
        response_text=str(value["response_text"]),
        exercise_selection=(
            str(value["exercise_selection"])
            if value.get("exercise_selection") is not None
            else None
        ),
        step_state=str(value["step_state"])
        if value.get("step_state") is not None
        else None,
    )


def _expect_equal_or_any_of(
    failures: list[str],
    field_name: str,
    actual: Any,
    *,
    exact: Any = None,
    any_of: Any = None,
) -> None:
    if exact is not None and actual != exact:
        failures.append(f"{field_name}: expected {exact!r}, got {actual!r}")
    if any_of is None:
        return
    if not isinstance(any_of, list | tuple):
        failures.append(f"{field_name}_any_of must be a list.")
        return
    if actual not in any_of:
        failures.append(f"{field_name}: expected one of {any_of!r}, got {actual!r}")


def _require_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key!r} must be an object.")
    return value


def _optional_mapping(source: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = source.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key!r} must be an object when provided.")
    return value


def _list_of_mappings(value: Any, *, field_name: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name!r} must be a list.")
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name!r} entries must be objects.")
        rows.append({str(key): str(item_value) for key, item_value in item.items()})
    return rows
