"""Shared therapeutic eval helpers."""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

ALLOWED_THERAPEUTIC_OUTPUT_KEYS = {
    "response_text",
    "response_style",
    "therapeutic_approach",
    "session_progress",
    "response_guidance",
    "exercise_state",
    "diagnostics",
}

_DICT_REDUCER_KEYS = {
    "diagnostics",
    "exercise_state",
    "grounded_lookup",
    "memory_control",
    "procedural_profile",
    "session_memory",
    "session_progress",
}


@dataclass(frozen=True)
class ScriptedDispatch:
    """Scripted dispatch decision returned by the fake control LLM."""

    response_style: str
    exercise_start_basis: str
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
                exercise_start_basis=dispatch.exercise_start_basis,
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


async def invoke_therapeutic_branch(
    case: TherapeuticEvalCase,
    *,
    llm_client: Any | None,
    response_llm: Any | None = None,
    memory_store: Any | None = None,
) -> dict[str, Any]:
    """Invoke the real therapeutic branch services for a case.

    Args:
        case (TherapeuticEvalCase): Parsed eval case.
        llm_client (Any | None): Control-plane LLM client.
        response_llm (Any | None): Response LLM client. Defaults to llm_client.
        memory_store (Any | None): Optional shared memory store.

    Returns:
        dict[str, Any]: Parent-visible therapeutic branch output.
    """

    from agent.audit.crisis_log import InMemoryCrisisLogBackend
    from agent.graph import build_initial_state
    from agent.memory.load_turn import build_load_memory_delta
    from agent.memory.modes import MemoryMode
    from agent.memory.store import OpenCouchMemoryStore
    from agent.models import AgentInput, Message
    from agent.runtime_context import WorkflowContext
    from agent.therapeutic.dispatch import (
        build_therapeutic_dispatch_update,
        plan_therapeutic_route,
    )
    from agent.therapeutic.exercises.runner import ExerciseRunner
    from agent.therapeutic.response import run_therapeutic_response_node

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

    context = WorkflowContext(
        llm_client=llm_client,
        response_llm=response_llm or llm_client,
        memory_store=memory_store or OpenCouchMemoryStore(),
        crisis_log_backend=InMemoryCrisisLogBackend(),
        memory_mode=MemoryMode.INCOGNITO,
    )
    _apply_agent_delta(state, await build_load_memory_delta(state, context))

    dispatch_plan = await plan_therapeutic_route(state, llm_client)
    _apply_agent_delta(
        state,
        build_therapeutic_dispatch_update(state, dispatch_plan),
    )

    if state.get("response_style") == "guided_exercise":
        delta = await ExerciseRunner(
            classifier_llm=llm_client,
            response_llm=response_llm or llm_client,
            memory_store=context.memory_store,
            memory_mode=context.memory_mode,
        ).run(state)
    else:
        delta = await run_therapeutic_response_node(
            state,
            SimpleNamespace(context=context),
        )
    _apply_agent_delta(state, delta)
    return {key: state[key] for key in ALLOWED_THERAPEUTIC_OUTPUT_KEYS if key in state}


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

    actual_entry = last_routing_entry(output)
    actual_decision = (
        str(actual_entry.get("decision"))
        if isinstance(actual_entry, Mapping) and actual_entry.get("decision")
        else None
    )
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

    _expect_equal_or_any_of(
        failures,
        "routing_source",
        actual_entry.get("source") if isinstance(actual_entry, Mapping) else None,
        exact=expected.get("routing_source"),
    )
    _expect_equal_or_any_of(
        failures,
        "routing_exercise_start_basis",
        (
            actual_entry.get("exercise_start_basis")
            if isinstance(actual_entry, Mapping)
            else None
        ),
        exact=expected.get("routing_exercise_start_basis"),
    )

    return failures


def last_routing_decision(output: Mapping[str, Any]) -> str | None:
    """Return the last dispatch routing decision from diagnostics.

    Args:
        output (Mapping[str, Any]): Subgraph output.

    Returns:
        str | None: Last routing decision when present.
    """

    entry = last_routing_entry(output)
    if not isinstance(entry, Mapping):
        return None
    decision = entry.get("decision")
    return str(decision) if decision is not None else None


def last_routing_entry(output: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the last dispatch routing trace entry.

    Args:
        output (Mapping[str, Any]): Subgraph output.

    Returns:
        Mapping[str, Any] | None: Last routing entry when present.
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
    return last


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


def _apply_agent_delta(target: dict[str, Any], delta: Mapping[str, Any]) -> None:
    for key, value in delta.items():
        if key in _DICT_REDUCER_KEYS:
            target[key] = {
                **dict(target.get(key, {}) or {}),
                **dict(value or {}),
            }
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
            exercise_start_basis=str(dispatch["exercise_start_basis"]),
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
