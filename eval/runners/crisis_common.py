"""Shared helpers for crisis-gate evaluators."""

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


@dataclass(frozen=True)
class CrisisEvalCase:
    """Parsed crisis eval case."""

    id: str
    message: str
    description: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] = field(default_factory=dict)


class ScriptedCrisisLLM:
    """LLM-shaped fake for crisis evals."""

    def __init__(
        self,
        case: CrisisEvalCase,
        *,
        text_delegate: Any | None = None,
    ) -> None:
        self.case = case
        self.text_delegate = text_delegate
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
        if self.text_delegate is not None:
            return await self.text_delegate.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                use_search=use_search,
            )
        return str(self.case.scripted.get("text_response", "scripted text response"))

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        self.text_stream_calls += 1
        if self.text_delegate is not None:
            async for chunk in self.text_delegate.generate_text_stream(
                prompt=prompt,
                system_instruction=system_instruction,
            ):
                yield chunk
            return
        yield str(
            self.case.scripted.get(
                "response_text",
                "scripted crisis response",
            )
        )

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

        if schema_name == "CrisisAssessmentSchema":
            return response_schema(**required_mapping(self.case.scripted, "crisis"))

        if schema_name == "TurnDispatchDecision":
            return response_schema(
                **required_mapping(self.case.scripted, "turn_dispatch")
            )

        if schema_name == "DispatchDecision":
            return response_schema(
                **required_mapping(self.case.scripted, "therapeutic_dispatch")
            )

        if schema_name == "CrisisLocationDecision":
            return response_schema(
                **dict(optional_mapping(self.case.scripted, "crisis_location"))
            )

        if schema_name == "CrisisResourceLookupResult":
            return response_schema(
                **required_mapping(self.case.scripted, "crisis_resource_lookup")
            )

        if self.text_delegate is not None:
            return await self.text_delegate.generate_structured(
                prompt=prompt,
                response_schema=response_schema,
                system_instruction=system_instruction,
                use_search=use_search,
            )

        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")


def parse_crisis_case(raw_case: Any) -> CrisisEvalCase:
    """Parse one raw crisis eval case.

    Args:
        raw_case (Any): Raw JSON case object.

    Returns:
        CrisisEvalCase: Parsed case.
    """

    if not isinstance(raw_case, Mapping):
        raise TypeError("Crisis eval cases must be JSON objects.")
    return CrisisEvalCase(
        id=str(raw_case["id"]),
        description=str(raw_case.get("description", "")),
        message=str(raw_case["message"]),
        history=list_of_mappings(raw_case.get("history", []), "history"),
        state=dict(optional_mapping(raw_case, "state")),
        scripted=dict(optional_mapping(raw_case, "scripted")),
        expected=dict(optional_mapping(raw_case, "expected")),
        rubric=dict(optional_mapping(raw_case, "rubric")),
    )


def build_graph_state(case: CrisisEvalCase) -> dict[str, Any]:
    """Build parent-graph state for a crisis eval case.

    Args:
        case (CrisisEvalCase): Parsed eval case.

    Returns:
        dict[str, Any]: Mutable graph input state.
    """

    from agent.graph import build_initial_state
    from agent.models import AgentInput, Message
    from eval.runners.therapeutic_common import deep_update

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


def routing_decision(output: Mapping[str, Any], *, stage: str) -> str | None:
    """Return the latest routing-trace decision for a stage."""

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


def optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object.")
    return value


def required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Case is missing scripted {key!r} object.")
    return value


def list_of_mappings(value: Any, field_name: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list.")
    items: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name} entries must be objects.")
        items.append({str(key): str(val) for key, val in item.items()})
    return items
