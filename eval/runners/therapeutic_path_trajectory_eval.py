"""Evaluate parent-graph therapeutic path trajectories."""

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
from eval.runners.memory_control_common import memory_snapshot, seed_memory_store

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "therapeutic" / "path_trajectory_v1.json"
)


@dataclass(frozen=True)
class TherapeuticPathTurn:
    """One turn in a parent-graph therapeutic trajectory."""

    message: str
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TherapeuticPathCase:
    """Parsed parent-graph therapeutic trajectory case."""

    id: str
    description: str = ""
    memory_seed: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    turns: list[TherapeuticPathTurn] = field(default_factory=list)


class ScriptedTherapeuticPathLLM:
    """Scripted LLM for one parent-graph therapeutic turn."""

    def __init__(self, turn: TherapeuticPathTurn) -> None:
        self.turn = turn
        self.structured_calls: dict[str, int] = {}
        self.structured_prompts: list[dict[str, str]] = []
        self.text_prompts: list[dict[str, str]] = []
        self.stream_prompts: list[dict[str, str]] = []

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        self.text_prompts.append(
            {
                "prompt": prompt,
                "system_instruction": system_instruction or "",
                "use_search": str(use_search),
            }
        )
        return str(self.turn.scripted.get("text_response", "scripted text response"))

    async def generate_text_stream(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        self.stream_prompts.append(
            {"prompt": prompt, "system_instruction": system_instruction or ""}
        )
        chunks = self.turn.scripted.get("response_chunks")
        if isinstance(chunks, list):
            for chunk in chunks:
                yield str(chunk)
            return
        yield str(self.turn.scripted.get("response_text", "scripted response"))

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
        self.structured_prompts.append(
            {
                "schema": schema_name,
                "prompt": prompt,
                "system_instruction": system_instruction or "",
                "use_search": str(use_search),
            }
        )

        if schema_name == "CrisisAssessmentSchema":
            return response_schema(**_required_mapping(self.turn.scripted, "crisis"))
        if schema_name == "TurnDispatchDecision":
            return response_schema(
                **_required_mapping(self.turn.scripted, "turn_dispatch")
            )
        if schema_name == "DispatchDecision":
            return response_schema(
                **_required_mapping(self.turn.scripted, "therapeutic_dispatch")
            )
        if schema_name == "ExerciseSelectionDecision":
            selected = self.turn.scripted.get("exercise_selection")
            if isinstance(selected, Mapping):
                return response_schema(**selected)
            if selected is None:
                raise RuntimeError("Case did not script exercise_selection.")
            return response_schema(
                exercise_type=str(selected),
                reasoning="scripted exercise selection",
                confidence="high",
            )
        if schema_name == "ExerciseStepDecision":
            step_state = self.turn.scripted.get("step_state")
            if step_state is None:
                raise RuntimeError("Case did not script step_state.")
            return response_schema(
                step_state=str(step_state),
                reasoning="scripted step classification",
                confidence="high",
            )
        if schema_name == "ExtractionResult":
            return response_schema(
                facts=[],
                reason="scripted therapeutic path eval extracts no facts",
            )
        if schema_name == "ProceduralExtractionResult":
            return response_schema(
                rules=[],
                reason="scripted therapeutic path eval extracts no rules",
            )
        if schema_name == "PreferenceRuleDecision":
            return response_schema(
                rule_text=str(
                    self.turn.scripted.get(
                        "preference_rule_text",
                        "You prefer concise responses.",
                    )
                ),
                reasoning="scripted preference rule",
                confidence="high",
            )
        if schema_name == "ProceduralReconciliationDecision":
            return response_schema(
                action="append",
                replace_indexes=[],
                reason="scripted preference reconciliation",
                confidence="high",
            )
        raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")

    def prompt_texts(self) -> list[str]:
        """Return all captured prompt text for containment checks."""

        texts: list[str] = []
        for prompt in (
            *self.structured_prompts,
            *self.text_prompts,
            *self.stream_prompts,
        ):
            texts.append(prompt.get("prompt", ""))
            texts.append(prompt.get("system_instruction", ""))
        return texts


class TherapeuticPathTrajectoryEvaluator(BaseEvaluator[TherapeuticPathCase]):
    """Run parent-graph therapeutic path trajectory checks."""

    def __init__(self, *, dataset_path: str | Path) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name="therapeutic_path_trajectory",
        )

    def parse_case(self, raw_case: Any) -> TherapeuticPathCase:
        """Parse one trajectory case."""

        return _parse_case(raw_case)

    def case_id(self, case: TherapeuticPathCase, index: int) -> str:
        """Return the stable case identifier."""

        return case.id

    async def run_case(self, case: TherapeuticPathCase) -> EvalResult:
        """Run and grade one parent-graph trajectory."""

        artifact = await _run_trajectory(case)
        failures = _grade_trajectory(case, artifact)
        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=1.0 if not failures else 0.0,
            details={
                "description": case.description,
                "failures": failures,
                "artifact": artifact,
            },
        )


async def _run_trajectory(case: TherapeuticPathCase) -> dict[str, Any]:
    from agent.audit.crisis_log import InMemoryCrisisLogBackend
    from agent.memory.modes import MemoryMode
    from agent.memory.store import OpenCouchMemoryStore
    from agent.persistence import PersistentAgentRuntime

    owner_id = f"eval-user-{case.id}"
    thread_id = f"eval-thread-{case.id}"
    store = OpenCouchMemoryStore()
    crisis_log = InMemoryCrisisLogBackend()
    await seed_memory_store(store, owner_id=owner_id, seed=case.memory_seed)

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
        return "Verified factual answer.\n\nSources:\n- Official source", "answered"

    async def fake_crisis_resources(
        state: dict[str, Any],
        *,
        llm_client: Any,
    ) -> tuple[str, list[dict[str, str]], str]:
        tool_calls["crisis_resources"].append({"message": state.get("message")})
        return (
            "Singapore",
            [{"name": "SOS Singapore", "description": "24-hour crisis support"}],
            "found",
        )

    initial_store = await memory_snapshot(store, owner_id=owner_id)
    turns: list[dict[str, Any]] = []

    with (
        patch(
            "agent.nodes.grounded_answer.answer_factual_lookup",
            new=fake_factual_lookup,
        ),
        patch(
            "agent.nodes.crisis_resource_lookup.find_crisis_resources",
            new=fake_crisis_resources,
        ),
    ):
        async with PersistentAgentRuntime(
            sqlite_path=":memory:",
            memory_store=store,
            crisis_log_backend=crisis_log,
            memory_mode=MemoryMode.LOCAL,
            speculative_memory_prefetch=False,
            extract_in_foreground=True,
            finalize_active_sessions_on_close=False,
            session_sweep_interval_seconds=3600.0,
        ) as runtime:
            for index, turn in enumerate(case.turns):
                llm = ScriptedTherapeuticPathLLM(turn)
                before_tool_counts = {
                    name: len(calls) for name, calls in tool_calls.items()
                }
                before_crisis_log_count = await crisis_log.arecord_count()
                result = await runtime.run_turn(
                    thread_id=thread_id,
                    message=turn.message,
                    user_id=owner_id,
                    llm_client=llm,
                    response_llm_client=llm,
                )
                after_crisis_log_count = await crisis_log.arecord_count()
                state = dict(result.state)
                turns.append(
                    {
                        "turn_index": index + 1,
                        "user_message": turn.message,
                        "output": jsonify(result.output),
                        "state_after": jsonify(state),
                        "response_text": result.output.response_text,
                        "response_style": result.output.response_style,
                        "therapeutic_approach": result.output.therapeutic_approach,
                        "routing": {
                            "safety": _routing_decision(state, stage="safety"),
                            "turn_dispatch": _routing_decision(
                                state, stage="turn_dispatch"
                            ),
                            "therapeutic_dispatch": _routing_decision(
                                state, stage="dispatch"
                            ),
                        },
                        "diagnostics": jsonify(state.get("diagnostics") or {}),
                        "working_memory_count": len(state.get("working_memory") or []),
                        "session_memory": jsonify(state.get("session_memory") or {}),
                        "procedural_profile": jsonify(
                            state.get("procedural_profile") or {}
                        ),
                        "exercise_state": jsonify(state.get("exercise_state") or {}),
                        "memory_control": jsonify(state.get("memory_control") or {}),
                        "grounded_lookup": jsonify(state.get("grounded_lookup") or {}),
                        "transcript": jsonify(state.get("transcript") or []),
                        "structured_calls": dict(llm.structured_calls),
                        "prompt_texts": llm.prompt_texts(),
                        "tool_call_counts": {
                            name: len(calls) - before_tool_counts.get(name, 0)
                            for name, calls in tool_calls.items()
                        },
                        "crisis_log_count": (
                            after_crisis_log_count - before_crisis_log_count
                        ),
                    }
                )
            final_state = await runtime.get_state(thread_id)
            final_store = await memory_snapshot(store, owner_id=owner_id)
            crisis_log_count = await crisis_log.arecord_count()

    return {
        "case_id": case.id,
        "description": case.description,
        "initial_store": initial_store,
        "turns": turns,
        "final_state": jsonify(final_state or {}),
        "final_store": final_store,
        "tool_calls": tool_calls,
        "crisis_log_count": crisis_log_count,
    }


def _grade_trajectory(
    case: TherapeuticPathCase,
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

    expected = case.expected
    tool_calls = artifact.get("tool_calls")
    if isinstance(tool_calls, Mapping):
        _expect_equal(
            failures,
            "factual_tool_calls",
            len(tool_calls.get("factual_lookup") or []),
            expected,
        )
        _expect_equal(
            failures,
            "crisis_resource_tool_calls",
            len(tool_calls.get("crisis_resources") or []),
            expected,
        )
    _expect_equal(
        failures,
        "crisis_log_count",
        artifact.get("crisis_log_count"),
        expected,
    )

    final_state = artifact.get("final_state")
    if isinstance(final_state, Mapping):
        transcript = final_state.get("transcript") or []
        _expect_equal(
            failures,
            "final_transcript_length",
            len(transcript) if isinstance(transcript, list) else None,
            expected,
        )
        _grade_expected_mapping(
            failures,
            label="final_exercise_state",
            actual=final_state.get("exercise_state"),
            expected=expected.get("final_exercise_state"),
        )
        _grade_expected_mapping(
            failures,
            label="final_memory_control",
            actual=final_state.get("memory_control"),
            expected=expected.get("final_memory_control"),
        )
        _grade_expected_mapping(
            failures,
            label="final_grounded_lookup",
            actual=final_state.get("grounded_lookup"),
            expected=expected.get("final_grounded_lookup"),
        )
    return failures


def _grade_turn(
    failures: list[str],
    *,
    turn_case: TherapeuticPathTurn,
    artifact: Mapping[str, Any],
) -> None:
    expected = turn_case.expected
    label = f"turn {artifact.get('turn_index')}"
    _expect_equal(
        failures, f"{label}.response_style", artifact.get("response_style"), expected
    )
    _expect_equal(
        failures,
        f"{label}.therapeutic_approach",
        artifact.get("therapeutic_approach"),
        expected,
    )
    _grade_text_contains(
        failures,
        label=f"{label}.response_text",
        text=str(artifact.get("response_text", "")),
        phrases=expected.get("response_text_contains", []),
    )

    routing = (
        artifact.get("routing") if isinstance(artifact.get("routing"), Mapping) else {}
    )
    _expect_equal(failures, f"{label}.safety_decision", routing.get("safety"), expected)
    _expect_equal(
        failures,
        f"{label}.turn_dispatch_decision",
        routing.get("turn_dispatch"),
        expected,
    )
    _expect_equal(
        failures,
        f"{label}.therapeutic_dispatch_decision",
        routing.get("therapeutic_dispatch"),
        expected,
    )

    diagnostics = artifact.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        for key in expected.get("diagnostics_contains", []):
            if key not in diagnostics:
                failures.append(f"{label}.diagnostics missing {key!r}")
        for key in expected.get("diagnostics_not_contains", []):
            if key in diagnostics:
                failures.append(f"{label}.diagnostics unexpectedly had {key!r}")

    _grade_minimum(
        failures,
        label=f"{label}.working_memory_count",
        actual=artifact.get("working_memory_count"),
        expected=expected.get("working_memory_count_min"),
    )
    procedural_profile = (
        artifact.get("procedural_profile")
        if isinstance(artifact.get("procedural_profile"), Mapping)
        else {}
    )
    _grade_text_contains(
        failures,
        label=f"{label}.procedural_rules",
        text="\n".join(
            str(rule) for rule in procedural_profile.get("procedural_rules", [])
        ),
        phrases=expected.get("procedural_rules_contain", []),
    )
    session_memory = (
        artifact.get("session_memory")
        if isinstance(artifact.get("session_memory"), Mapping)
        else {}
    )
    _grade_text_contains(
        failures,
        label=f"{label}.session_memory.summary",
        text=str(session_memory.get("summary", "")),
        phrases=expected.get("session_summary_contains", []),
    )

    prompt_text = "\n".join(str(text) for text in artifact.get("prompt_texts", []))
    _grade_text_contains(
        failures,
        label=f"{label}.prompt_text",
        text=prompt_text,
        phrases=expected.get("prompt_contains", []),
    )
    _grade_expected_mapping(
        failures,
        label=f"{label}.exercise_state",
        actual=artifact.get("exercise_state"),
        expected=expected.get("exercise_state"),
    )
    _grade_expected_mapping(
        failures,
        label=f"{label}.memory_control",
        actual=artifact.get("memory_control"),
        expected=expected.get("memory_control"),
    )
    _grade_expected_mapping(
        failures,
        label=f"{label}.grounded_lookup",
        actual=artifact.get("grounded_lookup"),
        expected=expected.get("grounded_lookup"),
    )
    tool_counts = (
        artifact.get("tool_call_counts")
        if isinstance(artifact.get("tool_call_counts"), Mapping)
        else {}
    )
    _expect_equal(
        failures,
        f"{label}.factual_tool_calls",
        tool_counts.get("factual_lookup"),
        expected,
    )
    _expect_equal(
        failures,
        f"{label}.crisis_resource_tool_calls",
        tool_counts.get("crisis_resources"),
        expected,
    )
    _expect_equal(
        failures,
        f"{label}.crisis_log_count",
        artifact.get("crisis_log_count"),
        expected,
    )
    _grade_expected_counts(
        failures,
        label=f"{label}.structured_calls",
        actual=artifact.get("structured_calls"),
        expected=expected.get("structured_calls"),
    )
    _grade_transcript(failures, label=label, artifact=artifact, expected=expected)


def _grade_transcript(
    failures: list[str],
    *,
    label: str,
    artifact: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    transcript = artifact.get("transcript")
    if not isinstance(transcript, list):
        if any(
            key in expected
            for key in (
                "transcript_length",
                "assistant_turn_count",
                "last_assistant_response_style",
            )
        ):
            failures.append(f"{label}.transcript is not a list")
        return

    _expect_equal(failures, f"{label}.transcript_length", len(transcript), expected)
    assistant_turns = [
        item
        for item in transcript
        if isinstance(item, Mapping) and item.get("role") == "assistant"
    ]
    _expect_equal(
        failures,
        f"{label}.assistant_turn_count",
        len(assistant_turns),
        expected,
    )
    if "last_assistant_response_style" in expected:
        actual = assistant_turns[-1].get("response_style") if assistant_turns else None
        if actual != expected["last_assistant_response_style"]:
            failures.append(
                f"{label}.last_assistant_response_style: expected "
                f"{expected['last_assistant_response_style']!r}, got {actual!r}"
            )


def _routing_decision(state: Mapping[str, Any], *, stage: str) -> str | None:
    diagnostics = state.get("diagnostics") or {}
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


def _grade_expected_mapping(
    failures: list[str],
    *,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if not isinstance(expected, Mapping):
        return
    actual_mapping = actual if isinstance(actual, Mapping) else {}
    for key, expected_value in expected.items():
        actual_value = actual_mapping.get(key)
        if isinstance(expected_value, Mapping):
            _grade_expected_mapping(
                failures,
                label=f"{label}.{key}",
                actual=actual_value,
                expected=expected_value,
            )
        elif actual_value != expected_value:
            failures.append(
                f"{label}.{key}: expected {expected_value!r}, got {actual_value!r}"
            )


def _grade_expected_counts(
    failures: list[str],
    *,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if not isinstance(expected, Mapping):
        return
    actual_mapping = actual if isinstance(actual, Mapping) else {}
    for key, expected_value in expected.items():
        actual_value = actual_mapping.get(key, 0)
        if actual_value != expected_value:
            failures.append(
                f"{label}.{key}: expected {expected_value!r}, got {actual_value!r}"
            )


def _grade_text_contains(
    failures: list[str],
    *,
    label: str,
    text: str,
    phrases: Any,
) -> None:
    phrase_list = phrases if isinstance(phrases, list) else [phrases]
    haystack = text.casefold()
    for phrase in [item for item in phrase_list if item is not None]:
        if str(phrase).casefold() not in haystack:
            failures.append(f"{label} missing {str(phrase)!r}")


def _grade_minimum(
    failures: list[str],
    *,
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if expected is None:
        return
    if not isinstance(actual, int | float) or actual < expected:
        failures.append(f"{label}: expected >= {expected!r}, got {actual!r}")


def _expect_equal(
    failures: list[str],
    name: str,
    actual: Any,
    expected: Mapping[str, Any],
) -> None:
    key = name.rsplit(".", 1)[-1]
    if key not in expected:
        return
    if actual != expected[key]:
        failures.append(f"{name}: expected {expected[key]!r}, got {actual!r}")


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Case is missing scripted {key!r} object.")
    return value


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object.")
    return value


def _parse_case(raw_case: Any) -> TherapeuticPathCase:
    if not isinstance(raw_case, Mapping):
        raise TypeError("Therapeutic path trajectory cases must be JSON objects.")
    turns_raw = raw_case.get("turns")
    if not isinstance(turns_raw, list):
        raise TypeError("Therapeutic path trajectory cases need a turns list.")
    turns: list[TherapeuticPathTurn] = []
    for raw_turn in turns_raw:
        if not isinstance(raw_turn, Mapping):
            raise TypeError("Turn entries must be objects.")
        turns.append(
            TherapeuticPathTurn(
                message=str(raw_turn["message"]),
                scripted=dict(_optional_mapping(raw_turn, "scripted")),
                expected=dict(_optional_mapping(raw_turn, "expected")),
            )
        )
    return TherapeuticPathCase(
        id=str(raw_case["id"]),
        description=str(raw_case.get("description", "")),
        memory_seed=dict(_optional_mapping(raw_case, "memory_seed")),
        expected=dict(_optional_mapping(raw_case, "expected")),
        turns=turns,
    )


def jsonify(value: Any) -> Any:
    """Convert model-ish values into JSON-compatible data."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonify(item) for item in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser(
        "Evaluate parent-graph therapeutic path trajectories."
    )
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    return parser


def main() -> int:
    """Run the therapeutic path trajectory evaluator CLI."""

    return run_evaluator_cli(
        lambda args: TherapeuticPathTrajectoryEvaluator(dataset_path=args.dataset),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
