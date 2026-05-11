"""Evaluate direct semantic and procedural memory extraction quality."""

from __future__ import annotations

import argparse
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "memory" / "extraction_quality_v1.json"
)


@dataclass(frozen=True)
class MemoryExtractionQualityCase:
    """Parsed memory extraction quality case."""

    id: str
    layer: str
    message: str
    description: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


class ScriptedExtractionLLM:
    """Scripted structured-output client for extraction quality evals."""

    def __init__(self, scripted: Mapping[str, Any]) -> None:
        self.scripted = dict(scripted)
        self.structured_calls: dict[str, int] = {}

    async def generate_text(
        self,
        *,
        prompt: str,  # noqa: ARG002 - LLM protocol
        system_instruction: str | None = None,  # noqa: ARG002 - LLM protocol
        use_search: bool = False,  # noqa: ARG002 - LLM protocol
    ) -> str:
        return "unused"

    async def generate_text_stream(
        self,
        *,
        prompt: str,  # noqa: ARG002 - LLM protocol
        system_instruction: str | None = None,  # noqa: ARG002 - LLM protocol
    ) -> AsyncIterator[str]:
        yield "unused"

    async def generate_structured(
        self,
        *,
        prompt: str,  # noqa: ARG002 - LLM protocol
        response_schema: type[Any],
        system_instruction: str | None = None,  # noqa: ARG002 - LLM protocol
    ) -> Any:
        schema_name = response_schema.__name__
        self.structured_calls[schema_name] = (
            self.structured_calls.get(schema_name, 0) + 1
        )

        if schema_name == "ExtractionResult":
            payload = self.scripted.get("semantic_extraction")
        elif schema_name == "ProceduralExtractionResult":
            payload = self.scripted.get("procedural_extraction")
        else:
            raise RuntimeError(f"Unexpected structured schema {schema_name!r}.")

        if not isinstance(payload, Mapping):
            raise RuntimeError(f"Scripted case missing payload for {schema_name}.")
        return response_schema(**payload)


class MemoryExtractionQualityEvaluator(BaseEvaluator[MemoryExtractionQualityCase]):
    """Run direct extractor quality checks."""

    def __init__(self, *, dataset_path: str | Path, mode: str) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"memory_extraction_quality_{mode}",
        )
        self.mode = mode
        self._live_llm: Any | None = None

    def parse_case(self, raw_case: Any) -> MemoryExtractionQualityCase:
        if not isinstance(raw_case, Mapping):
            raise TypeError("Memory extraction eval cases must be JSON objects.")
        return MemoryExtractionQualityCase(
            id=str(raw_case["id"]),
            layer=str(raw_case["layer"]),
            message=str(raw_case["message"]),
            description=str(raw_case.get("description", "")),
            history=[dict(item) for item in raw_case.get("history", [])],
            scripted=dict(_optional_mapping(raw_case, "scripted")),
            expected=dict(_optional_mapping(raw_case, "expected")),
        )

    def case_id(self, case: MemoryExtractionQualityCase, index: int) -> str:
        return case.id

    async def run_case(self, case: MemoryExtractionQualityCase) -> EvalResult:
        artifact = await self._run_extraction(case)
        failures = _grade_case(case, artifact)
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

    async def _run_extraction(
        self,
        case: MemoryExtractionQualityCase,
    ) -> dict[str, Any]:
        from agent.memory.models import ExtractionResult, ProceduralExtractionResult
        from agent.memory.prompts.extraction import (
            build_extraction_system_prompt,
            build_extraction_user_prompt,
        )
        from agent.memory.prompts.procedural import (
            build_procedural_writer_system_prompt,
            build_procedural_writer_user_prompt,
        )
        from agent.state import AgentState

        state = cast(
            AgentState,
            {
                "message": case.message,
                "history": case.history,
                "user_id": "eval-user",
                "session_id": "eval-session",
                "session_progress": {"turn_count": 1},
            },
        )
        llm = self._llm_for_case(case)

        try:
            if case.layer == "semantic":
                result = await llm.generate_structured(
                    prompt=build_extraction_user_prompt(state, turn_index=0),
                    response_schema=ExtractionResult,
                    system_instruction=build_extraction_system_prompt(),
                )
                return {
                    "layer": case.layer,
                    "result": result.model_dump(mode="json"),
                    "structured_calls": getattr(llm, "structured_calls", {}),
                }
            if case.layer == "procedural":
                result = await llm.generate_structured(
                    prompt=build_procedural_writer_user_prompt(state),
                    response_schema=ProceduralExtractionResult,
                    system_instruction=build_procedural_writer_system_prompt(),
                )
                return {
                    "layer": case.layer,
                    "result": result.model_dump(mode="json"),
                    "structured_calls": getattr(llm, "structured_calls", {}),
                }
            raise ValueError(f"Unsupported memory extraction layer {case.layer!r}.")
        except Exception as exc:  # noqa: BLE001 - eval artifact records failures
            return {
                "layer": case.layer,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "structured_calls": getattr(llm, "structured_calls", {}),
            }

    def _llm_for_case(self, case: MemoryExtractionQualityCase) -> Any:
        if self.mode == "live":
            if self._live_llm is None:
                from config import create_configured_control_llm_client

                self._live_llm = create_configured_control_llm_client()
            return self._live_llm
        return ScriptedExtractionLLM(case.scripted)


def _grade_case(
    case: MemoryExtractionQualityCase,
    artifact: Mapping[str, Any],
) -> list[str]:
    if case.layer == "semantic":
        return _grade_semantic(case, artifact)
    if case.layer == "procedural":
        return _grade_procedural(case, artifact)
    return [f"unsupported layer {case.layer!r}"]


def _grade_semantic(
    case: MemoryExtractionQualityCase,
    artifact: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected = case.expected
    result = artifact.get("result")
    facts = _result_items(result, "facts")
    expected_count = expected.get("count")

    if artifact.get("exception_type"):
        failures.append(f"unexpected exception: {artifact.get('exception')}")
        return failures

    if expected_count is not None and len(facts) != int(expected_count):
        failures.append(f"count: expected {expected_count}, got {len(facts)}")

    unmatched = list(facts)
    for index, expected_fact in enumerate(_optional_sequence(expected, "facts")):
        match = _pop_matching_fact(unmatched, expected_fact)
        if match is None:
            failures.append(f"fact[{index}] no matching extracted fact")

    return failures


def _grade_procedural(
    case: MemoryExtractionQualityCase,
    artifact: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected = case.expected
    result = artifact.get("result")
    rules = _result_items(result, "rules")
    expected_count = expected.get("count")

    if artifact.get("exception_type"):
        failures.append(f"unexpected exception: {artifact.get('exception')}")
        return failures

    if expected_count is not None and len(rules) != int(expected_count):
        failures.append(f"count: expected {expected_count}, got {len(rules)}")

    unmatched = list(rules)
    for index, expected_rule in enumerate(_optional_sequence(expected, "rules")):
        match = _pop_matching_rule(unmatched, expected_rule)
        if match is None:
            failures.append(f"rule[{index}] no matching extracted rule")

    return failures


def _pop_matching_fact(
    facts: list[Mapping[str, Any]],
    expected: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for index, fact in enumerate(facts):
        if _fact_matches(fact, expected):
            return facts.pop(index)
    return None


def _fact_matches(fact: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if expected.get("category") and fact.get("category") != expected["category"]:
        return False
    if expected.get("predicate") and fact.get("predicate") != expected["predicate"]:
        return False

    obj = fact.get("object")
    obj_mapping = obj if isinstance(obj, Mapping) else {}
    if (
        expected.get("object_type")
        and obj_mapping.get("type") != expected["object_type"]
    ):
        return False
    object_text = str(obj_mapping.get("identifier", ""))
    if not _contains_all(object_text, _string_list(expected.get("object_contains"))):
        return False

    evidence = str(fact.get("evidence_quote", ""))
    return _contains_all(evidence, _string_list(expected.get("evidence_contains")))


def _pop_matching_rule(
    rules: list[Mapping[str, Any]],
    expected: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    for index, rule in enumerate(rules):
        if _rule_matches(rule, expected):
            return rules.pop(index)
    return None


def _rule_matches(rule: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    rule_text = str(rule.get("rule", ""))
    evidence_text = " ".join(str(item) for item in rule.get("evidence", []))

    if not _contains_all(rule_text, _string_list(expected.get("rule_contains"))):
        return False
    if _contains_any(rule_text, _string_list(expected.get("forbidden_rule_contains"))):
        return False
    return _contains_all(
        evidence_text,
        _string_list(expected.get("evidence_contains")),
    )


def _result_items(result: Any, key: str) -> list[Mapping[str, Any]]:
    if not isinstance(result, Mapping):
        return []
    value = result.get(key)
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _contains_all(text: str, phrases: Sequence[str]) -> bool:
    lowered = text.lower()
    return all(phrase.lower() in lowered for phrase in phrases)


def _contains_any(text: str, phrases: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(phrase.lower() in lowered for phrase in phrases)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [str(item) for item in value]
    return [str(value)]


def _optional_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def _optional_sequence(
    mapping: Mapping[str, Any], key: str
) -> Sequence[Mapping[str, Any]]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for this evaluator.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """

    parser = build_base_arg_parser(__doc__ or "")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="Use scripted fixture outputs or the configured live control LLM.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the memory extraction quality evaluator CLI.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments.

    Returns:
        int: Process exit code.
    """

    parser = build_parser()
    return run_evaluator_cli(
        lambda args: MemoryExtractionQualityEvaluator(
            dataset_path=args.dataset,
            mode=args.mode,
        ),
        parser=parser,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
