"""Evaluate semantic and procedural memory reconciliation decisions."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    _REPO_ROOT / "eval" / "datasets" / "memory" / "reconciliation_v1.json"
)


@dataclass(frozen=True)
class MemoryReconciliationCase:
    """Parsed memory reconciliation case."""

    id: str
    layer: str
    description: str = ""
    modes: tuple[str, ...] = ("scripted", "live")
    semantic_fact: dict[str, Any] = field(default_factory=dict)
    existing_semantic_records: tuple[dict[str, Any], ...] = ()
    procedural_rule: dict[str, Any] = field(default_factory=dict)
    existing_procedural_rules: tuple[dict[str, Any], ...] = ()
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    expected_by_mode: dict[str, dict[str, Any]] = field(default_factory=dict)

    def expected_for_mode(self, mode: str) -> dict[str, Any]:
        """Return expected grading fields for one evaluator mode.

        Args:
            mode (str): Evaluator mode, such as ``"scripted"`` or ``"live"``.

        Returns:
            dict[str, Any]: Base expectations merged with mode-specific fields.
        """

        expected = dict(self.expected)
        expected.update(self.expected_by_mode.get(mode, {}))
        return expected


class ScriptedReconciliationLLM:
    """Scripted structured-output client for reconciliation evals."""

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
        if self.scripted.get("fail"):
            raise RuntimeError("scripted reconciliation LLM failure")
        schema_name = response_schema.__name__
        self.structured_calls[schema_name] = (
            self.structured_calls.get(schema_name, 0) + 1
        )
        decision = self.scripted.get("decision")
        if not isinstance(decision, Mapping):
            raise RuntimeError("Reconciliation eval case needs scripted.decision.")
        return response_schema(**decision)


class CountingReconciliationLLM:
    """Live LLM wrapper that records structured call counts for grading."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.structured_calls: dict[str, int] = {}

    def reset_structured_calls(self) -> None:
        """Clear per-case structured call counters.

        Returns:
            None: Mutates the wrapper's call counters.
        """

        self.structured_calls.clear()

    async def generate_text(
        self,
        *,
        prompt: str,
        system_instruction: str | None = None,
        use_search: bool = False,
    ) -> str:
        return await self.client.generate_text(
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
        async for chunk in self.client.generate_text_stream(
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
        return await self.client.generate_structured(
            prompt=prompt,
            response_schema=response_schema,
            system_instruction=system_instruction,
            use_search=use_search,
        )


class MemoryReconciliationEvaluator(BaseEvaluator[MemoryReconciliationCase]):
    """Run direct memory reconciliation checks."""

    def __init__(self, *, dataset_path: str | Path, mode: str) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"memory_reconciliation_{mode}",
        )
        self.mode = mode
        self._live_llm: Any | None = None

    def parse_case(self, raw_case: Any) -> MemoryReconciliationCase:
        if not isinstance(raw_case, Mapping):
            raise TypeError("Memory reconciliation eval cases must be JSON objects.")
        return MemoryReconciliationCase(
            id=str(raw_case["id"]),
            layer=str(raw_case["layer"]),
            description=str(raw_case.get("description", "")),
            modes=tuple(
                str(mode) for mode in raw_case.get("modes", ["scripted", "live"])
            ),
            semantic_fact=dict(_optional_mapping(raw_case, "semantic_fact")),
            existing_semantic_records=tuple(
                _mapping_sequence(raw_case, "existing_semantic_records")
            ),
            procedural_rule=dict(_optional_mapping(raw_case, "procedural_rule")),
            existing_procedural_rules=tuple(
                _mapping_sequence(raw_case, "existing_procedural_rules")
            ),
            scripted=dict(_optional_mapping(raw_case, "scripted")),
            expected=dict(_optional_mapping(raw_case, "expected")),
            expected_by_mode=_expected_by_mode(raw_case),
        )

    def case_id(self, case: MemoryReconciliationCase, index: int) -> str:
        return case.id

    def load_cases(self) -> list[MemoryReconciliationCase]:
        cases = super().load_cases()
        return [case for case in cases if self.mode in case.modes]

    async def run_case(self, case: MemoryReconciliationCase) -> EvalResult:
        artifact = await self._run_reconciliation(case)
        failures = _grade_case(
            artifact,
            expected=case.expected_for_mode(self.mode),
        )
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

    async def _run_reconciliation(
        self,
        case: MemoryReconciliationCase,
    ) -> dict[str, Any]:
        from agent.memory.models import ProceduralRule, SemanticFact
        from agent.memory.reconciliation import (
            plan_procedural_rule_write_llm_primary,
            plan_semantic_write_llm_primary,
        )
        from agent.memory.store import StoreRecord

        llm = self._llm_for_case(case)
        if hasattr(llm, "reset_structured_calls"):
            llm.reset_structured_calls()
        try:
            with _mute_expected_reconciliation_failure_logs(case):
                if case.layer == "semantic":
                    fact = SemanticFact.model_validate(case.semantic_fact)
                    records = [
                        StoreRecord(
                            namespace=("user-1", "semantic"),
                            key=str(record["id"]),
                            value=dict(record),
                        )
                        for record in case.existing_semantic_records
                    ]
                    plan = await plan_semantic_write_llm_primary(
                        fact,
                        records,
                        llm_client=llm,
                    )
                    return _semantic_artifact(plan, llm)
                if case.layer == "procedural":
                    rule = ProceduralRule.model_validate(case.procedural_rule)
                    existing_rules = [
                        ProceduralRule.model_validate(raw_rule)
                        for raw_rule in case.existing_procedural_rules
                    ]
                    plan = await plan_procedural_rule_write_llm_primary(
                        rule,
                        existing_rules,
                        llm_client=llm,
                    )
                    return _procedural_artifact(plan, llm)
                raise ValueError(f"Unsupported memory layer {case.layer!r}.")
        except Exception as exc:  # noqa: BLE001 - eval artifact records failures
            return {
                "layer": case.layer,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "structured_calls": dict(getattr(llm, "structured_calls", {})),
            }

    def _llm_for_case(self, case: MemoryReconciliationCase) -> Any:
        if self.mode == "live":
            if self._live_llm is None:
                from config import create_configured_control_llm_client

                self._live_llm = CountingReconciliationLLM(
                    create_configured_control_llm_client()
                )
            return self._live_llm
        return ScriptedReconciliationLLM(case.scripted)


@contextmanager
def _mute_expected_reconciliation_failure_logs(
    case: MemoryReconciliationCase,
) -> Any:
    if "exception_type" not in case.expected:
        yield
        return

    logger = logging.getLogger("agent.memory.reconciliation")
    previous_disabled = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = previous_disabled


def _semantic_artifact(plan: Any, llm: Any) -> dict[str, Any]:
    if plan.bump_record is not None:
        action = "bump"
    elif plan.supersede_records:
        action = "supersede"
    else:
        action = "coexist"
    return {
        "layer": "semantic",
        "action": action,
        "bump_key": plan.bump_record.key if plan.bump_record is not None else None,
        "supersede_keys": [record.key for record in plan.supersede_records],
        "structured_calls": dict(getattr(llm, "structured_calls", {})),
    }


def _procedural_artifact(plan: Any, llm: Any) -> dict[str, Any]:
    return {
        "layer": "procedural",
        "action": plan.action,
        "replace_indexes": list(plan.replace_indexes),
        "structured_calls": dict(getattr(llm, "structured_calls", {})),
    }


def _grade_case(
    artifact: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []

    for key in (
        "exception_type",
        "action",
        "bump_key",
        "supersede_keys",
        "replace_indexes",
    ):
        if key in expected and artifact.get(key) != expected[key]:
            failures.append(
                f"{key}: expected {expected[key]!r}, got {artifact.get(key)!r}"
            )

    calls = artifact.get("structured_calls")
    if isinstance(calls, Mapping):
        for schema, count in _optional_mapping(expected, "structured_calls").items():
            if calls.get(schema, 0) != count:
                failures.append(
                    f"structured_calls.{schema}: expected {count!r}, "
                    f"got {calls.get(schema, 0)!r}"
                )
    return failures


def _optional_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if item is None:
        return {}
    if not isinstance(item, Mapping):
        raise TypeError(f"{key} must be a mapping.")
    return item


def _mapping_sequence(value: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    item = value.get(key, [])
    if not isinstance(item, list):
        raise TypeError(f"{key} must be a list.")
    result: list[dict[str, Any]] = []
    for entry in item:
        if not isinstance(entry, Mapping):
            raise TypeError(f"{key} entries must be mappings.")
        result.append(dict(entry))
    return result


def _expected_by_mode(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = _optional_mapping(value, "expected_by_mode")
    expected_by_mode: dict[str, dict[str, Any]] = {}
    for mode, expected in raw.items():
        if not isinstance(expected, Mapping):
            raise TypeError("expected_by_mode values must be mappings.")
        expected_by_mode[str(mode)] = dict(expected)
    return expected_by_mode


def _build_evaluator(args: argparse.Namespace) -> MemoryReconciliationEvaluator:
    return MemoryReconciliationEvaluator(
        dataset_path=args.dataset or _DEFAULT_DATASET,
        mode=args.mode,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate memory reconciliation decisions.")
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="Use scripted reconciliation decisions or the configured control LLM.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    return run_evaluator_cli(_build_evaluator, parser=parser, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
