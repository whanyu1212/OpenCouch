"""Evaluate semantic and procedural memory write-policy decisions."""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from collections.abc import AsyncIterator, Mapping, Sequence
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

_DEFAULT_DATASET = _REPO_ROOT / "eval" / "datasets" / "memory" / "write_policy_v1.json"


@dataclass(frozen=True)
class MemoryWritePolicyCase:
    """Parsed memory write-policy case."""

    id: str
    layer: str
    message: str
    description: str = ""
    semantic_write: dict[str, Any] = field(default_factory=dict)
    procedural_rule: dict[str, Any] = field(default_factory=dict)
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


class ScriptedPolicyLLM:
    """Scripted structured-output client for write-policy evals."""

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
            raise RuntimeError("scripted policy LLM failure")
        schema_name = response_schema.__name__
        self.structured_calls[schema_name] = (
            self.structured_calls.get(schema_name, 0) + 1
        )
        decision = self.scripted.get("decision")
        if not isinstance(decision, Mapping):
            raise RuntimeError("Policy eval case needs scripted.decision.")
        return response_schema(**decision)


class MemoryWritePolicyEvaluator(BaseEvaluator[MemoryWritePolicyCase]):
    """Run direct write-policy decision checks."""

    def __init__(self, *, dataset_path: str | Path, mode: str) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"memory_write_policy_{mode}",
        )
        self.mode = mode
        self._live_llm: Any | None = None

    def parse_case(self, raw_case: Any) -> MemoryWritePolicyCase:
        if not isinstance(raw_case, Mapping):
            raise TypeError("Memory write-policy eval cases must be JSON objects.")
        return MemoryWritePolicyCase(
            id=str(raw_case["id"]),
            layer=str(raw_case["layer"]),
            message=str(raw_case["message"]),
            description=str(raw_case.get("description", "")),
            semantic_write=dict(_optional_mapping(raw_case, "semantic_write")),
            procedural_rule=dict(_optional_mapping(raw_case, "procedural_rule")),
            scripted=dict(_optional_mapping(raw_case, "scripted")),
            expected=dict(_optional_mapping(raw_case, "expected")),
        )

    def case_id(self, case: MemoryWritePolicyCase, index: int) -> str:
        return case.id

    async def run_case(self, case: MemoryWritePolicyCase) -> EvalResult:
        artifact = await self._run_policy(case)
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

    async def _run_policy(self, case: MemoryWritePolicyCase) -> dict[str, Any]:
        from agent.memory.models import MemoryWrite, ProceduralRuleDraft
        from agent.memory.policy.candidates import (
            build_procedural_candidate,
            build_semantic_candidate,
        )
        from agent.memory.policy.write import (
            decide_procedural_candidate_llm_primary,
            decide_semantic_candidate_llm_primary,
        )

        llm = self._llm_for_case(case)
        try:
            with _mute_expected_policy_failure_logs(case):
                if case.layer == "semantic":
                    write = MemoryWrite.model_validate(case.semantic_write)
                    candidate = build_semantic_candidate(write, message=case.message)
                    decision = await decide_semantic_candidate_llm_primary(
                        candidate,
                        llm_client=llm,
                    )
                elif case.layer == "procedural":
                    draft = ProceduralRuleDraft.model_validate(case.procedural_rule)
                    candidate = build_procedural_candidate(
                        draft,
                        message=case.message,
                        session_id="eval-session",
                        turn_index=0,
                    )
                    decision = await decide_procedural_candidate_llm_primary(
                        candidate,
                        llm_client=llm,
                    )
                else:
                    raise ValueError(f"Unsupported memory layer {case.layer!r}.")
        except Exception as exc:  # noqa: BLE001 - eval artifact records failures
            return {
                "layer": case.layer,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "structured_calls": getattr(llm, "structured_calls", {}),
            }

        return {
            "layer": case.layer,
            "candidate": {
                "policy_recommendation": candidate.policy_recommendation,
                "scope": candidate.scope,
                "durability": candidate.durability,
                "sensitivity": candidate.sensitivity,
                "explicitness": candidate.explicitness,
            },
            "decision": decision.model_dump(mode="json"),
            "structured_calls": getattr(llm, "structured_calls", {}),
        }

    def _llm_for_case(self, case: MemoryWritePolicyCase) -> Any:
        if self.mode == "live":
            if self._live_llm is None:
                from config import create_configured_control_llm_client

                self._live_llm = create_configured_control_llm_client()
            return self._live_llm
        return ScriptedPolicyLLM(case.scripted)


@contextmanager
def _mute_expected_policy_failure_logs(case: MemoryWritePolicyCase) -> Any:
    if "exception_type" not in case.expected:
        yield
        return

    logger = logging.getLogger("agent.memory.policy.write")
    previous_disabled = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = previous_disabled


def _grade_case(
    case: MemoryWritePolicyCase,
    artifact: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    expected = case.expected
    decision = artifact.get("decision")

    for key in ("exception_type",):
        if key in expected and artifact.get(key) != expected[key]:
            failures.append(
                f"{key}: expected {expected[key]!r}, got {artifact.get(key)!r}"
            )

    if "action" in expected:
        if not isinstance(decision, Mapping):
            failures.append("decision missing")
        elif decision.get("action") != expected["action"]:
            failures.append(
                f"action: expected {expected['action']!r}, got {decision.get('action')!r}"
            )

    if "policy_version" in expected:
        if not isinstance(decision, Mapping):
            failures.append("decision missing")
        elif decision.get("policy_version") != expected["policy_version"]:
            failures.append(
                "policy_version: expected "
                f"{expected['policy_version']!r}, got {decision.get('policy_version')!r}"
            )

    if "reason_contains" in expected:
        reason = (
            str(decision.get("reason", "")) if isinstance(decision, Mapping) else ""
        )
        for phrase in expected["reason_contains"]:
            if str(phrase) not in reason:
                failures.append(f"reason missing {str(phrase)!r}")

    calls = artifact.get("structured_calls")
    if isinstance(calls, Mapping):
        for schema, count in _optional_mapping(expected, "structured_calls").items():
            if calls.get(schema, 0) != count:
                failures.append(
                    f"structured_calls.{schema}: expected {count!r}, "
                    f"got {calls.get(schema, 0)!r}"
                )

    candidate = artifact.get("candidate")
    if isinstance(candidate, Mapping):
        for key, value in _optional_mapping(expected, "candidate").items():
            if candidate.get(key) != value:
                failures.append(
                    f"candidate.{key}: expected {value!r}, got {candidate.get(key)!r}"
                )
    return failures


def _optional_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    item = value.get(key)
    if item is None:
        return {}
    if not isinstance(item, Mapping):
        raise TypeError(f"{key} must be a mapping.")
    return item


def _build_evaluator(args: argparse.Namespace) -> MemoryWritePolicyEvaluator:
    return MemoryWritePolicyEvaluator(
        dataset_path=args.dataset or _DEFAULT_DATASET,
        mode=args.mode,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate memory write-policy decisions.")
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="Use scripted policy decisions or the configured control LLM.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    return run_evaluator_cli(_build_evaluator, parser=parser, argv=argv)


if __name__ == "__main__":
    raise SystemExit(main())
