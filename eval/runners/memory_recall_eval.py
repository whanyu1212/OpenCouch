"""Evaluate direct memory recall contracts."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
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
from eval.runners.memory_control_common import seed_memory_store

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = _REPO_ROOT / "apps" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_DEFAULT_DATASET = (
    _REPO_ROOT / "eval" / "datasets" / "memory" / "recall_contract_v1.json"
)


@dataclass(frozen=True)
class MemoryRecallCase:
    """Parsed memory recall contract case."""

    id: str
    query: str
    description: str = ""
    owner_id: str = "eval-user"
    is_first_turn: bool = False
    embedding_provider: str = "none"
    memory_seed: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


class FailingEmbeddingProvider:
    """Embedding provider that forces token-recall fallback."""

    model_name = "eval-failing-embedding-provider"

    async def aembed(
        self,
        texts: Sequence[str],  # noqa: ARG002 - embedding protocol
        *,
        task_type: str,  # noqa: ARG002 - embedding protocol
    ) -> list[list[float] | None]:
        """Raise an embedding failure for fallback-path evals.

        Args:
            texts (Sequence[str]): Text inputs to embed.
            task_type (str): Embedding task type.

        Returns:
            list[list[float] | None]: Never returned.

        Raises:
            RuntimeError: Always raised to exercise fallback behavior.
        """

        raise RuntimeError("scripted embedding provider failure")


class MemoryRecallEvaluator(BaseEvaluator[MemoryRecallCase]):
    """Run direct memory recall contract checks."""

    def parse_case(self, raw_case: Any) -> MemoryRecallCase:
        if not isinstance(raw_case, Mapping):
            raise TypeError("Memory recall eval cases must be JSON objects.")
        return MemoryRecallCase(
            id=str(raw_case["id"]),
            query=str(raw_case["query"]),
            description=str(raw_case.get("description", "")),
            owner_id=str(raw_case.get("owner_id", "eval-user")),
            is_first_turn=bool(raw_case.get("is_first_turn", False)),
            embedding_provider=str(raw_case.get("embedding_provider", "none")),
            memory_seed=dict(_optional_mapping(raw_case, "memory_seed")),
            expected=dict(_optional_mapping(raw_case, "expected")),
        )

    def case_id(self, case: MemoryRecallCase, index: int) -> str:
        return case.id

    async def run_case(self, case: MemoryRecallCase) -> EvalResult:
        artifact = await _run_recall(case)
        failures = _grade_case(artifact, expected=case.expected)
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


async def _run_recall(case: MemoryRecallCase) -> dict[str, Any]:
    from agent.memory.entries import format_working_memory_entries
    from agent.memory.recall import load_memory_for_turn
    from agent.memory.store import OpenCouchMemoryStore

    store = OpenCouchMemoryStore()
    await seed_memory_store(store, owner_id=case.owner_id, seed=case.memory_seed)
    provider = _embedding_provider(case.embedding_provider)
    with _mute_expected_embedding_failure_logs(case):
        result = await load_memory_for_turn(
            memory_store=store,
            embedding_provider=provider,
            owner_id=case.owner_id,
            query=case.query,
            is_first_turn=case.is_first_turn,
        )
    try:
        return {
            "query": case.query,
            "is_first_turn": case.is_first_turn,
            "working_memory": result.working_memory,
            "working_memory_rendered": format_working_memory_entries(
                result.working_memory
            ),
            "summary": result.summary,
            "procedural_rules": result.procedural_rules,
            "proactive_recall_enabled": result.proactive_recall_enabled,
            "diagnostics": result.diagnostics,
        }
    finally:
        await store.aclose()


def _embedding_provider(kind: str) -> Any | None:
    if kind == "none":
        return None
    if kind == "failing":
        return FailingEmbeddingProvider()
    raise ValueError(f"Unsupported embedding_provider {kind!r}.")


@contextmanager
def _mute_expected_embedding_failure_logs(case: MemoryRecallCase) -> Any:
    if case.embedding_provider != "failing":
        yield
        return

    logger = logging.getLogger("agent.memory.recall")
    previous_disabled = logger.disabled
    logger.disabled = True
    try:
        yield
    finally:
        logger.disabled = previous_disabled


def _grade_case(
    artifact: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    working_memory = artifact.get("working_memory")
    procedural_rules = artifact.get("procedural_rules")
    diagnostics = _mapping_or_empty(artifact.get("diagnostics"))

    _expect_equal(
        failures,
        "working_memory_count",
        len(working_memory) if isinstance(working_memory, list) else None,
        expected,
    )
    _grade_minimum(
        failures,
        label="working_memory_count",
        actual=len(working_memory) if isinstance(working_memory, list) else None,
        expected=expected.get("working_memory_count_min"),
    )
    _grade_text_collection(
        failures,
        label="working_memory",
        values=[
            *(_flatten_text(working_memory) if working_memory is not None else []),
            *[
                str(item)
                for item in _list_or_empty(artifact.get("working_memory_rendered"))
            ],
        ],
        contains=expected.get("working_memory_contains"),
        absent=expected.get("working_memory_not_contains"),
    )
    _grade_text_collection(
        failures,
        label="procedural_rules",
        values=[str(item) for item in _list_or_empty(procedural_rules)],
        contains=expected.get("procedural_rules_contain"),
        absent=expected.get("procedural_rules_not_contains"),
    )
    _grade_text_collection(
        failures,
        label="summary",
        values=[str(artifact.get("summary", ""))],
        contains=expected.get("summary_contains"),
        absent=expected.get("summary_not_contains"),
    )
    _expect_equal(
        failures,
        "proactive_recall_enabled",
        artifact.get("proactive_recall_enabled"),
        expected,
    )
    for key in (
        "retrieval_path",
        "semantic_hits",
        "semantic_store_size",
        "episodic_hits",
        "episodic_store_size",
        "procedural_count",
        "proactive_recall",
    ):
        _expect_equal(failures, key, diagnostics.get(key), expected)
    _grade_minimum(
        failures,
        label="semantic_hits",
        actual=diagnostics.get("semantic_hits"),
        expected=expected.get("semantic_hits_min"),
    )
    _grade_minimum(
        failures,
        label="episodic_hits",
        actual=diagnostics.get("episodic_hits"),
        expected=expected.get("episodic_hits_min"),
    )
    return failures


def _expect_equal(
    failures: list[str],
    key: str,
    actual: Any,
    expected: Mapping[str, Any],
) -> None:
    if key in expected and actual != expected[key]:
        failures.append(f"{key}: expected {expected[key]!r}, got {actual!r}")


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


def _grade_text_collection(
    failures: list[str],
    *,
    label: str,
    values: Sequence[str],
    contains: Any,
    absent: Any,
) -> None:
    haystack = "\n".join(values).casefold()
    for phrase in _as_list(contains):
        if str(phrase).casefold() not in haystack:
            failures.append(f"{label} missing {str(phrase)!r}")
    for phrase in _as_list(absent):
        if str(phrase).casefold() in haystack:
            failures.append(f"{label} contains forbidden {str(phrase)!r}")


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        texts: list[str] = []
        for item in value.values():
            texts.extend(_flatten_text(item))
        return texts
    if isinstance(value, list | tuple):
        texts: list[str] = []
        for item in value:
            texts.extend(_flatten_text(item))
        return texts
    return [str(value)]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for this evaluator.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """

    parser = build_base_arg_parser(__doc__ or "")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the memory recall evaluator CLI.

    Args:
        argv (Sequence[str] | None): Optional CLI arguments.

    Returns:
        int: Process exit code.
    """

    parser = build_parser()
    return run_evaluator_cli(
        lambda args: MemoryRecallEvaluator(dataset_path=args.dataset),
        parser=parser,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
