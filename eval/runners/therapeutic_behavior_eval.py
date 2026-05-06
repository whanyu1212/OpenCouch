"""Runner for end-to-end therapeutic behavior evaluation.

This eval grades both routing outcomes (response_style / approach) and
lightweight response-text behavior assertions on curated scenarios.

Usage:
    python eval/runners/therapeutic_behavior_eval.py --mode deterministic
    python eval/runners/therapeutic_behavior_eval.py --mode hybrid
    python eval/runners/therapeutic_behavior_eval.py --mode auto
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

BACKEND_ROOT = Path(__file__).resolve().parents[2] / "apps" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agent.memory.modes import MemoryMode
from agent.graph import run_agent
from agent.models import AgentInput, AgentOutput, Message
from agent.persistence import PersistentAgentRuntime
from config import create_configured_llm_client
from llm.base import BaseLLMClient

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "therapeutic_behavior_v1_1.json"
)

EvalMode = Literal["auto", "deterministic", "hybrid"]


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        The configured argument parser for the eval runner.
    """

    parser = argparse.ArgumentParser(
        description="Run therapeutic end-to-end behavior evaluation."
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "deterministic", "hybrid"],
        default="auto",
        help=(
            "Evaluation mode. 'auto' uses a configured LLM client when available "
            "and falls back to deterministic mode otherwise."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help=f"Dataset JSON path. Default: {DATASET_PATH}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional case limit (from top of dataset) for quick smoke runs.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional single case id to run.",
    )
    return parser


def _load_cases(path: Path) -> list[dict[str, Any]]:
    """Load eval cases from disk.

    Args:
        path: The dataset JSON file to read.

    Returns:
        The decoded list of case dictionaries.
    """

    return json.loads(path.read_text())


def _resolve_llm_client(mode: EvalMode) -> tuple[BaseLLMClient | None, str]:
    """Resolve the live client for the requested mode.

    Args:
        mode: The requested eval mode.

    Returns:
        A ``(client, resolved_mode)`` tuple.
    """

    if mode == "deterministic":
        return None, "deterministic"

    if mode == "hybrid":
        return create_configured_llm_client(), "hybrid"

    try:
        return create_configured_llm_client(), "hybrid"
    except Exception:
        return None, "deterministic"


def _build_agent_input(case: dict[str, Any]) -> AgentInput:
    """Build one-shot agent input from a case record.

    Args:
        case: The eval case configuration.

    Returns:
        The ``AgentInput`` for a one-shot run.
    """

    history = [Message(**message) for message in case.get("history", [])]
    working_memory = case.get("working_memory", [])
    installed_skills = case.get("installed_skills", [])

    return AgentInput(
        message=case["message"],
        history=history,
        working_memory=working_memory,
        installed_skills=installed_skills,
        user_id=case.get("user_id"),
        session_id=case.get("session_id"),
    )


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def _matching_terms(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def _sentence_count(text: str) -> int:
    parts = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    return len(parts)


def _question_mark_count(text: str) -> int:
    return text.count("?")


def _first_sentence(text: str) -> str:
    match = re.search(r"(.+?[.!?])(?:\s|$)", text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def _first_match_index(text: str, terms: list[str]) -> int | None:
    lowered = text.lower()
    matches = [
        lowered.find(term.lower())
        for term in terms
        if term and lowered.find(term.lower()) != -1
    ]
    if not matches:
        return None
    return min(matches)


def _evaluate_assertions(
    response_text: str,
    assertions: dict[str, Any] | None,
) -> list[str]:
    """Evaluate text assertions against one response.

    Args:
        response_text: The assistant response text to grade.
        assertions: The assertion block from the dataset.

    Returns:
        A list of failure messages. Empty when all assertions pass.
    """

    if not assertions:
        return []

    failures: list[str] = []

    must_include_any = assertions.get("must_include_any") or []
    if must_include_any and not _contains_any(response_text, must_include_any):
        failures.append(f"missing required terms: one of {must_include_any!r}")

    must_not_include_any = assertions.get("must_not_include_any") or []
    found_forbidden = _matching_terms(response_text, must_not_include_any)
    if found_forbidden:
        failures.append(f"contains forbidden terms: {found_forbidden!r}")

    require_question = bool(assertions.get("require_question", False))
    if require_question and "?" not in response_text:
        failures.append("expected a question mark in response")

    max_question_marks = assertions.get("max_question_marks")
    question_marks = _question_mark_count(response_text)
    if isinstance(max_question_marks, int) and question_marks > max_question_marks:
        failures.append(
            f"response has too many question marks: {question_marks} (max {max_question_marks})"
        )

    sentence_count = _sentence_count(response_text)
    max_sentences = assertions.get("max_sentences")
    if isinstance(max_sentences, int) and sentence_count > max_sentences:
        failures.append(
            f"response too long: {sentence_count} sentences (max {max_sentences})"
        )

    min_sentences = assertions.get("min_sentences")
    if isinstance(min_sentences, int) and sentence_count < min_sentences:
        failures.append(
            f"response too short: {sentence_count} sentences (min {min_sentences})"
        )

    first_sentence_must_include_any = assertions.get("first_sentence_must_include_any")
    if first_sentence_must_include_any:
        first_sentence = _first_sentence(response_text)
        if not _contains_any(first_sentence, first_sentence_must_include_any):
            failures.append(
                "first sentence missing required terms: "
                f"one of {first_sentence_must_include_any!r}"
            )

    must_include_any_before_question = (
        assertions.get("must_include_any_before_question") or []
    )
    if must_include_any_before_question:
        question_index = response_text.find("?")
        if question_index == -1:
            failures.append("expected a question mark so ordering could be checked")
        else:
            before_question = response_text[:question_index]
            if not _contains_any(before_question, must_include_any_before_question):
                failures.append(
                    "missing required terms before first question: "
                    f"one of {must_include_any_before_question!r}"
                )

    must_include_any_before_any_of = assertions.get("must_include_any_before_any_of")
    if must_include_any_before_any_of:
        required = must_include_any_before_any_of.get("required") or []
        anchors = must_include_any_before_any_of.get("before") or []
        required_index = _first_match_index(response_text, required)
        anchor_index = _first_match_index(response_text, anchors)
        if required_index is None:
            failures.append(f"missing required terms: one of {required!r}")
        elif anchor_index is None:
            failures.append(f"missing anchor terms: one of {anchors!r}")
        elif required_index >= anchor_index:
            failures.append(
                "required terms did not appear before anchor terms: "
                f"required={required!r} before={anchors!r}"
            )

    return failures


def _evaluate_output(case: dict[str, Any], output: AgentOutput) -> list[str]:
    """Evaluate output-level expectations for one case.

    Args:
        case: The case or turn expectation block.
        output: The normalized agent output.

    Returns:
        A list of failure messages. Empty when the output passes.
    """

    failures: list[str] = []

    expected_style = case.get("expected_response_style") or case.get("expected_mode")
    if expected_style and output.response_style != expected_style:
        failures.append(
            f"response_style mismatch: got {output.response_style!r}, expected {expected_style!r}"
        )

    expected_approach = case.get("expected_therapeutic_approach")
    if (
        expected_approach is not None
        and output.therapeutic_approach != expected_approach
    ):
        failures.append(
            "therapeutic_approach mismatch: "
            f"got {output.therapeutic_approach!r}, expected {expected_approach!r}"
        )

    text_failures = _evaluate_assertions(output.response_text, case.get("assertions"))
    failures.extend(text_failures)

    return failures


def _evaluate_state(case: dict[str, Any], state: dict[str, Any] | None) -> list[str]:
    """Evaluate persisted state expectations for one case.

    Args:
        case: The case or turn expectation block.
        state: The final graph state for the turn.

    Returns:
        A list of failure messages. Empty when the state passes.
    """

    if not state:
        return []

    failures: list[str] = []
    exercise_state = state.get("exercise_state", {}) or {}
    exercise_active = (
        exercise_state.get("exercise_type") is not None
        and exercise_state.get("exercise_step") is not None
    )

    expected_exercise_active = case.get("expected_exercise_active")
    if (
        expected_exercise_active is not None
        and exercise_active != expected_exercise_active
    ):
        failures.append(
            "exercise_active mismatch: "
            f"got {exercise_active!r}, expected {expected_exercise_active!r}"
        )

    if "expected_exercise_type" in case:
        actual_type = exercise_state.get("exercise_type")
        if actual_type != case["expected_exercise_type"]:
            failures.append(
                "exercise_type mismatch: "
                f"got {actual_type!r}, expected {case['expected_exercise_type']!r}"
            )

    if "expected_exercise_step" in case:
        actual_step = exercise_state.get("exercise_step")
        if actual_step != case["expected_exercise_step"]:
            failures.append(
                "exercise_step mismatch: "
                f"got {actual_step!r}, expected {case['expected_exercise_step']!r}"
            )

    expected_exercise_step_in = case.get("expected_exercise_step_in")
    if (
        expected_exercise_step_in is not None
        and exercise_state.get("exercise_step") not in expected_exercise_step_in
    ):
        failures.append(
            "exercise_step mismatch: "
            f"got {exercise_state.get('exercise_step')!r}, expected one of "
            f"{expected_exercise_step_in!r}"
        )

    return failures


async def _evaluate_case(
    case: dict[str, Any],
    *,
    llm_client: BaseLLMClient | None,
) -> tuple[bool, list[str]]:
    """Evaluate one case in one-shot or multi-turn mode.

    Args:
        case: The case configuration to run.
        llm_client: The live client to use, or ``None`` for deterministic mode.

    Returns:
        A ``(passed, details)`` tuple. ``details`` is empty on success.
    """

    if case.get("turns"):
        return await _evaluate_multi_turn_case(case, llm_client=llm_client)

    output = await run_agent(_build_agent_input(case), llm_client=llm_client)
    failures = _evaluate_output(case, output)
    if not failures:
        return True, []

    failure_text = "; ".join(failures)
    detail = (
        f"FAIL [{case.get('dispatch_tier', '?')}] {case['id']}: {failure_text}. "
        f"message={case['message']!r} "
        f"style={output.response_style!r} approach={output.therapeutic_approach!r} "
        f"response={output.response_text!r}"
    )
    return False, [detail]


async def _evaluate_multi_turn_case(
    case: dict[str, Any],
    *,
    llm_client: BaseLLMClient | None,
) -> tuple[bool, list[str]]:
    """Evaluate one persistent multi-turn case.

    Args:
        case: The multi-turn case configuration.
        llm_client: The live client to use, or ``None`` for deterministic mode.

    Returns:
        A ``(passed, details)`` tuple. ``details`` is empty on success.
    """

    failures: list[str] = []
    case_user_id = case.get("user_id") or f"eval-user-{case['id']}"
    thread_id = case.get("thread_id") or f"therapeutic-behavior-{uuid4().hex}"
    installed_skills = list(case.get("installed_skills", []))

    async with PersistentAgentRuntime(
        memory_mode=MemoryMode.INCOGNITO,
        finalize_active_sessions_on_close=False,
    ) as runtime:
        for turn_index, turn in enumerate(case["turns"], start=1):
            result = await runtime.run_turn(
                thread_id=thread_id,
                message=turn["message"],
                user_id=turn.get("user_id") or case_user_id,
                installed_skills=list(turn.get("installed_skills", installed_skills)),
                llm_client=llm_client,
            )

            turn_failures = _evaluate_output(turn, result.output)
            turn_failures.extend(_evaluate_state(turn, result.state))
            if not turn_failures:
                continue

            exercise_state = (result.state or {}).get("exercise_state", {}) or {}
            failures.append(
                f"FAIL [{case.get('dispatch_tier', '?')}] {case['id']} turn {turn_index}: "
                f"{'; '.join(turn_failures)}. "
                f"message={turn['message']!r} "
                f"style={result.output.response_style!r} "
                f"approach={result.output.therapeutic_approach!r} "
                f"exercise_type={exercise_state.get('exercise_type')!r} "
                f"exercise_step={exercise_state.get('exercise_step')!r} "
                f"response={result.output.response_text!r}"
            )

    return not failures, failures


def _select_cases(
    cases: list[dict[str, Any]],
    *,
    limit: int | None,
    case_id: str | None,
) -> list[dict[str, Any]]:
    """Filter the loaded cases for one run.

    Args:
        cases: The full loaded dataset.
        limit: Optional prefix limit.
        case_id: Optional case identifier to isolate.

    Returns:
        The selected cases in run order.
    """

    selected = cases
    if case_id:
        selected = [case for case in selected if case.get("id") == case_id]
    if limit is not None:
        selected = selected[:limit]
    return selected


async def _run(
    mode: EvalMode,
    dataset_path: Path,
    *,
    limit: int | None = None,
    case_id: str | None = None,
) -> int:
    """Run the therapeutic behavior eval.

    Args:
        mode: The requested eval mode.
        dataset_path: The dataset file to execute.
        limit: Optional prefix limit for quick runs.
        case_id: Optional case identifier to isolate.

    Returns:
        A process-style exit code.
    """

    cases = _select_cases(_load_cases(dataset_path), limit=limit, case_id=case_id)
    llm_client, resolved_mode = _resolve_llm_client(mode)

    print(
        f"Running therapeutic behavior eval in {resolved_mode} mode "
        f"on {len(cases)} case(s) from {dataset_path.name}."
    )
    print()

    by_tier: dict[str, dict[str, int]] = {
        "regex": {"total": 0, "passed": 0},
        "llm": {"total": 0, "passed": 0},
    }
    failures: list[str] = []

    for case in cases:
        tier = case.get("dispatch_tier", "regex")
        if tier not in by_tier:
            by_tier[tier] = {"total": 0, "passed": 0}
        by_tier[tier]["total"] += 1

        passed, details = await _evaluate_case(case, llm_client=llm_client)
        if passed:
            by_tier[tier]["passed"] += 1
        else:
            failures.extend(details)

    for tier_name, counts in sorted(by_tier.items()):
        if counts["total"] == 0:
            continue
        print(f"  {tier_name:10s} {counts['passed']:2d}/{counts['total']:2d} passed")

    overall_total = sum(counts["total"] for counts in by_tier.values())
    overall_passed = sum(counts["passed"] for counts in by_tier.values())
    print()
    print(f"Overall: {overall_passed}/{overall_total} passed")

    if failures:
        print()
        print("Failures:")
        for detail in failures:
            print(f"  {detail}")

    if resolved_mode == "deterministic":
        regex_counts = by_tier.get("regex", {"total": 0, "passed": 0})
        regex_failed = regex_counts["total"] - regex_counts["passed"]
        if regex_failed > 0:
            print()
            print(
                f"{regex_failed} regex-tier failure(s) — these should pass "
                "regardless of LLM availability."
            )
            return 1

        llm_counts = by_tier.get("llm", {"total": 0, "passed": 0})
        if llm_counts["total"] > 0:
            print()
            print(
                f"Note: {llm_counts['total'] - llm_counts['passed']} llm-tier case(s) "
                "did not pass in deterministic mode. This is expected — those "
                "cases require the LLM classifier. Run with --mode hybrid to "
                "grade them."
            )
        return 0

    if overall_passed < overall_total:
        return 1
    return 0


def main() -> int:
    """Run the eval runner CLI entrypoint.

    Returns:
        A process-style exit code.
    """

    args = _build_parser().parse_args()
    return asyncio.run(
        _run(
            args.mode,
            args.dataset,
            limit=args.limit,
            case_id=args.case,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
