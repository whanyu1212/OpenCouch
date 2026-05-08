"""Evaluate grounded factual lookup and crisis-resource output quality."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.judges.rubric import RubricDimension, RubricJudgeArtifact, RubricLLMJudge
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

_DEFAULT_DATASET = _REPO_ROOT / "eval" / "datasets" / "tool_usage" / "quality_v1.json"
_VALID_KINDS = {"factual_lookup", "crisis_resources"}
_UNVERIFIED_MARKERS = (
    "could not verify",
    "could not be verified",
    "couldn't verify",
    "couldn't be verified",
    "cannot verify",
    "cannot be verified",
    "can't verify",
    "can't be verified",
    "unable to verify",
    "no verified",
    "not externally verifiable",
    "isn't something i can verify",
    "is not something i can verify",
    "not something i can verify",
)
_THERAPY_ADVICE_MARKERS = (
    "try a breathing exercise",
    "grounding exercise",
    "you should see a therapist",
    "diagnose",
    "take medication",
)
_BAD_CONTACT_MARKERS = (
    "unknown",
    "not available",
    "not found",
    "not provided",
    "see website",
    "n/a",
)


@dataclass(frozen=True)
class GroundedToolQualityCase:
    """Parsed grounded tool quality eval case."""

    id: str
    kind: str
    message: str
    query: str = ""
    description: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    scripted: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    rubric: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutput:
    """Normalized tool output for grading."""

    kind: str
    answer: str = ""
    status: str = ""
    location: str = ""
    resources: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the output.

        Returns:
            dict[str, Any]: JSON-compatible output payload.
        """

        payload: dict[str, Any] = {
            "kind": self.kind,
            "status": self.status,
        }
        if self.kind == "factual_lookup":
            payload["answer"] = self.answer
        if self.kind == "crisis_resources":
            payload["location"] = self.location
            payload["resources"] = self.resources
        return payload


class GroundedToolQualityEvaluator(BaseEvaluator[GroundedToolQualityCase]):
    """Run quality checks for grounded tool outputs."""

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        mode: str,
        judge_mode: str,
        min_judge_score: float | None,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"grounded_tool_quality_{mode}",
        )
        self.mode = mode
        self.judge_mode = judge_mode
        self.min_judge_score = min_judge_score

    def parse_case(self, raw_case: Any) -> GroundedToolQualityCase:
        """Parse one raw quality case.

        Args:
            raw_case (Any): Raw JSON case object.

        Returns:
            GroundedToolQualityCase: Parsed case.
        """

        if not isinstance(raw_case, Mapping):
            raise TypeError("Grounded tool quality cases must be JSON objects.")
        kind = str(raw_case["kind"])
        if kind not in _VALID_KINDS:
            raise ValueError(f"Unknown grounded tool quality kind={kind!r}.")
        return GroundedToolQualityCase(
            id=str(raw_case["id"]),
            kind=kind,
            description=str(raw_case.get("description", "")),
            message=str(raw_case["message"]),
            query=str(raw_case.get("query", "")),
            history=_list_of_mappings(raw_case.get("history", []), "history"),
            state=dict(_optional_mapping(raw_case, "state")),
            scripted=dict(_optional_mapping(raw_case, "scripted")),
            expected=dict(_optional_mapping(raw_case, "expected")),
            rubric=dict(_optional_mapping(raw_case, "rubric")),
        )

    def case_id(self, case: GroundedToolQualityCase, index: int) -> str:
        """Return the stable dataset id for one case.

        Args:
            case (GroundedToolQualityCase): Parsed case.
            index (int): Zero-based case index.

        Returns:
            str: Stable case id.
        """

        return case.id

    async def run_case(self, case: GroundedToolQualityCase) -> EvalResult:
        """Run and grade one grounded tool quality case.

        Args:
            case (GroundedToolQualityCase): Parsed case.

        Returns:
            EvalResult: Case result.
        """

        output = await _run_tool_case(case, mode=self.mode)
        hard_failures = _grade_output(case, output)
        judge_details: dict[str, Any] | None = None
        score = 1.0 if not hard_failures else 0.0
        failures = list(hard_failures)

        if self.judge_mode == "live":
            judge_outcome = await _judge_output(
                case,
                output,
                hard_failures=hard_failures,
                min_score=self._min_score_for_case(case),
            )
            judge_details = judge_outcome.to_dict()
            score = judge_outcome.score
            failures = judge_outcome.failures

        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=score,
            details={
                "description": case.description,
                "mode": self.mode,
                "judge_mode": self.judge_mode,
                "failures": failures,
                "output": output.to_dict(),
                "judge": judge_details,
            },
        )

    def _min_score_for_case(self, case: GroundedToolQualityCase) -> float:
        """Return the judge threshold for one case.

        Args:
            case (GroundedToolQualityCase): Parsed case.

        Returns:
            float: Minimum acceptable judge score.
        """

        if self.min_judge_score is not None:
            return self.min_judge_score
        return float(case.rubric.get("min_judge_score", 0.8))


async def _run_tool_case(case: GroundedToolQualityCase, *, mode: str) -> ToolOutput:
    if mode == "scripted":
        return _scripted_output(case)
    return await _live_output(case)


def _scripted_output(case: GroundedToolQualityCase) -> ToolOutput:
    if case.kind == "factual_lookup":
        return ToolOutput(
            kind=case.kind,
            answer=str(case.scripted.get("answer", "")),
            status=str(case.scripted.get("status", "")),
        )
    if case.kind == "crisis_resources":
        return ToolOutput(
            kind=case.kind,
            location=str(case.scripted.get("location", "")),
            resources=_resources(case.scripted.get("resources", [])),
            status=str(case.scripted.get("status", "")),
        )
    raise ValueError(f"Unknown grounded tool quality kind={case.kind!r}.")


async def _live_output(case: GroundedToolQualityCase) -> ToolOutput:
    from agent.tools.grounded_search import answer_factual_lookup, find_crisis_resources
    from config import create_configured_control_llm_client

    state = _build_state(case)
    llm_client = create_configured_control_llm_client()
    if case.kind == "factual_lookup":
        answer, status = await answer_factual_lookup(
            state,
            llm_client=llm_client,
            query=case.query or case.message,
        )
        return ToolOutput(kind=case.kind, answer=answer, status=status)
    if case.kind == "crisis_resources":
        location, resources, status = await find_crisis_resources(
            state,
            llm_client=llm_client,
        )
        return ToolOutput(
            kind=case.kind,
            location=location,
            resources=resources,
            status=status,
        )
    raise ValueError(f"Unknown grounded tool quality kind={case.kind!r}.")


def _build_state(case: GroundedToolQualityCase) -> dict[str, Any]:
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


def _grade_output(case: GroundedToolQualityCase, output: ToolOutput) -> list[str]:
    if case.kind == "factual_lookup":
        return _grade_factual(case, output)
    if case.kind == "crisis_resources":
        return _grade_crisis_resources(case, output)
    return [f"unknown case kind {case.kind!r}"]


def _grade_factual(
    case: GroundedToolQualityCase,
    output: ToolOutput,
) -> list[str]:
    expected = case.expected
    failures: list[str] = []
    _expect_equal(failures, "status", output.status, expected.get("status"))

    text = output.answer.strip()
    if not text:
        failures.append("answer is empty")
    _check_common_text_expectations(failures, text, expected)

    if expected.get("sources_required") and not _has_source_signal(text):
        failures.append("answer is missing source signal")
    if expected.get("unverified_statement_required") and not _has_unverified_marker(
        text
    ):
        failures.append("answer does not clearly state verification failed")
    if expected.get("forbid_therapy_advice"):
        found = [
            marker for marker in _THERAPY_ADVICE_MARKERS if marker in text.casefold()
        ]
        if found:
            failures.append(f"answer includes therapy-advice marker: {found}")
    return failures


def _grade_crisis_resources(
    case: GroundedToolQualityCase,
    output: ToolOutput,
) -> list[str]:
    expected = case.expected
    failures: list[str] = []
    _expect_equal(failures, "status", output.status, expected.get("status"))
    _expect_equal(failures, "location", output.location, expected.get("location"))

    min_resources = expected.get("min_resources")
    if min_resources is not None and len(output.resources) < int(min_resources):
        failures.append(
            f"resource count {len(output.resources)} below minimum {min_resources}"
        )
    max_resources = expected.get("max_resources")
    if max_resources is not None and len(output.resources) > int(max_resources):
        failures.append(
            f"resource count {len(output.resources)} exceeds maximum {max_resources}"
        )

    if expected.get("resources_actionable"):
        failures.extend(_resource_actionability_failures(output.resources))

    forbidden_contact_values = tuple(
        str(value).casefold()
        for value in expected.get("forbidden_contact_values", _BAD_CONTACT_MARKERS)
    )
    for index, resource in enumerate(output.resources, start=1):
        for field_name in ("phone", "url", "website"):
            value = str(resource.get(field_name, "")).casefold()
            if any(marker in value for marker in forbidden_contact_values):
                failures.append(
                    f"resource {index} {field_name} has non-actionable value"
                )

    for needle in expected.get("resource_name_contains", []):
        needle_text = str(needle).casefold()
        if not any(
            needle_text in resource.get("name", "").casefold()
            for resource in output.resources
        ):
            failures.append(f"no resource name contains {needle!r}")
    return failures


def _check_common_text_expectations(
    failures: list[str],
    text: str,
    expected: Mapping[str, Any],
) -> None:
    max_words = expected.get("max_words")
    if max_words is not None and _word_count(text) > int(max_words):
        failures.append(f"word count {_word_count(text)} exceeds {max_words}")

    for phrase in expected.get("required_phrases", []):
        phrase_text = str(phrase)
        if phrase_text.casefold() not in text.casefold():
            failures.append(f"answer missing required phrase {phrase_text!r}")

    for group in expected.get("required_any_phrase_groups", []):
        if not isinstance(group, list):
            raise TypeError("required_any_phrase_groups entries must be lists.")
        phrases = [str(phrase) for phrase in group]
        if not any(phrase.casefold() in text.casefold() for phrase in phrases):
            failures.append(f"answer missing one of required phrases {phrases!r}")

    for phrase in expected.get("forbidden_phrases", []):
        phrase_text = str(phrase)
        if phrase_text.casefold() in text.casefold():
            failures.append(f"answer contains forbidden phrase {phrase_text!r}")


def _resource_actionability_failures(resources: list[dict[str, str]]) -> list[str]:
    failures: list[str] = []
    for index, resource in enumerate(resources, start=1):
        for field_name in ("name", "phone", "region"):
            if not str(resource.get(field_name, "")).strip():
                failures.append(f"resource {index} missing {field_name}")
        url = str(resource.get("url") or resource.get("website") or "")
        if not url:
            failures.append(f"resource {index} missing url")
        elif not url.startswith(("http://", "https://")):
            failures.append(f"resource {index} url is not a URL")
    return failures


async def _judge_output(
    case: GroundedToolQualityCase,
    output: ToolOutput,
    *,
    hard_failures: list[str],
    min_score: float,
) -> Any:
    from config import create_configured_control_llm_client

    judge = RubricLLMJudge(llm_client=create_configured_control_llm_client())
    judged_output = output.to_dict()
    judged_output.pop("kind", None)
    verdict = await judge.judge(
        RubricJudgeArtifact(
            task=_judge_task(case),
            input={
                "case_id": case.id,
                "description": case.description,
                "message": case.message,
                "query": case.query,
                "expected": case.expected,
            },
            output=judged_output,
            rubric=_rubric_dimensions(case),
            hard_failures=hard_failures,
        )
    )
    return judge.combine(
        verdict=verdict,
        hard_failures=hard_failures,
        min_score=min_score,
    )


def _rubric_dimensions(case: GroundedToolQualityCase) -> list[RubricDimension]:
    raw_dimensions = case.rubric.get("dimensions")
    if isinstance(raw_dimensions, list) and raw_dimensions:
        return [RubricDimension.model_validate(item) for item in raw_dimensions]
    if case.kind == "factual_lookup":
        return [
            RubricDimension(
                name="factual_focus",
                question="Does the answer directly answer the factual request only?",
            ),
            RubricDimension(
                name="source_quality",
                question="Does the answer cite reputable sources or clearly say it cannot verify?",
            ),
        ]
    return [
        RubricDimension(
            name="actionability",
            question="Are crisis resources location-appropriate and actionable?",
        ),
        RubricDimension(
            name="no_fabrication",
            question="Does the output avoid invented or vague contact details?",
        ),
    ]


def _judge_task(case: GroundedToolQualityCase) -> str:
    if case.kind == "crisis_resources":
        return (
            "Judge structured crisis-resource lookup output quality. The output "
            "is expected to contain resource rows, not a prose answer."
        )
    return (
        "Judge grounded factual lookup output quality. The kind field names the "
        "tool under evaluation; it is not itself a classification verdict."
    )


def _has_source_signal(text: str) -> bool:
    return "sources:" in text.casefold() or "http://" in text or "https://" in text


def _has_unverified_marker(text: str) -> bool:
    lowered = _normalize_marker_text(text)
    return any(marker in lowered for marker in _UNVERIFIED_MARKERS)


def _normalize_marker_text(value: str) -> str:
    return value.casefold().replace("’", "'").replace("‘", "'").replace("`", "'")


def _word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def _expect_equal(
    failures: list[str],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    if expected is None:
        return
    if actual != expected:
        failures.append(f"{name}: expected {expected!r}, got {actual!r}")


def _optional_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be an object.")
    return value


def _list_of_mappings(value: Any, field_name: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list.")
    items: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name} entries must be objects.")
        items.append({str(key): str(val) for key, val in item.items()})
    return items


def _resources(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise TypeError("resources must be a list.")
    resources: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("resource entries must be objects.")
        resources.append({str(key): str(val) for key, val in item.items()})
    return resources


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate grounded tool output quality.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="scripted grades fixture outputs; live calls configured provider-backed tools.",
    )
    parser.add_argument(
        "--judge-mode",
        choices=("off", "live"),
        default="off",
        help="off uses hard checks only; live adds configured LLM-as-judge.",
    )
    parser.add_argument(
        "--min-judge-score",
        type=float,
        default=None,
        help="Override per-case minimum judge score.",
    )
    return parser


def main() -> int:
    """Run the grounded tool quality evaluator CLI.

    Returns:
        int: Shell exit code.
    """

    return run_evaluator_cli(
        lambda args: GroundedToolQualityEvaluator(
            dataset_path=args.dataset,
            mode=args.mode,
            judge_mode=args.judge_mode,
            min_judge_score=args.min_judge_score,
        ),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
