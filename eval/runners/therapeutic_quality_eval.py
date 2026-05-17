"""Evaluate therapeutic response quality with hard rubrics."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eval.runners.base import (
    BaseEvaluator,
    EvalResult,
    build_base_arg_parser,
    run_evaluator_cli,
)
from eval.runners.therapeutic_common import (
    TherapeuticEvalCase,
    build_live_therapeutic_llms,
    build_scripted_llm,
    grade_therapeutic_output,
    invoke_therapeutic_branch,
    parse_therapeutic_case,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "eval" / "datasets" / "therapeutic" / "quality_v1.json"


class TherapeuticQualityEvaluator(BaseEvaluator[TherapeuticEvalCase]):
    """Run therapeutic output quality checks against scripted or live LLMs."""

    def __init__(self, *, dataset_path: str | Path, mode: str) -> None:
        super().__init__(
            dataset_path=dataset_path,
            name=f"therapeutic_quality_{mode}",
        )
        self.mode = mode

    def parse_case(self, raw_case: Any) -> TherapeuticEvalCase:
        """Parse one therapeutic quality case.

        Args:
            raw_case (Any): Raw JSON case.

        Returns:
            TherapeuticEvalCase: Parsed case.
        """

        return parse_therapeutic_case(raw_case)

    def case_id(self, case: TherapeuticEvalCase, index: int) -> str:
        """Return the dataset id for one therapeutic case.

        Args:
            case (TherapeuticEvalCase): Parsed case.
            index (int): Zero-based position.

        Returns:
            str: Stable case id.
        """

        return case.id

    async def run_case(self, case: TherapeuticEvalCase) -> EvalResult:
        """Run and grade one therapeutic quality case.

        Args:
            case (TherapeuticEvalCase): Parsed case.

        Returns:
            EvalResult: Case result.
        """

        if self.mode == "scripted":
            llm = build_scripted_llm(case)
            output = await invoke_therapeutic_branch(
                case,
                llm_client=llm,
                response_llm=llm,
            )
        else:
            control_llm, response_llm = build_live_therapeutic_llms()
            output = await invoke_therapeutic_branch(
                case,
                llm_client=control_llm,
                response_llm=response_llm,
            )

        hard_failures = grade_therapeutic_output(case, output)
        rubric_result = _grade_rubric(
            text=str(output.get("response_text", "")),
            rubric=case.rubric,
        )
        failures = [*hard_failures, *rubric_result["failures"]]
        min_score = float(case.expected.get("min_quality_score", 1.0))
        if rubric_result["score"] < min_score:
            failures.append(
                f"quality score {rubric_result['score']:.2f} below minimum {min_score:.2f}"
            )

        return EvalResult(
            case_id=case.id,
            passed=not failures,
            score=rubric_result["score"] if not hard_failures else 0.0,
            details={
                "description": case.description,
                "mode": self.mode,
                "failures": failures,
                "rubric": rubric_result,
                "output": output,
            },
        )


def _grade_rubric(*, text: str, rubric: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            failures.append(detail or name)

    clean_text = text.strip()
    words = _word_count(clean_text)
    record("non_empty", bool(clean_text), "response_text is empty")

    max_words = rubric.get("max_words")
    if max_words is not None:
        max_words_int = int(max_words)
        record(
            "max_words",
            words <= max_words_int,
            f"word count {words} exceeds {max_words_int}",
        )

    max_questions = rubric.get("max_questions")
    if max_questions is not None:
        question_count = clean_text.count("?")
        max_questions_int = int(max_questions)
        record(
            "max_questions",
            question_count <= max_questions_int,
            f"question count {question_count} exceeds {max_questions_int}",
        )

    for phrase in rubric.get("required_phrases", []):
        phrase_text = str(phrase)
        record(
            f"required_phrase:{phrase_text}",
            phrase_text.casefold() in clean_text.casefold(),
            f"missing required phrase {phrase_text!r}",
        )

    for phrase in rubric.get("forbidden_phrases", []):
        phrase_text = str(phrase)
        record(
            f"forbidden_phrase:{phrase_text}",
            phrase_text.casefold() not in clean_text.casefold(),
            f"contains forbidden phrase {phrase_text!r}",
        )

    if rubric.get("must_not_offer_menu"):
        menu_phrases = ("which would you like", "choose one", "option 1", "option 2")
        found = [phrase for phrase in menu_phrases if phrase in clean_text.casefold()]
        record(
            "must_not_offer_menu",
            not found,
            f"looks like a menu: {found}",
        )

    passed = sum(1 for check in checks if check["passed"])
    score = passed / len(checks) if checks else 1.0
    return {
        "score": score,
        "checks": checks,
        "failures": failures,
    }


def _word_count(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def _build_parser() -> argparse.ArgumentParser:
    parser = build_base_arg_parser("Evaluate therapeutic response quality.")
    parser.set_defaults(dataset=_DEFAULT_DATASET)
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="scripted avoids provider calls; live uses configured LLM clients.",
    )
    return parser


def main() -> int:
    """Run the therapeutic quality evaluator CLI.

    Returns:
        int: Shell exit code.
    """

    return run_evaluator_cli(
        lambda args: TherapeuticQualityEvaluator(
            dataset_path=args.dataset,
            mode=args.mode,
        ),
        parser=_build_parser(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
