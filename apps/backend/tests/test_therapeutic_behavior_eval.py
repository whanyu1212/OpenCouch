"""Unit tests for therapeutic_behavior_eval helper logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "eval"
    / "runners"
    / "therapeutic_behavior_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("therapeutic_behavior_eval", RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("therapeutic_behavior_eval", _MODULE)
_SPEC.loader.exec_module(_MODULE)

_evaluate_assertions = _MODULE._evaluate_assertions
_evaluate_output = _MODULE._evaluate_output
_evaluate_state = _MODULE._evaluate_state


class _FakeOutput:
    def __init__(
        self,
        *,
        response_text: str,
        response_style: str | None,
        therapeutic_approach: str | None = None,
    ) -> None:
        self.response_text = response_text
        self.response_style = response_style
        self.therapeutic_approach = therapeutic_approach


def test_evaluate_assertions_flags_missing_required_terms() -> None:
    failures = _evaluate_assertions(
        "I hear you. Tell me more.",
        {"must_include_any": ["grounding", "exercise"]},
    )
    assert failures == ["missing required terms: one of ['grounding', 'exercise']"]


def test_evaluate_assertions_flags_forbidden_terms_and_question_requirement() -> None:
    failures = _evaluate_assertions(
        "Let's do a grounding exercise now.",
        {
            "must_not_include_any": ["exercise", "grounding"],
            "require_question": True,
        },
    )
    assert "contains forbidden terms: ['exercise', 'grounding']" in failures
    assert "expected a question mark in response" in failures


def test_evaluate_output_checks_style_and_text_assertions() -> None:
    case = {
        "id": "sample",
        "expected_response_style": "supportive",
        "assertions": {"max_sentences": 2},
    }
    output = _FakeOutput(
        response_text="Sentence one. Sentence two. Sentence three.",
        response_style="reflective",
        therapeutic_approach="none",
    )

    failures = _evaluate_output(case, output)
    assert any("response_style mismatch" in failure for failure in failures)
    assert any("response too long" in failure for failure in failures)


def test_evaluate_assertions_checks_first_sentence_and_question_count() -> None:
    failures = _evaluate_assertions(
        "Start with the thought. What feels most true about it?",
        {
            "first_sentence_must_include_any": ["hard", "heavy"],
            "max_question_marks": 0,
        },
    )

    assert any(
        "first sentence missing required terms" in failure for failure in failures
    )
    assert any(
        "response has too many question marks" in failure for failure in failures
    )


def test_evaluate_assertions_checks_ordering_before_anchor_terms() -> None:
    failures = _evaluate_assertions(
        "Yes. Start with the thought that's hitting hardest.",
        {
            "must_include_any_before_any_of": {
                "required": ["sounds", "understandable"],
                "before": ["start", "let's"],
            }
        },
    )

    assert failures == ["missing required terms: one of ['sounds', 'understandable']"]


def test_evaluate_state_checks_active_exercise_expectations() -> None:
    failures = _evaluate_state(
        {
            "expected_exercise_active": True,
            "expected_exercise_type": "grounding_5_4_3_2_1",
            "expected_exercise_step_in": [0, 1],
        },
        {
            "exercise_state": {
                "exercise_type": "grounding_5_4_3_2_1",
                "exercise_step": 2,
            }
        },
    )

    assert failures == ["exercise_step mismatch: got 2, expected one of [0, 1]"]
