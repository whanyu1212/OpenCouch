"""Unit tests for crisis_gate_eval helper logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from agent.models import CrisisAssessment

RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "eval" / "runners" / "crisis_gate_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("crisis_gate_eval", RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("crisis_gate_eval", _MODULE)
_SPEC.loader.exec_module(_MODULE)

_evaluate_case = _MODULE._evaluate_case


@pytest.mark.asyncio
async def test_evaluate_case_uses_truth_table_helper_for_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Override assessments should still be normalized through the truth table.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to patch runner helpers.

    Returns:
        None: This test asserts on the runner's override path.
    """

    override_assessment = CrisisAssessment(
        level=2,
        confidence="high",
        reason="override",
        needs_crisis_response=False,
        needs_clarification=True,
    )
    seen: list[CrisisAssessment] = []

    def fake_detect_crisis_override(_state: object) -> tuple[str, CrisisAssessment]:
        return "imminent_risk", override_assessment

    def fake_enforce_crisis_truth_table(
        assessment: CrisisAssessment,
    ) -> CrisisAssessment:
        seen.append(assessment)
        return assessment.model_copy(
            update={
                "needs_crisis_response": True,
                "needs_clarification": False,
            }
        )

    monkeypatch.setattr(_MODULE, "detect_crisis_override", fake_detect_crisis_override)
    monkeypatch.setattr(
        _MODULE,
        "enforce_crisis_truth_table",
        fake_enforce_crisis_truth_table,
    )

    passed, detail = await _evaluate_case(
        {
            "id": "override_case",
            "message": "I have a plan.",
            "history": [],
            "expected_level": 2,
            "expected_needs_crisis_response": True,
            "expected_needs_clarification": False,
        },
        llm_client=None,
    )

    assert passed is True
    assert detail is None
    assert seen == [override_assessment]


@pytest.mark.asyncio
async def test_evaluate_case_uses_llm_path_when_client_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should mirror the node by preferring the LLM when available.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to patch runner helpers.

    Returns:
        None: This test asserts on the runner's non-override decision tree.
    """

    llm_calls: list[str] = []

    def fake_detect_crisis_override(_state: object) -> None:
        return None

    def fake_assess_crisis_risk_deterministically(_state: object) -> CrisisAssessment:
        return CrisisAssessment(
            level=2,
            confidence="high",
            reason="deterministic",
            needs_crisis_response=True,
            needs_clarification=False,
        )

    async def fake_assess_crisis_risk_with_llm(
        _state: object,
        *,
        llm_client: object,
    ) -> CrisisAssessment:
        llm_calls.append("called")
        return CrisisAssessment(
            level=1,
            confidence="medium",
            reason="llm",
            needs_crisis_response=False,
            needs_clarification=True,
        )

    monkeypatch.setattr(_MODULE, "detect_crisis_override", fake_detect_crisis_override)
    monkeypatch.setattr(
        _MODULE,
        "assess_crisis_risk_deterministically",
        fake_assess_crisis_risk_deterministically,
    )
    monkeypatch.setattr(
        _MODULE,
        "assess_crisis_risk_with_llm",
        fake_assess_crisis_risk_with_llm,
    )

    passed, detail = await _evaluate_case(
        {
            "id": "llm_path_case",
            "message": "I feel awful.",
            "history": [],
            "expected_level": 1,
            "expected_needs_crisis_response": False,
            "expected_needs_clarification": True,
        },
        llm_client=object(),
    )

    assert passed is True
    assert detail is None
    assert llm_calls == ["called"]


@pytest.mark.asyncio
async def test_evaluate_case_accepts_expected_level_range_and_optional_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should support range-based level expectations.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture used to patch runner helpers.

    Returns:
        None: This test asserts on the flexible dataset contract.
    """

    def fake_detect_crisis_override(_state: object) -> None:
        return None

    def fake_assess_crisis_risk_deterministically(_state: object) -> CrisisAssessment:
        return CrisisAssessment(
            level=1,
            confidence="medium",
            reason="ambiguous",
            needs_crisis_response=False,
            needs_clarification=True,
        )

    monkeypatch.setattr(_MODULE, "detect_crisis_override", fake_detect_crisis_override)
    monkeypatch.setattr(
        _MODULE,
        "assess_crisis_risk_deterministically",
        fake_assess_crisis_risk_deterministically,
    )

    passed, detail = await _evaluate_case(
        {
            "id": "level_range_case",
            "message": "I don't want to be here at this job anymore.",
            "history": [],
            "expected_level_in": [0, 1],
            "expected_needs_crisis_response": False,
        },
        llm_client=None,
    )

    assert passed is True
    assert detail is None
