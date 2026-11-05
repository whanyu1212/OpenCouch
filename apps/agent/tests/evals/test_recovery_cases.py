"""Reviewed recovery cases distinguish typed reassurance from a valid safe tap."""

from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from evals.crisis.cases import load_cases
from evals.runner import check_turn_expectation, load_suite, play_case

EVALS = Path(__file__).parents[2] / "evals"


@pytest.mark.parametrize(
    ("case_id", "final_level"),
    [
        ("crisis-ambiguous-recovery-typed-safe", "CRISIS"),
        ("crisis-clear-recovery-and-safe-tap", "ELEVATED"),
    ],
)
async def test_calm_readings_only_step_down_after_a_valid_tap(
    case_id: str, final_level: str
) -> None:
    case = next(
        case for case in load_suite(EVALS / "cases/coach.yaml") if case.id == case_id
    )
    model = TestModel(custom_output_text="Please call SOS at 1767.", call_tools=[])
    row = await play_case(case, coach_model=model, crisis_model=model)
    assert row.turns[-1].risk_level == final_level
    assert row.turns[-2].state["crisis_check_in"] is not None
    for turn, script in zip(row.turns, case.turns, strict=True):
        assert check_turn_expectation(turn, script.expect) == []
        assert set(turn.state["crisis_resources"]) == {"sg-sos", "sg-emergency"}


@pytest.mark.parametrize(
    ("filename", "count", "labels"),
    [
        ("recovery_recency.jsonl", 16, {"no_new_risk", "new_risk", "ambiguous"}),
        ("recovery_negation.jsonl", 6, {"no_new_risk", "new_risk"}),
    ],
)
def test_recency_pairs_change_only_history(
    filename: str, count: int, labels: set[str]
) -> None:
    cases = load_cases(EVALS / "crisis/samples" / filename)
    by_id = {case.id: case for case in cases}
    assert len(by_id) == len(cases) == count
    histories = set()
    for case in cases:
        if case.id.endswith("-alone"):
            paired = by_id[case.id.removesuffix("-alone") + "-history"]
            assert not case.history
            assert paired.history
            assert (case.text, case.label) == (paired.text, paired.label)
            histories.add(paired.history)
    assert len(histories) == 1
    assert {case.label for case in cases} == labels
