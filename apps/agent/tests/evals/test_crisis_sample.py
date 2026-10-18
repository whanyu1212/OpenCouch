"""Re-score a committed recording of Jev's answers with the current policy.

The recording holds single-message cases with an expected level each. A policy
change that misses a case it used to catch, or flags one it used to leave
alone, fails here before it reaches the full BHH evals. So does a change that
fixes a case, so the improvement is recorded in the baseline file too.
"""

import json
from pathlib import Path

from evals.crisis.cases import load_cases
from evals.crisis.detection import score_rows
from evals.crisis.runs import check_input_version, load_meta
from evals.crisis.score import load_run
from opencouch_agent.safety import RiskLevel

SAMPLES = Path(__file__).parents[2] / "evals" / "crisis" / "samples"
CASES = SAMPLES / "single_turn.jsonl"
RECORDING = SAMPLES / "single_turn_recorded"
BASELINE = json.loads((SAMPLES / "single_turn_baseline.json").read_text())


def test_the_recording_matches_the_current_jev_input() -> None:
    # Fails when the questions or transcript framing change. Re-record with
    # `python -m evals.crisis.record --cases evals/crisis/samples/single_turn.jsonl`
    # and replace single_turn_recorded/.
    check_input_version(load_meta(RECORDING), RECORDING)


def test_misses_and_over_escalations_match_the_baseline() -> None:
    _, rows, errors = load_run(RECORDING)
    scored = score_rows(rows)
    assert errors == 0
    assert scored.invalid == []
    assert len(scored.cases) == len(load_cases(CASES))

    below = {c.id for c in scored.cases if c.level.rank < RiskLevel(c.label).rank}
    above = {c.id for c in scored.cases if c.level.rank > RiskLevel(c.label).rank}

    assert below == set(BASELINE["below_expected"]), "misses changed"
    assert above == set(BASELINE["above_expected"]), "over-escalations changed"
