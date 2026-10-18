"""Re-score a committed recording of Jev's answers with the current policy.

The recording holds 41 single-message cases with an expected level each. A
policy change that misses a case it used to catch, or flags one it used to
leave alone, fails here before it reaches the full BHH evals.
"""

import json
from pathlib import Path

from evals.crisis.record import question_set_version
from evals.crisis.score import load_run
from opencouch_agent.safety import RiskLevel

SAMPLES = Path(__file__).parents[2] / "evals" / "crisis" / "samples"
RECORDING = SAMPLES / "single_turn_recorded"
BASELINE = json.loads((SAMPLES / "single_turn_baseline.json").read_text())


def test_the_recording_matches_the_current_questions() -> None:
    meta = json.loads((RECORDING / "meta.json").read_text())
    assert meta["question_set_version"] == question_set_version(), (
        "The Jev questions changed. Re-record the sample with "
        "`python -m evals.crisis.record --cases evals/crisis/samples/"
        "single_turn.jsonl` and replace single_turn_recorded/."
    )


def test_no_new_misses_or_false_alarms_on_the_sample() -> None:
    _, scored, errors = load_run(RECORDING)
    assert errors == 0
    assert scored.invalid == []
    assert len(scored.cases) == 41

    below = {c.id for c in scored.cases if c.level.rank < RiskLevel[c.label].rank}
    above = {c.id for c in scored.cases if c.level.rank > RiskLevel[c.label].rank}

    assert below <= set(BASELINE["below_expected"]), "new misses"
    assert above <= set(BASELINE["above_expected"]), "new over-escalations"
