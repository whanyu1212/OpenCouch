"""Offline risk replay retains live evidence, rechecks and fail-closed decisions."""

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import httpx2
import pytest
from pydantic_ai.models.test import TestModel
from typesafe_sdk import Noul, TypeSafeAPIConnectionError, TypeSafeRateLimitError

from evals.runner import load_suite, play_case, run_suite
from evals.runner.replay import main as replay_main
from evals.runner.replay import replay_turn
from evals.runner.runs import SavedRunRow
from evals.runner.tracing import RecordingJev, ScreeningError, ScreeningSuccess
from opencouch_agent.safety.jev import FailureReason
from opencouch_agent.safety.policy import RiskLevel
from opencouch_agent.safety.questions import user_turn_questions
from opencouch_agent.safety.screening import ScreeningFailed, screen_user_message

SUITE = Path(__file__).parents[2] / "evals/cases/coach.yaml"


class RecoveryJev:
    def __init__(self, first: str) -> None:
        self.first = first
        self.calls = 0

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        self.calls += 1
        reading = dict.fromkeys(questions, 0.0)
        if self.calls == 1:
            if self.first == "timeout":
                raise TimeoutError("private provider payload")
            if self.first == "rate_limited":
                raise TypeSafeRateLimitError(429, {}, httpx2.Headers())
            if self.first == "unavailable":
                raise TypeSafeAPIConnectionError("private provider payload")
            if self.first == "invalid":
                return {"intent": 9.0}
            if self.first == "error":
                raise ValueError("private provider payload")
            if self.first == "renewed":
                reading.update(medical_emergency=0.95, latest_adds_risk=0.5)
        return reading


@pytest.mark.parametrize(
    "first",
    ["calm", "renewed", "timeout", "invalid", "rate_limited", "unavailable", "error"],
)
async def test_recorded_risk_replays_after_serialization(first: str) -> None:
    case = next(
        case for case in load_suite(SUITE) if case.id == "crisis-recovery-and-safe-tap"
    )
    model = TestModel(
        custom_output_text="Please ask your sister to stay with you.", call_tools=[]
    )
    row = await play_case(
        case, coach_model=model, crisis_model=model, jev=RecoveryJev(first)
    )
    saved = SavedRunRow.model_validate_json(row.model_dump_json())
    for turn in saved.turns:
        trace = turn.risk_trace
        assert trace is not None
        assert (await replay_turn(turn)).risk == trace.after
        assert trace.calls
    trace = saved.turns[0].risk_trace
    assert trace is not None and trace.calls[0].source == "scripted"
    trace = saved.turns[1].risk_trace
    assert trace is not None and trace.calls[0].source == "live"
    assert "private provider payload" not in saved.model_dump_json()
    if first == "renewed":
        assert saved.turns[-1].risk_level == "CRISIS"
    else:
        assert saved.turns[-1].risk_level == "ELEVATED"
    if first in {"timeout", "invalid", "rate_limited", "unavailable", "error"}:
        assert trace.after.unscreened
        assert trace.after.level is RiskLevel.CRISIS
        recheck = saved.turns[2].risk_trace
        assert recheck is not None and len(recheck.calls) == 2
        assert not recheck.after.unscreened


async def test_fake_readings_are_explicit_and_missing_evidence_is_rejected() -> None:
    model = TestModel(custom_output_text="Let's look at that worry.", call_tools=[])
    row = await play_case(load_suite(SUITE)[0], coach_model=model)
    turn = row.turns[0]
    trace = turn.risk_trace
    assert trace is not None and trace.calls[0].source == "fake"
    assert isinstance(trace.calls[0], ScreeningSuccess)
    assert (await replay_turn(turn)).risk.level == RiskLevel.NONE
    original = list(trace.calls)
    trace.calls.clear()
    with pytest.raises(ValueError, match="inputs/questions changed"):
        await replay_turn(turn)
    trace.calls = [*original, *original]
    with pytest.raises(ValueError, match="inputs/questions changed"):
        await replay_turn(turn)
    trace.calls = original
    trace.calls[0].questions = "changed questions"
    with pytest.raises(ValueError, match="inputs/questions changed"):
        await replay_turn(turn)
    trace.config = replace(trace.config, min_calm_minutes=1)
    with pytest.raises(ValueError, match="configuration changed"):
        await replay_turn(turn)
    turn.risk_trace = None
    with pytest.raises(ValueError, match="predates risk tracing"):
        await replay_turn(turn)


async def test_screening_deadline_is_recorded_without_swallowing_cancellation() -> None:
    started = asyncio.Event()
    never = asyncio.Event()

    class PendingJev:
        async def ask(
            self, state: str, questions: Mapping[str, Noul]
        ) -> dict[str, float]:
            started.set()
            await never.wait()
            return dict.fromkeys(questions, 0.0)

    recorder = RecordingJev(PendingJev())
    result = await screen_user_message(recorder, [], "hello", deadline_seconds=0.01)
    assert result == ScreeningFailed(FailureReason.TIMEOUT)
    assert isinstance(recorder.calls[0], ScreeningError)
    assert recorder.calls[0].failure == FailureReason.TIMEOUT
    started.clear()
    task = asyncio.create_task(recorder.ask("hello", user_turn_questions()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("changed", ["none", "score", "offer"])
async def test_replay_cli_checks_recorded_outcomes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], changed: str
) -> None:
    outcome = await run_suite(SUITE, split="dev", reps=1, output_dir=tmp_path)
    if changed == "score":
        trace = outcome.results[0].turns[0].risk_trace
        assert trace is not None
        trace.after = replace(trace.after, score=99)
    elif changed == "offer":
        trace = next(
            turn.risk_trace
            for row in outcome.results
            for turn in row.turns
            if turn.risk_trace is not None and turn.risk_trace.after.check_in_offer
        )
        trace.after = replace(trace.after, check_in_offer=None)
    if changed != "none":
        (tmp_path / "results.jsonl").write_text(
            "".join(row.model_dump_json() + "\n" for row in outcome.results)
        )
    capsys.readouterr()
    assert await replay_main([str(tmp_path)]) == int(changed != "none")
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 13
    assert sum(not row["matches"] for row in rows) == int(changed != "none")


async def test_replay_keeps_an_unoffered_safe_tap_in_crisis() -> None:
    case = next(
        case
        for case in load_suite(SUITE)
        if case.id == "crisis-masking-early-tap-stays-crisis"
    )
    model = TestModel(custom_output_text="Please call 995.", call_tools=[])
    row = await play_case(case, coach_model=model, crisis_model=model)
    last = row.turns[-1]
    assert last.risk_trace is not None and last.risk_trace.before.check_in_offer is None
    decision = await replay_turn(last)
    assert decision.risk == last.risk_trace.after
    assert decision.risk.level is RiskLevel.CRISIS


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "missing",
        "extra",
        "duplicate",
        "reordered",
        "index",
        "user",
        "trace",
        "tap",
        "time",
    ],
)
async def test_replay_rejects_incomplete_or_altered_conversations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    outcome = await run_suite(SUITE, split="dev", reps=1, output_dir=tmp_path)
    row = next(
        row for row in outcome.results if row.case_id == "crisis-recovery-and-safe-tap"
    )
    if damage == "empty":
        row.turns.clear()
    elif damage == "missing":
        row.turns.pop(1)
    elif damage == "extra":
        row.turns.append(row.turns[-1])
    elif damage == "duplicate":
        row.turns[1] = row.turns[0]
    elif damage == "reordered":
        row.turns.reverse()
    elif damage == "index":
        row.turns[0].turn_index = 99
    elif damage == "user":
        row.turns[0].user = "Different input"
    elif damage == "trace":
        row.turns[0].risk_trace = None
    else:
        trace = row.turns[-1].risk_trace
        assert trace is not None
        if damage == "tap":
            trace.submitted_check_in = None
        else:
            trace.at = trace.at.replace(year=2027)
    (tmp_path / "results.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in outcome.results)
    )
    capsys.readouterr()
    with pytest.raises(ValueError, match=r"crisis-recovery-and-safe-tap.*rep 1"):
        await replay_main([str(tmp_path)])
    assert capsys.readouterr().out == ""
