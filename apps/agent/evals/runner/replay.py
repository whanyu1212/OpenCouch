"""Replay recorded screening through the real gate, without Jev or model calls.

Run `python -m evals.runner.replay <run-dir>` to verify recorded risk decisions.
"""

import argparse
import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import httpx2
from pydantic_ai.messages import ModelRequest, UserPromptPart
from typesafe_sdk import Noul, TypeSafeAPIConnectionError, TypeSafeRateLimitError

from evals.runner.cases import ConversationCase
from evals.runner.execute import DEFAULT_BASE_TIME, DEFAULT_TURN_STEP
from evals.runner.runs import (
    SavedRunRow,
    TurnRecord,
    load_meta_cases,
    load_run_meta,
    read_error_rows,
    read_saved_runs,
    validate_run_rows,
)
from evals.runner.tracing import (
    ScreeningError,
    ScreeningRecord,
    question_definitions,
)
from opencouch_agent.safety.crisis_resources import (
    load_shipped_catalog,
    select_resources,
)
from opencouch_agent.safety.follow_up import (
    CheckInSubmission,
    accepted_check_in,
    prepare_follow_up,
)
from opencouch_agent.safety.gate import GateDecision, run_gate
from opencouch_agent.safety.jev import FailureReason
from opencouch_agent.safety.risk import DEFAULT_RISK_CONFIG


class _RecordedJev:
    def __init__(self, calls: list[ScreeningRecord]) -> None:
        self.remaining = list(calls)
        self.mismatch = False

    async def ask(self, state: str, questions: Mapping[str, Noul]) -> dict[str, float]:
        definitions = question_definitions(questions)
        for index, call in enumerate(self.remaining):
            if call.state == state and call.questions == definitions:
                self.remaining.pop(index)
                if isinstance(call, ScreeningError):
                    _raise_recorded_failure(call.failure)
                else:
                    return call.probabilities
        self.mismatch = True
        raise ValueError("No recorded screening input matches this call")


def _raise_recorded_failure(failure: FailureReason) -> None:
    if failure == FailureReason.TIMEOUT:
        raise TimeoutError("recorded screening timeout")
    if failure == FailureReason.RATE_LIMITED:
        raise TypeSafeRateLimitError(429, {}, httpx2.Headers())
    if failure == FailureReason.UNAVAILABLE:
        raise TypeSafeAPIConnectionError("recorded screening unavailability")
    raise ValueError("recorded screening error")


async def replay_turn(turn: TurnRecord) -> GateDecision:
    """Re-run the gate and follow-up, refusing missing evidence or changed settings."""
    trace = turn.risk_trace
    if trace is None:
        raise ValueError(
            f"Turn {turn.turn_index} predates risk tracing; record a new run"
        )
    if trace.config != DEFAULT_RISK_CONFIG:
        raise ValueError("Risk configuration changed since recording")
    submitted = (
        CheckInSubmission(
            offer_id=trace.before.check_in_offer or "unoffered",
            answer=trace.submitted_check_in,
        )
        if trace.submitted_check_in is not None
        else None
    )
    check_in = accepted_check_in(submitted, trace.before, turn.user, at=trace.at)
    jev = _RecordedJev(trace.calls)
    decision = await run_gate(
        jev,
        trace.before,
        trace.history,
        [ModelRequest(parts=[UserPromptPart(turn.user)])],
        at=trace.at,
        check_in=check_in,
    )
    if jev.remaining or jev.mismatch:
        raise ValueError(
            "Screening inputs/questions changed or recording is incomplete"
        )
    resources = trace.before.crisis_resources or ()
    if decision.path == "crisis":
        resources = tuple(
            resource.key
            for resource in select_resources(
                load_shipped_catalog(),
                "SG",
                decision.risk.crisis_types,
                imminent=decision.imminent,
            )
        )
    decision = prepare_follow_up(decision, trace.before, resources, at=trace.at)
    # Offer IDs are random; compare whether an offer exists, not its fresh UUID.
    if decision.risk.check_in_offer and trace.after.check_in_offer:
        decision = replace(
            decision,
            risk=replace(decision.risk, check_in_offer=trace.after.check_in_offer),
        )
    return decision


def _validate_conversation(row: SavedRunRow, case: ConversationCase) -> None:
    context = f"Case {row.case_id!r}, rep {row.rep}"
    if len(row.turns) != len(case.turns):
        raise ValueError(
            f"{context}: expected {len(case.turns)} turns, got {len(row.turns)}"
        )
    at = DEFAULT_BASE_TIME
    for index, (turn, script) in enumerate(zip(row.turns, case.turns, strict=True)):
        if script.minutes is not None:
            at = DEFAULT_BASE_TIME + timedelta(minutes=script.minutes)
        elif index > 0:
            at += DEFAULT_TURN_STEP
        if turn.turn_index != index or turn.user != script.user:
            raise ValueError(
                f"{context}, turn {index}: index or user differs from script"
            )
        trace = turn.risk_trace
        if trace is None:
            raise ValueError(f"{context}, turn {index}: missing risk trace")
        if trace.submitted_check_in != script.check_in or trace.at != at:
            raise ValueError(
                f"{context}, turn {index}: check-in or time differs from script"
            )


async def main(argv: list[str] | None = None) -> int:
    """Print per-turn replay agreement and diagnostic risk state; fail on mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    meta = load_run_meta(args.run_dir)
    rows = read_saved_runs(args.run_dir)
    errors = read_error_rows(args.run_dir)
    cases = load_meta_cases(meta, args.run_dir)
    validate_run_rows(meta, cases, rows, errors)
    cases_by_id = {case.id: case for case in cases}
    for row in rows:
        _validate_conversation(row, cases_by_id[row.case_id])
    matched = True
    for row in rows:
        for turn in row.turns:
            decision = await replay_turn(turn)
            trace = turn.risk_trace
            agrees = trace is not None and decision.risk == trace.after
            agrees = (
                agrees
                and ("crisis" if decision.path == "crisis" else "coach") == turn.path
            )
            matched = matched and agrees
            print(
                json.dumps(
                    {
                        "case_id": row.case_id,
                        "rep": row.rep,
                        "turn": turn.turn_index + 1,
                        "matches": agrees,
                        "level": decision.risk.level.name,
                        "score": decision.risk.score,
                        "calm_turns": decision.risk.calm_turns,
                        "last_risk_at": str(decision.risk.last_risk_at),
                        "check_in_offered": decision.risk.check_in_offer is not None,
                        "screening_failure": decision.screening_failure,
                        "changes": [change.reason for change in decision.risk.changes],
                    }
                )
            )
    return int(not matched or bool(errors) or not rows)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
