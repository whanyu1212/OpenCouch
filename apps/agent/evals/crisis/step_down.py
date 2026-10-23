"""Tune when CRISIS steps down, on recorded Jev answers, offline.

    uv run python -m evals.crisis.step_down <step-down run> \
        --conversations evals/crisis/conversations/recorded
    uv run python -m evals.crisis.step_down <run> --split holdout \
        --exit-score 0.15 --calm-minutes 10 --entry-score 0 --clock-reset 0.15

Every setting in the grid replays the step-down conversations through
`update_risk`. A setting passes only if both hold:

- no step-down exchange is below its floor (no early step-down, no missed
  relapse, no level lost on a new thread);
- on FirstRead's conversations it misses nothing the per-message rule
  catches.

Among the settings that pass, fewer exchanges above the ceiling (stepping
down late) is better, and ties go to the setting with more passing
neighbours, so the choice sits in a flat region rather than on an edge. Only
the step-down settings vary; the rest of `RiskConfig` stays at the measured
baseline. Tune on `dev`, then confirm one setting on `holdout`.

Limit: taps happen where the script puts them. Live, the buttons appear as
soon as `check_in_due` holds, so a setting that would offer the check-in
earlier than the scripted tap gets no credit for it. Small differences in
late step-downs between nearby settings are partly the script's timing.
"""

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, replace
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Final

from evals.crisis.runs import RunCheckError, resolve_path
from evals.crisis.score import load_run
from evals.crisis.score_conversations import (
    REPLAY_START,
    SHORT_LEVEL,
    LoadedSplit,
    Split,
    conversation_risk,
    exchange_signals,
    load_split,
    per_message_escalation,
    score_split,
)
from evals.crisis.step_down_cases import StepDownConversation, load_step_down
from opencouch_agent.safety import RiskLevel, UserSignals
from opencouch_agent.safety.floor import initial_risk, risk_floor
from opencouch_agent.safety.risk import (
    DEFAULT_RISK_CONFIG,
    RiskConfig,
    RiskState,
    update_risk,
)

EXIT_SCORES: Final = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
CALM_MINUTES: Final = (0.0, 5.0, 10.0, 15.0, 20.0)
ENTRY_SCORES: Final = (0.0, 0.5, 1.0)
CLOCK_RESETS: Final = (0.15, 0.25, 0.35, 0.45)
AXES: Final = (EXIT_SCORES, CALM_MINUTES, ENTRY_SCORES, CLOCK_RESETS)


@dataclass(frozen=True)
class StepDownResult:
    """One conversation's levels and where they leave the labels."""

    id: str
    category: str
    levels: list[str]
    below_floor: list[int]
    above_ceiling: list[int]


@dataclass(frozen=True)
class Setting:
    """One point in the grid: the step-down settings being tuned."""

    exit_score: float
    calm_minutes: float
    entry_score: float
    clock_reset: float

    def config(self) -> RiskConfig:
        """The baseline config with these step-down settings."""
        return replace(
            DEFAULT_RISK_CONFIG,
            crisis_exit_score=self.exit_score,
            min_calm_minutes=self.calm_minutes,
            crisis_entry_score=self.entry_score,
            clock_reset_adds=self.clock_reset,
        )

    def position(self) -> tuple[int, ...]:
        """Where the setting sits on each axis of the grid."""
        values = astuple(self)
        return tuple(
            axis.index(value) for axis, value in zip(AXES, values, strict=True)
        )


@dataclass(frozen=True)
class GridRow:
    """How one setting did."""

    setting: Setting
    below_floor: int
    above_ceiling: int
    missed: int
    passes: bool
    passing_neighbours: int = 0


def replay(
    conversation: StepDownConversation,
    signals: Mapping[int, UserSignals],
    config: RiskConfig,
) -> list[RiskLevel]:
    """The level after each exchange, as the live gate would track it."""
    state = RiskState()
    levels = []
    for number, exchange in enumerate(conversation.exchanges, start=1):
        at = REPLAY_START + timedelta(minutes=exchange.minutes)
        if exchange.new_thread:
            state = initial_risk(risk_floor(state), at)
        state = update_risk(
            state, signals[number], at=at, check_in=exchange.check_in, config=config
        )
        levels.append(state.level)
    return levels


def score_step_down(
    conversation: StepDownConversation, levels: Sequence[RiskLevel]
) -> StepDownResult:
    """Compare the levels with the conversation's floor and ceiling."""
    numbered = list(enumerate(levels, start=1))
    return StepDownResult(
        id=conversation.id,
        category=conversation.category,
        levels=[level.name for level in levels],
        below_floor=[
            n
            for n, level in numbered
            if level.rank < RiskLevel[conversation.floor[n - 1]].rank
        ],
        above_ceiling=[
            n
            for n, level in numbered
            if level.rank > RiskLevel[conversation.ceiling[n - 1]].rank
        ],
    )


def load_step_down_run(
    run_dir: Path, split: Split
) -> list[tuple[StepDownConversation, dict[int, UserSignals]]]:
    """The conversations in a split, each with its recorded signals.

    Raises:
        RunCheckError: If the run can't be trusted, isn't a step-down run, or
            has an exchange that errored.
    """
    meta, rows, _ = load_run(run_dir)
    if meta.source != "step_down":
        raise RunCheckError(f"{run_dir} is not a step-down run")
    signals = exchange_signals(rows)
    loaded = []
    for conversation in load_step_down(resolve_path(meta.cases_path)):
        if split != "all" and conversation.split != split:
            continue
        exchanges = signals.get(conversation.id, {})
        if len(exchanges) != len(conversation.exchanges):
            raise RunCheckError(
                f"{conversation.id} has an exchange with no valid answer in "
                f"{run_dir}. Finish the run with --resume."
            )
        loaded.append((conversation, exchanges))
    return loaded


def evaluate(
    config: RiskConfig,
    conversations: Sequence[tuple[StepDownConversation, dict[int, UserSignals]]],
) -> list[StepDownResult]:
    """Score every step-down conversation under one setting."""
    return [
        score_step_down(conversation, replay(conversation, signals, config))
        for conversation, signals in conversations
    ]


def grid_search(
    conversations: Sequence[tuple[StepDownConversation, dict[int, UserSignals]]],
    detection: LoadedSplit | None,
) -> list[GridRow]:
    """Every setting in the grid, best first.

    With `detection`, a setting also fails if it misses any of FirstRead's
    conversations that the per-message rule catches.
    """
    allowed: set[str] = set()
    if detection is not None:
        allowed = set(score_split(detection, per_message_escalation)[1].missed)
    rows = []
    for values in product(*AXES):
        setting = Setting(*values)
        results = evaluate(setting.config(), conversations)
        missed: set[str] = set()
        if detection is not None:
            tracker = conversation_risk(setting.config())
            missed = set(score_split(detection, tracker)[1].missed)
        below = sum(len(result.below_floor) for result in results)
        rows.append(
            GridRow(
                setting=setting,
                below_floor=below,
                above_ceiling=sum(len(result.above_ceiling) for result in results),
                missed=len(missed),
                passes=below == 0 and missed <= allowed,
            )
        )
    return _ranked(rows)


def _ranked(rows: list[GridRow]) -> list[GridRow]:
    passing = {row.setting.position() for row in rows if row.passes}
    counted = [
        replace(
            row,
            passing_neighbours=sum(
                position in passing for position in _neighbours(row.setting)
            ),
        )
        for row in rows
    ]
    return sorted(
        counted,
        key=lambda row: (not row.passes, row.above_ceiling, -row.passing_neighbours),
    )


def _neighbours(setting: Setting) -> list[tuple[int, ...]]:
    """Grid positions one step away along a single axis."""
    position = setting.position()
    neighbours = []
    for axis, values in enumerate(AXES):
        for step in (-1, 1):
            moved = list(position)
            moved[axis] += step
            if 0 <= moved[axis] < len(values):
                neighbours.append(tuple(moved))
    return neighbours


def print_grid(rows: Sequence[GridRow], top: int) -> None:
    """Print the best settings, then how many pass."""
    print("exit  calm  entry  clock  below  above  missed  neighbours  passes")
    for row in rows[:top]:
        setting = row.setting
        print(
            f"{setting.exit_score:<6}{setting.calm_minutes:<6}"
            f"{setting.entry_score:<7}{setting.clock_reset:<7}"
            f"{row.below_floor:<7}{row.above_ceiling:<7}{row.missed:<8}"
            f"{row.passing_neighbours:<12}{'yes' if row.passes else 'no'}"
        )
    passing = sum(row.passes for row in rows)
    print(f"\n{passing} of {len(rows)} settings pass")


def print_results(results: Sequence[StepDownResult]) -> None:
    """Print one line per conversation for one setting."""
    print(f"{'conversation':<34}{'category':<17}{'levels':<12}result")
    for result in results:
        levels = "".join(SHORT_LEVEL[level] for level in result.levels)
        parts = []
        if result.below_floor:
            parts.append(f"BELOW FLOOR at {result.below_floor}")
        if result.above_ceiling:
            parts.append(f"above ceiling at {result.above_ceiling}")
        verdict = ", ".join(parts) or "ok"
        print(f"{result.id:<34}{result.category:<17}{levels:<12}{verdict}")
    print("\nlevels: - NONE, W WATCH, E ELEVATED, C CRISIS")


def main() -> None:
    """Run the grid, or score one setting when all four are given."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="a recorded step-down run")
    parser.add_argument("--conversations", type=Path, help="a conversations run")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--exit-score", type=float)
    parser.add_argument("--calm-minutes", type=float)
    parser.add_argument("--entry-score", type=float)
    parser.add_argument("--clock-reset", type=float)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--baseline", type=Path, help="write the results here")
    args = parser.parse_args()
    try:
        conversations = load_step_down_run(args.run_dir, args.split)
        values = (
            args.exit_score,
            args.calm_minutes,
            args.entry_score,
            args.clock_reset,
        )
        if None in values:
            detection = None
            if args.conversations:
                detection = load_split(args.conversations, args.split)
            rows = grid_search(conversations, detection)
            print_grid(rows, args.top)
            payload: object = [asdict(row) for row in rows]
        else:
            config = Setting(*values).config()
            results = evaluate(config, conversations)
            print_results(results)
            payload = {
                "config": asdict(config),
                "conversations": [asdict(result) for result in results],
            }
    except RunCheckError as error:
        raise SystemExit(str(error)) from error
    if args.baseline:
        report = {"run": args.run_dir.name, "split": args.split, "results": payload}
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
