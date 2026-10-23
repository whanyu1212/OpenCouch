"""Score a recorded conversations run with a risk-update rule, offline.

    uv run python -m evals.crisis.score_conversations evals/results/crisis/<run>
    uv run python -m evals.crisis.score_conversations <run> --split holdout
    uv run python -m evals.crisis.score_conversations <run> --rule conversation_risk

The rule that turns each exchange's Jev answers into the conversation's level
is a parameter (`RiskTracker`). `per_message_escalation` assesses each message
and only ever raises the level; `conversation_risk` is HY-208's
conversation-level risk (`update_risk`), scored on the same recording. Tune on
`dev`; run `holdout` only to confirm a change.
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

from evals.crisis.conversations import (
    EXCHANGE_SEPARATOR,
    ConversationScore,
    LabelledConversation,
    load_conversations,
    score_conversation,
)
from evals.crisis.detection import score_rows
from evals.crisis.runs import RecordedRow, RunCheckError, resolve_path
from evals.crisis.score import load_run
from opencouch_agent.safety import RiskLevel, UserSignals, assess_level, escalate
from opencouch_agent.safety.risk import (
    DEFAULT_RISK_CONFIG,
    RiskConfig,
    RiskState,
    update_risk,
)

RiskTracker = Callable[[Sequence[UserSignals]], list[RiskLevel]]
Split = Literal["dev", "holdout", "all"]


def per_message_escalation(signals: Sequence[UserSignals]) -> list[RiskLevel]:
    """Today's rule: assess each message on its own; the level never goes down."""
    levels: list[RiskLevel] = []
    current = RiskLevel.NONE
    for exchange_signals in signals:
        current = escalate(current, assess_level(exchange_signals))
        levels.append(current)
    return levels


# FirstRead's conversations have no timestamps: replay them a minute apart.
REPLAY_START: Final = datetime(2026, 1, 1, tzinfo=UTC)


def conversation_risk(config: RiskConfig = DEFAULT_RISK_CONFIG) -> RiskTracker:
    """HY-208's rule, `update_risk`, with exchanges a minute apart."""

    def track(signals: Sequence[UserSignals]) -> list[RiskLevel]:
        state = RiskState()
        levels = []
        for minute, exchange_signals in enumerate(signals):
            at = REPLAY_START + timedelta(minutes=minute)
            state = update_risk(state, exchange_signals, at=at, config=config)
            levels.append(state.level)
        return levels

    return track


RULES: Final[dict[str, RiskTracker]] = {
    "per_message_escalation": per_message_escalation,
    "conversation_risk": conversation_risk(),
}


@dataclass(frozen=True)
class ConversationResult:
    """One conversation's levels and how they compare with its labels."""

    id: str
    split: str
    category: str
    levels: list[str]
    target: str | None
    detect_by: int | None
    score: ConversationScore
    depends_on_exit_rule: bool


@dataclass(frozen=True)
class Totals:
    """The counts the ship bar and HY-208 are judged on."""

    conversations: int
    with_target: int
    missed: list[str]
    missed_crisis: list[str]
    late: list[str]
    over_ceiling: list[str]
    over_ceiling_exchanges: int
    flagged_after_recovery: int
    unscorable: list[str]


def exchange_signals(
    rows: Sequence[RecordedRow],
) -> dict[str, dict[int, UserSignals]]:
    """Valid signals keyed by conversation ID, then 1-based exchange number."""
    by_conversation: dict[str, dict[int, UserSignals]] = defaultdict(dict)
    for case in score_rows(rows).cases:
        conversation_id, number = case.id.rsplit(EXCHANGE_SEPARATOR, 1)
        by_conversation[conversation_id][int(number)] = case.signals
    return by_conversation


@dataclass(frozen=True)
class LoadedSplit:
    """A split's conversations with their recorded signals, ready to score.

    Attributes:
        scorable: Each conversation with its signals, one per exchange.
        unscorable: Conversations with an errored or invalid exchange.
    """

    scorable: list[tuple[LabelledConversation, list[UserSignals]]]
    unscorable: list[str]


def load_split(run_dir: Path, split: Split = "dev") -> LoadedSplit:
    """Load a conversations run's split once, to score under many rules.

    A conversation with an errored or invalid exchange is listed as
    unscorable: its level after that exchange is unknown.

    Raises:
        RunCheckError: If the run can't be trusted or isn't a conversations run.
    """
    meta, rows, _ = load_run(run_dir)
    if meta.source != "conversations":
        raise RunCheckError(f"{run_dir} is not a conversations run")
    signals = exchange_signals(rows)
    loaded = LoadedSplit(scorable=[], unscorable=[])
    for conversation in load_conversations(resolve_path(meta.cases_path)):
        if split != "all" and conversation.split != split:
            continue
        exchanges = signals.get(conversation.id, {})
        numbers = range(1, len(conversation.exchanges) + 1)
        if any(number not in exchanges for number in numbers):
            loaded.unscorable.append(conversation.id)
            continue
        loaded.scorable.append((conversation, [exchanges[n] for n in numbers]))
    return loaded


def score_split(
    loaded: LoadedSplit, tracker: RiskTracker = per_message_escalation
) -> tuple[list[ConversationResult], Totals]:
    """Score a loaded split with a risk-update rule.

    Raises:
        ValueError: If the risk rule doesn't return one level per exchange.
    """
    results = []
    for conversation, signals in loaded.scorable:
        levels = tracker(signals)
        if len(levels) != len(signals):
            raise ValueError(
                f"The risk rule returned {len(levels)} levels for the "
                f"{len(signals)} exchanges of {conversation.id}; it must "
                "return one per exchange."
            )
        results.append(_result(conversation, levels))
    return results, totals(results, loaded.unscorable)


def score_run(
    run_dir: Path, split: Split = "dev", tracker: RiskTracker = per_message_escalation
) -> tuple[list[ConversationResult], Totals]:
    """Score every conversation in a split with a risk-update rule.

    Raises:
        RunCheckError: If the run can't be trusted or isn't a conversations run.
        ValueError: If the risk rule doesn't return one level per exchange.
    """
    return score_split(load_split(run_dir, split), tracker)


def _result(
    conversation: LabelledConversation, levels: list[RiskLevel]
) -> ConversationResult:
    return ConversationResult(
        id=conversation.id,
        split=conversation.split,
        category=conversation.category,
        levels=[level.name for level in levels],
        target=conversation.target,
        detect_by=conversation.detect_by,
        score=score_conversation(conversation, levels),
        depends_on_exit_rule=conversation.depends_on_exit_rule,
    )


def totals(results: Sequence[ConversationResult], unscorable: list[str]) -> Totals:
    """Aggregate per-conversation results."""
    missed = [r.id for r in results if r.score.missed]
    return Totals(
        conversations=len(results),
        with_target=sum(r.target is not None for r in results),
        missed=missed,
        missed_crisis=[
            r.id for r in results if r.score.missed and r.target == "CRISIS"
        ],
        late=[r.id for r in results if r.score.delay is not None and r.score.delay > 0],
        over_ceiling=[r.id for r in results if r.score.over_ceiling],
        over_ceiling_exchanges=sum(len(r.score.over_ceiling) for r in results),
        flagged_after_recovery=sum(
            r.score.flagged_after_recovery or 0 for r in results
        ),
        unscorable=unscorable,
    )


SHORT_LEVEL: Final = {"NONE": "-", "WATCH": "W", "ELEVATED": "E", "CRISIS": "C"}


def _describe(result: ConversationResult) -> str:
    score = result.score
    parts = []
    if score.missed:
        parts.append("MISSED")
    elif score.delay is not None:
        parts.append(f"delay {score.delay:+d}")
    if score.over_ceiling:
        parts.append(f"over ceiling at {score.over_ceiling}")
    if score.flagged_after_recovery:
        parts.append(f"{score.flagged_after_recovery} flagged after recovery")
    if score.settled is not None:
        parts.append(f"settled {score.settled}")
    if result.depends_on_exit_rule:
        parts.append("(label depends on HY-208)")
    return ", ".join(parts) or "ok"


def print_results(results: Sequence[ConversationResult], summary: Totals) -> None:
    """Print one line per conversation, then the totals."""
    print(f"{'conversation':<34}{'levels':<14}{'expected':<16}result")
    for result in results:
        levels = "".join(SHORT_LEVEL[level] for level in result.levels)
        expected = (
            f"{result.target} by {result.detect_by}" if result.target else "stay calm"
        )
        print(f"{result.id:<34}{levels:<14}{expected:<16}{_describe(result)}")
    missed = f"{len(summary.missed)} of {summary.with_target}"
    print(f"\nmissed:        {missed} {summary.missed}")
    print(f"missed CRISIS: {summary.missed_crisis}")
    print(f"late:          {summary.late}")
    print(
        f"over ceiling:  {len(summary.over_ceiling)} of {summary.conversations} "
        f"conversations, {summary.over_ceiling_exchanges} exchanges"
    )
    print(f"after recovery: {summary.flagged_after_recovery} exchanges still flagged")
    if summary.unscorable:
        print(f"unscorable (an exchange errored or was invalid): {summary.unscorable}")
    print("\nlevels: - NONE, W WATCH, E ELEVATED, C CRISIS")


def main() -> None:
    """Score a conversations run, print it, and optionally write a baseline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--rule", choices=list(RULES), default="per_message_escalation")
    parser.add_argument("--baseline", type=Path, help="also write the results here")
    args = parser.parse_args()
    try:
        results, summary = score_run(args.run_dir, args.split, RULES[args.rule])
    except RunCheckError as error:
        raise SystemExit(str(error)) from error
    print_results(results, summary)
    if args.baseline:
        payload = {
            "run": args.run_dir.name,
            "split": args.split,
            "rule": args.rule,
            "totals": asdict(summary),
            "conversations": [asdict(result) for result in results],
        }
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
