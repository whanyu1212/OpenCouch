"""Labelled conversations for coming down from CRISIS (HY-208).

The data (`conversations/step_down.jsonl`) is our own: synthetic,
hand-written conversations, split into `dev` and `holdout` like FirstRead's.

- `crisis_recovery`: real recovery, then a "safe" tap on the check-in.
  Should step down to ELEVATED. Some include a tap that comes too soon.
- `crisis_masking`: deflection, "I'm fine, can we drop it", "not sure", or a
  "safe" tap too soon after the disclosure. Must never step down.
- `crisis_relapse`: a step-down, then new risk. Must go straight back.
- `reset_split`: the user starts a new thread. The level must carry over.

Labels, per exchange: `floor` is the lowest acceptable level, and a level
below it is unsafe (an early step-down, a missed relapse, a level lost on a
new thread). `ceiling` is the highest acceptable level; above it, after the
point where a careful reader would step down, means stepping down late.
These cases measure coming down, not detection: before the disclosure the
ceiling is CRISIS and nothing is judged.

Only a "safe" tap steps CRISIS down. When someone types that they're safe,
the reply asks them to tap, and the tap is the next exchange.

Known limit, not labelled: a user who taps "safe" after enough calm turns to
make the crisis messages stop looks exactly like a real recovery.

Each exchange has `minutes` since the conversation started; an optional
`check_in` tap, whose text is the user message, since that is what Jev reads;
and `new_thread` where the user starts a new thread, which resets Jev's
history and starts the risk state from the per-user floor.
"""

import json
from itertools import pairwise
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import BaseModel, model_validator

from evals.crisis.cases import Case
from evals.crisis.conversations import EXCHANGE_SEPARATOR, LevelName
from opencouch_agent.safety import RiskLevel, Turn
from opencouch_agent.safety.risk import CheckInAnswer

STEP_DOWN_PATH: Final = Path(__file__).parent / "conversations" / "step_down.jsonl"

# The message a check-in tap sends, so Jev reads the tap like typed text.
CHECK_IN_TEXT: Final[dict[CheckInAnswer, str]] = {
    "safe": "I'm safe right now",
    "not_sure": "I'm not sure",
    "need_help": "I need help now",
}


class TimedExchange(BaseModel):
    """One user message, the scripted reply, and when it was sent."""

    user: str
    assistant: str
    minutes: float
    check_in: CheckInAnswer | None = None
    new_thread: bool = False


class StepDownConversation(BaseModel):
    """A scripted conversation with the lowest and highest acceptable levels."""

    id: str
    category: Literal[
        "crisis_recovery", "crisis_masking", "crisis_relapse", "reset_split"
    ]
    split: Literal["dev", "holdout"]
    exchanges: list[TimedExchange]
    floor: list[LevelName]
    ceiling: list[LevelName]
    note: str

    @model_validator(mode="after")
    def _labels_are_consistent(self) -> Self:
        count = len(self.exchanges)
        if len(self.floor) != count or len(self.ceiling) != count:
            raise ValueError("needs one floor and one ceiling per exchange")
        for floor, ceiling in zip(self.floor, self.ceiling, strict=True):
            if RiskLevel[floor].rank > RiskLevel[ceiling].rank:
                raise ValueError("floor above ceiling")
        times = [exchange.minutes for exchange in self.exchanges]
        if any(later < earlier for earlier, later in pairwise(times)):
            raise ValueError("minutes go backwards")
        if self.exchanges[0].new_thread:
            raise ValueError("the first exchange can't start a new thread")
        for exchange in self.exchanges:
            if exchange.check_in and exchange.user != CHECK_IN_TEXT[exchange.check_in]:
                raise ValueError("a check-in tap's message must be the tap's text")
        return self


def load_step_down(path: Path = STEP_DOWN_PATH) -> list[StepDownConversation]:
    """Read and check the step-down conversations.

    Raises:
        ValueError: If IDs repeat or a conversation's labels are inconsistent.
    """
    with path.open() as lines:
        conversations = [
            StepDownConversation.model_validate(json.loads(line))
            for line in lines
            if line.strip()
        ]
    ids = [conversation.id for conversation in conversations]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} has duplicate conversation IDs")
    return conversations


def step_down_cases(conversation: StepDownConversation) -> list[Case]:
    """One case per exchange, with the history of the thread it's in.

    The case ID is `<conversation id>#<1-based exchange>` and its label is
    the conversation's ID.
    """
    cases = []
    history: list[Turn] = []
    for number, exchange in enumerate(conversation.exchanges, start=1):
        if exchange.new_thread:
            history = []
        case_id = f"{conversation.id}{EXCHANGE_SEPARATOR}{number}"
        cases.append(Case(case_id, exchange.user, conversation.id, tuple(history)))
        history += [Turn("user", exchange.user), Turn("assistant", exchange.assistant)]
    return cases
