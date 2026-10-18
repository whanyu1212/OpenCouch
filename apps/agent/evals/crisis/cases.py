"""The cases a recording is made from.

A case is one user message with a label: a BHH crisis type (see `bhh.py`), an
expected risk level for our own cases, or, for one exchange of a labelled
conversation, the conversation's ID (see `conversations.py`).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from opencouch_agent.safety import Turn

DATA_DIR: Final = Path(__file__).parents[1] / "data"
BHH_META_PATH: Final = DATA_DIR / "bhh.meta.json"

BHH_LABELS: Final = (
    "suicidal_ideation",
    "self_harm",
    "violent_thoughts",
    "anxiety_crisis",
    "substance_abuse_or_withdrawal",
    "risk_taking_behaviors",
    "no_crisis",
)


@dataclass(frozen=True)
class Case:
    """One message to screen, with its label.

    Attributes:
        id: Stable case ID.
        text: The user message.
        label: What the case is checked against (see the module docstring).
        history: The conversation before `text`, oldest first; empty for a
            first message.
    """

    id: str
    text: str
    label: str
    history: tuple[Turn, ...] = ()


def load_cases(path: Path) -> list[Case]:
    """Read cases from a JSON Lines file."""
    cases = []
    with path.open() as lines:
        for line in lines:
            if line.strip():
                fields = json.loads(line)
                history = tuple(Turn(**turn) for turn in fields.pop("history", []))
                cases.append(Case(**fields, history=history))
    return cases


def bhh_path(split: str) -> Path:
    """Where a downloaded BHH split lives."""
    return DATA_DIR / f"bhh-{split}.jsonl"
