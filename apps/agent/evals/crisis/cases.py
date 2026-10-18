"""The cases a recording is made from.

A case is one user message with a label: a BHH crisis type (see `bhh.py`), or
an expected risk level for our own cases.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
        label: A BHH crisis type, or an expected risk level for our own cases.
    """

    id: str
    text: str
    label: str


def load_cases(path: Path) -> list[Case]:
    """Read cases from a JSON Lines file."""
    with path.open() as lines:
        return [Case(**json.loads(line)) for line in lines if line.strip()]


def bhh_path(split: str) -> Path:
    """Where a downloaded BHH split lives."""
    return DATA_DIR / f"bhh-{split}.jsonl"
