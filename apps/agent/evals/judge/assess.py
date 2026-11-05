"""Prepare untrusted reply data and perform scalar or blind pairwise judging."""

import hashlib
import json
from pathlib import Path

from evals.judge.client import JudgeClient
from evals.judge.models import (
    JudgeAttempt,
    JudgeInput,
    PairwiseVerdict,
    QualityVerdict,
)

RUBRIC_PATH = Path(__file__).with_name("rubric.md")


def load_rubric() -> str:
    """Read the reviewed rubric independently of candidate data."""
    return RUBRIC_PATH.read_text(encoding="utf-8")


def rubric_digest() -> str:
    """Identify the exact rubric used to produce a saved verdict."""
    return hashlib.sha256(load_rubric().encode()).hexdigest()


async def assess_reply(
    client: JudgeClient, candidate: JudgeInput
) -> tuple[QualityVerdict, list[JudgeAttempt]]:
    """Judge all criteria in one bounded request, with one output per criterion."""
    prompt = json.dumps(
        {"task": "quality", "data": candidate.model_dump(mode="json")},
        ensure_ascii=False,
    )
    return await client.request(prompt, QualityVerdict)


async def compare_replies(
    client: JudgeClient,
    candidate: JudgeInput,
    alternative: JudgeInput,
    *,
    reverse: bool,
) -> tuple[PairwiseVerdict, list[JudgeAttempt]]:
    """Judge one blind ordering; callers run both orders and retain each result."""
    first, second = candidate, alternative
    if reverse:
        first, second = second, first
    prompt = json.dumps(
        {
            "task": "pairwise",
            "A": first.model_dump(mode="json"),
            "B": second.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
    return await client.request(prompt, PairwiseVerdict)
