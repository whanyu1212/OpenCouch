"""Tests for guided exercise skill docs and loadouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.skills.guided_exercises.registry import (
    EXERCISE_5_4_3_2_1,
    EXERCISE_BOX_BREATHING,
    iter_exercise_definitions,
)
from agent.skills.guided_exercises.rendering.skill_docs import (
    get_guided_exercise_skill_doc,
    iter_guided_exercise_skill_docs,
    validate_guided_exercise_skill_docs,
)


def test_guided_exercise_skill_docs_validate_against_registry() -> None:
    validate_guided_exercise_skill_docs(iter_exercise_definitions())


def test_guided_exercise_skill_doc_loads_by_exercise_id() -> None:
    doc = get_guided_exercise_skill_doc(EXERCISE_5_4_3_2_1)

    assert doc is not None
    assert doc.name == EXERCISE_5_4_3_2_1
    assert doc.description
    assert doc.version == 1
    assert doc.category == "grounding"
    assert doc.channels == ("text", "voice")
    assert "Operating boundaries" in doc.body


def test_guided_exercise_skill_docs_are_directory_ordered() -> None:
    docs = iter_guided_exercise_skill_docs()

    assert [doc.name for doc in docs] == [
        EXERCISE_5_4_3_2_1,
        EXERCISE_BOX_BREATHING,
    ]


def test_guided_exercise_skill_doc_validation_rejects_unknown_exercise(
    tmp_path: Path,
) -> None:
    _write_skill_doc(
        tmp_path / "unknown" / "SKILL.md",
        name="unknown_exercise",
        description="Unknown exercise.",
    )

    with pytest.raises(ValueError, match="unknown exercise"):
        validate_guided_exercise_skill_docs(
            iter_exercise_definitions(),
            catalog_dir=tmp_path,
        )


def test_guided_exercise_skill_doc_validation_rejects_duplicate_names(
    tmp_path: Path,
) -> None:
    _write_skill_doc(
        tmp_path / "one" / "SKILL.md",
        name=EXERCISE_5_4_3_2_1,
        description="First copy.",
    )
    _write_skill_doc(
        tmp_path / "two" / "SKILL.md",
        name=EXERCISE_5_4_3_2_1,
        description="Second copy.",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        validate_guided_exercise_skill_docs(
            iter_exercise_definitions(),
            catalog_dir=tmp_path,
        )


def test_guided_exercise_skill_doc_validation_rejects_channel_mismatch(
    tmp_path: Path,
) -> None:
    _write_skill_doc(
        tmp_path / EXERCISE_5_4_3_2_1 / "SKILL.md",
        name=EXERCISE_5_4_3_2_1,
        description="Grounding exercise.",
        channels=("video",),
    )

    with pytest.raises(ValueError, match="unsupported channels"):
        validate_guided_exercise_skill_docs(
            iter_exercise_definitions(),
            catalog_dir=tmp_path,
        )


def _write_skill_doc(
    path: Path,
    *,
    name: str,
    description: str,
    version: int = 1,
    category: str = "grounding",
    channels: tuple[str, ...] = ("text", "voice"),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channel_lines = "\n".join(f"  - {channel}" for channel in channels)
    path.write_text(
        (
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            f"version: {version}\n"
            f"category: {category}\n"
            "channels:\n"
            f"{channel_lines}\n"
            "---\n\n"
            "# Test Skill\n\n"
            "## Operating boundaries\n\n"
            "- Follow runtime state.\n"
        ),
        encoding="utf-8",
    )
