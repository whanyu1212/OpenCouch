"""Tests for guided exercise skill docs and loadouts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from agent.skills.guided_exercises.loadout import build_guided_exercise_loadout
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
from agent.state import AgentState


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


def test_guided_exercise_loadout_projects_available_ids_from_state() -> None:
    loadout = build_guided_exercise_loadout(
        cast(
            AgentState,
            {
                "installed_skills": [],
                "channel": "text",
                "therapeutic_approach": None,
            },
        )
    )

    assert EXERCISE_5_4_3_2_1 in loadout.available_exercise_ids
    assert loadout.selected_exercise_id is None
    assert loadout.channel == "text"
    assert loadout.therapeutic_approach is None
    assert loadout.installed_skills == ()


def test_guided_exercise_loadout_respects_voice_channel() -> None:
    loadout = build_guided_exercise_loadout(
        cast(
            AgentState,
            {
                "installed_skills": [],
                "channel": "voice",
                "therapeutic_approach": None,
            },
        ),
        selected_exercise_id=EXERCISE_BOX_BREATHING,
    )

    assert loadout.available_exercise_ids
    assert EXERCISE_BOX_BREATHING in loadout.available_exercise_ids
    assert loadout.selected_exercise_id == EXERCISE_BOX_BREATHING
    assert loadout.channel == "voice"


def test_legacy_skill_doc_import_path_reexports_parser() -> None:
    from agent.skills.guided_exercises.skill_docs import (
        get_guided_exercise_skill_doc as legacy_get_doc,
    )

    assert legacy_get_doc(EXERCISE_5_4_3_2_1) is not None


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
