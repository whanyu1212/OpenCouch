from agent.prompts import (
    build_core_system_prompt,
    build_crisis_response_system_prompt,
    build_system_prompt,
    build_therapeutic_system_prompt,
)
import pytest
from agent.prompts.loader import get_knowledge_root, load_knowledge_file


def test_knowledge_root_exists() -> None:
    """Knowledge root resolution should point at the repo knowledge directory."""

    root = get_knowledge_root()
    assert root.exists()
    assert root.name == "knowledge"


def test_load_knowledge_file_reads_markdown_content() -> None:
    """Knowledge files should load repo-backed markdown content."""

    content = load_knowledge_file("soul.md")
    assert "OpenCouch" in content
    assert "mental health support assistant" in content


def test_core_system_prompt_is_composed_from_knowledge_files() -> None:
    """Core system prompt should include the expected knowledge fragments."""

    prompt = build_core_system_prompt()
    assert (
        "OpenCouch is a calm, direct, and humane mental health support assistant."
        in prompt
    )
    assert "OpenCouch presents itself as an AI mental health support product." in prompt
    assert "OpenCouch must not:" in prompt


def test_mode_prompts_include_policy_and_modality_knowledge() -> None:
    """Mode prompts should compose policy and modality overlays."""

    therapeutic = build_therapeutic_system_prompt()
    crisis = build_crisis_response_system_prompt()

    assert "Motivational Interviewing" in therapeutic
    assert "This policy overrides all normal support behavior." in crisis
    assert "Psychological First Aid" in crisis


def test_generic_system_prompt_composes_mode_and_modalities() -> None:
    """Generic system prompt builder should compose arbitrary overlays."""

    prompt = build_system_prompt(
        mode="pattern_reflection",
        modalities=("grief_support", "act"),
    )

    assert "Pattern Reflection Mode" in prompt
    assert "Grief Support" in prompt
    assert "Acceptance and Commitment Therapy" in prompt


def test_new_modality_overlays_are_available_for_composition() -> None:
    """New modality overlays should be available through the prompt catalog."""

    prompt = build_system_prompt(
        mode="guided_exercise",
        modalities=("act", "pfa"),
    )

    assert "Acceptance and Commitment Therapy" in prompt
    assert "Psychological First Aid" in prompt


def test_psychoeducation_mode_is_available_for_composition() -> None:
    """Psychoeducation mode should compose its dedicated knowledge file."""

    prompt = build_system_prompt(
        mode="psychoeducation",
        modalities=("act",),
    )

    assert "Psychoeducation Mode" in prompt
    assert "Acceptance and Commitment Therapy" in prompt


def test_invalid_modality_for_mode_raises() -> None:
    """Invalid mode and modality combinations should fail fast."""

    with pytest.raises(ValueError):
        build_system_prompt(
            mode="crisis_response",
            modalities=("cbt",),
        )
