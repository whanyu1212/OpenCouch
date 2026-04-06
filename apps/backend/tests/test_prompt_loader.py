from agent.prompts import (
    build_core_system_prompt,
    build_crisis_response_system_prompt,
    build_system_prompt,
    build_therapeutic_system_prompt,
)
import pytest
from agent.prompts.loader import get_knowledge_root, load_knowledge_file


def test_knowledge_root_exists() -> None:
    root = get_knowledge_root()
    assert root.exists()
    assert root.name == "knowledge"


def test_load_knowledge_file_reads_markdown_content() -> None:
    content = load_knowledge_file("soul.md")
    assert "OpenCouch" in content
    assert "mental health support assistant" in content


def test_core_system_prompt_is_composed_from_knowledge_files() -> None:
    prompt = build_core_system_prompt()
    assert "OpenCouch is a calm, direct, and humane mental health support assistant." in prompt
    assert "OpenCouch presents itself as an AI mental health support product." in prompt
    assert "OpenCouch must not:" in prompt


def test_mode_prompts_include_policy_and_modality_knowledge() -> None:
    therapeutic = build_therapeutic_system_prompt()
    crisis = build_crisis_response_system_prompt()

    assert "Motivational Interviewing" in therapeutic
    assert "This policy overrides all normal support behavior." in crisis
    assert "Psychological First Aid" in crisis


def test_generic_system_prompt_composes_mode_and_modalities() -> None:
    prompt = build_system_prompt(
        mode="reflection",
        modalities=("grief_support", "motivational_interviewing"),
    )

    assert "Reflection Mode" in prompt
    assert "Grief Support" in prompt
    assert "Motivational Interviewing" in prompt


def test_invalid_modality_for_mode_raises() -> None:
    with pytest.raises(ValueError):
        build_system_prompt(
            mode="crisis_response",
            modalities=("cbt",),
        )
