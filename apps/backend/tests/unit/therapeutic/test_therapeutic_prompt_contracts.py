"""Characterization contracts for therapeutic prompt products."""

from __future__ import annotations

from hashlib import sha256
from typing import cast

from agent.specialists.therapeutic_response.runtime_prompts import (
    build_therapeutic_agent_input,
    build_therapeutic_response_llm_request,
    operational_context_for_prompt,
)
from agent.runtime.openai_text_runtime import OpenAITextRuntime
from agent.specialists.therapeutic_response.prompts import (
    build_therapeutic_response_prompt,
)
from agent.state import AgentState


def _prompt_state() -> AgentState:
    return cast(
        AgentState,
        {
            "message": "I keep freezing before presentations.",
            "transcript": [
                {"role": "user", "content": "Presentations make me anxious."},
                {"role": "assistant", "content": "What feels hardest about them?"},
            ],
            "working_memory": [
                {
                    "type": "semantic",
                    "evidence_quote": "My sister Maya helps me rehearse.",
                }
            ],
            "procedural_profile": {
                "proactive_recall_enabled": True,
                "procedural_rules": [],
            },
            "memory_reference": {"mode": "none"},
            "memory_control": {"pending_action": None},
            "response_style": "supportive",
            "therapeutic_approach": "motivational_interviewing",
            "turn_lifecycle": {
                "active_flow": "none",
                "action": "none",
                "triage_confidence": "high",
                "clarification_needed": False,
                "clarification_kind": "none",
                "no_clarification_reason": "clear_single_intent",
            },
        },
    )


def _dynamic_state() -> AgentState:
    return cast(
        AgentState,
        {
            "message": "Should I look this up or keep talking?",
            "transcript": [],
            "working_memory": [],
            "procedural_profile": {
                "proactive_recall_enabled": False,
                "procedural_rules": [],
            },
            "memory_reference": {"mode": "explicit"},
            "memory_control": {
                "pending_action": {"target": {"preview": "presentation anxiety note"}}
            },
            "response_style": "clarifying",
            "therapeutic_approach": "none",
            "turn_lifecycle": {
                "active_flow": "none",
                "action": "none",
                "triage_confidence": "low",
                "tentative_route": "grounded_lookup",
                "clarification_needed": True,
                "clarification_kind": "soft",
                "intent_summary": "User may want lookup and support.",
            },
            "crisis": {"needs_clarification": True},
        },
    )


def _digest(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def test_sdk_agent_input_contract_with_history() -> None:
    runtime = OpenAITextRuntime(model="gpt-test")
    prompt = runtime._input_text_for_state(_prompt_state(), include_recent_history=True)  # noqa: SLF001

    assert len(prompt) == 41993
    assert (
        _digest(prompt)
        == "8c3d0aa17f1b4faa5003e035b7cb1af99367c4a5e9b73fe3cfe85cead42ab4fd"
    )
    sections = [
        "Therapeutic response guidance:",
        "Recent conversation:",
        "Relevant context from past sessions:",
        "Current user message:",
        "Operational context:",
    ]
    positions = [prompt.index(section) for section in sections]
    assert positions == sorted(positions)


def test_sdk_agent_input_includes_optional_prompt_appendix() -> None:
    appendix = "TUI-only command guidance"
    prompt = build_therapeutic_agent_input(
        _prompt_state(),
        include_recent_history=True,
        prompt_appendix=appendix,
    )
    request = build_therapeutic_response_llm_request(
        _prompt_state(),
        include_recent_history=True,
        prompt_appendix=appendix,
    )

    assert prompt.count(appendix) == 1
    assert request.system_instruction.count(appendix) == 1


def test_sdk_agent_input_contract_without_history() -> None:
    runtime = OpenAITextRuntime(model="gpt-test")
    prompt = runtime._input_text_for_state(
        _prompt_state(), include_recent_history=False
    )  # noqa: SLF001

    assert len(prompt) == 41933
    assert (
        _digest(prompt)
        == "64cbdfa92fb60e3f8db9cb9346213ff999f784b9fdd38ce2b4eb32dd4db64f59"
    )
    assert "Presentations make me anxious." not in prompt
    assert "Recent conversation:\n(no prior history)" in prompt
    assert "My sister Maya helps me rehearse." in prompt
    assert "Operational context:" in prompt


def test_sdk_agent_input_keeps_full_state_operational_context_without_history() -> None:
    state = _dynamic_state()
    state["transcript"] = [{"role": "user", "content": "old private turn"}]

    prompt = build_therapeutic_agent_input(
        state,
        include_recent_history=False,
    )

    assert "old private turn" not in prompt
    assert "Recent conversation:\n(no prior history)" in prompt
    assert "Target preview: presentation anxiety note" in prompt
    assert "Triage tentatively suggested 'grounded_lookup'" in prompt


def test_operational_context_omits_inactive_dynamic_sections() -> None:
    context = operational_context_for_prompt(_prompt_state())

    assert "Pending memory deletion exists" not in context
    assert "The user's intent is ambiguous" not in context
    assert "explicitly asked to use prior conversation context" not in context
    assert "None" not in context


def test_operational_context_contract() -> None:
    context = operational_context_for_prompt(_dynamic_state())

    assert len(context) == 859
    assert (
        _digest(context)
        == "e7be3dc04d04c22dd447417ef92976e35e75b8075d8a0c0c0ffbbeb63c4dadba"
    )
    assert "Target preview: presentation anxiety note" in context
    assert "Triage tentatively suggested 'grounded_lookup'" in context
    assert "explicitly asked to use prior conversation context" in context


def test_response_llm_request_contracts_history_boundary() -> None:
    # Both structured fallback and streaming response-LLM paths call this same
    # request builder after orchestration resolves the prompt-history policy.
    state = _prompt_state()
    with_history = build_therapeutic_response_llm_request(
        state, include_recent_history=True
    )
    without_history = build_therapeutic_response_llm_request(
        state, include_recent_history=False
    )

    assert len(with_history.prompt) == 591
    assert (
        _digest(with_history.prompt)
        == "cfe2a79eefce9ff4978e110402ffa6f63eb8b2ab7b3d4585a4cfa50951d21e0b"
    )
    assert len(without_history.prompt) == 531
    assert (
        _digest(without_history.prompt)
        == "4cd0558530425e740924d827dc4ba9a2187ec7f505b4ef837b8085720df74c5f"
    )
    assert "Presentations make me anxious." in with_history.prompt
    assert "Presentations make me anxious." not in without_history.prompt
    assert "Recent conversation:\n(no prior history)" in without_history.prompt
    assert "My sister Maya helps me rehearse." in without_history.prompt
    assert with_history.system_instruction == without_history.system_instruction


def test_response_llm_system_instruction_contract() -> None:
    request = build_therapeutic_response_llm_request(
        _prompt_state(), include_recent_history=True
    )

    assert len(request.system_instruction) == 41047
    assert (
        _digest(request.system_instruction)
        == "8590b5a2fa94d07c0e0c33600ca1aa76c1c0a2236f04ea5f59ccad8590a3742d"
    )
    assert "- response_style: supportive" in request.system_instruction
    assert (
        "- therapeutic_approach: motivational_interviewing"
        in request.system_instruction
    )


def test_response_llm_safety_clarification_contract() -> None:
    request = build_therapeutic_response_llm_request(
        _dynamic_state(), include_recent_history=True
    )

    assert len(request.system_instruction) == 24073
    assert (
        _digest(request.system_instruction)
        == "92a405d270236706ee24d2977a5b695cc75f298c9e8eae5a2433d1100edebf82"
    )
    assert "Safety-check override:" in request.system_instruction
    assert "Include exactly one direct safety question" in request.system_instruction
    assert (
        "Do not provide hotline, 988, emergency-services" in request.system_instruction
    )


def test_style_task_prompt_contracts() -> None:
    state = _prompt_state()
    supportive = build_therapeutic_response_prompt(state, response_style="supportive")
    step = build_therapeutic_response_prompt(
        state,
        response_style="technique",
        step_directive="Guide one slow exhale, then pause for the user response.",
    )

    assert len(supportive) == 537
    assert (
        _digest(supportive)
        == "2df6c4bb529a9dece8fc8ffe6ae571e86daa32ab399a7d9db52f2cdccebb5a70"
    )
    assert len(step) == 610
    assert (
        _digest(step)
        == "4b53867fa0caabf7f25969d076a0ba2674cd5260ed997f624bd1e4a01c7822e5"
    )
    assert "in the supportive response style" in supportive
    assert "Step directive:\nGuide one slow exhale" in step


def test_style_task_prompt_soft_clarification_contract() -> None:
    prompt = build_therapeutic_response_prompt(
        _dynamic_state(), response_style="grounded_lookup"
    )

    assert len(prompt) == 551
    assert (
        _digest(prompt)
        == "e977575bbc06274c89088582491244d2c6a5ef7beb2611e621c0d3cef7ca0812"
    )
    assert "Intent summary: User may want lookup and support." in prompt
    assert "Proceed with the selected action" in prompt
    assert "do not block the response with a question" in prompt
