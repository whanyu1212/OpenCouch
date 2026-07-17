"""Characterization contracts for therapeutic prompt products."""

from __future__ import annotations

from hashlib import sha256
from typing import cast

from agent.flows.therapeutic import (
    operational_context_for_prompt,
    therapeutic_response_llm_request_for_state,
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
    # Prompt rendering is deterministic UTF-8 text; CLI commands come from
    # static SLASH_COMMANDS metadata rather than environment-specific values.
    return sha256(text.encode()).hexdigest()


def test_sdk_agent_input_contract_with_history() -> None:
    runtime = OpenAITextRuntime(model="gpt-test")
    prompt = runtime._input_text_for_state(_prompt_state(), include_recent_history=True)  # noqa: SLF001

    assert len(prompt) == 44421
    assert (
        _digest(prompt)
        == "8e1d3b48fb7aef97f292d510025a27480ca3312e76610b77268fac1ab371a977"
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


def test_sdk_agent_input_contract_without_history() -> None:
    runtime = OpenAITextRuntime(model="gpt-test")
    prompt = runtime._input_text_for_state(
        _prompt_state(), include_recent_history=False
    )  # noqa: SLF001

    assert len(prompt) == 44361
    assert (
        _digest(prompt)
        == "bb8f58b13cc33752bcf193819c74dc5559605911ffa7dcf87773121846c8ca2a"
    )
    assert "Presentations make me anxious." not in prompt
    assert "Recent conversation:\n(no prior history)" in prompt
    assert "My sister Maya helps me rehearse." in prompt
    assert "Operational context:" in prompt


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
    # request builder; session history policy is exactly None versus non-None.
    state = _prompt_state()
    with_history = therapeutic_response_llm_request_for_state(state, session=None)
    without_history = therapeutic_response_llm_request_for_state(
        state, session=object()
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
    request = therapeutic_response_llm_request_for_state(_prompt_state(), session=None)

    assert len(request.system_instruction) == 43475
    assert (
        _digest(request.system_instruction)
        == "eacbd0eb8573ef09b3d559631fc9af4017758f14ed25af678b7a8eac4235410e"
    )
    assert "- response_style: supportive" in request.system_instruction
    assert (
        "- therapeutic_approach: motivational_interviewing"
        in request.system_instruction
    )


def test_response_llm_safety_clarification_contract() -> None:
    request = therapeutic_response_llm_request_for_state(_dynamic_state(), session=None)

    assert len(request.system_instruction) == 26501
    assert (
        _digest(request.system_instruction)
        == "413b6cf65f5248479ed7115fa774efad372dedc9ce772abaea731b53176149b4"
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
