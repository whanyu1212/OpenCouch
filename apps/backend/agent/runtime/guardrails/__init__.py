"""Runtime guardrails for OpenAI text-agent execution."""

from agent.runtime.guardrails.assessment import (
    CrisisGateResult,
    assess_crisis_gate,
    build_crisis_gate_delta,
)
from agent.runtime.guardrails.crisis import (
    CrisisInputGuardrailOutput,
    crisis_input_guardrail,
    run_crisis_input_guardrail,
)
from agent.runtime.guardrails.prompts import (
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)
from agent.runtime.guardrails.service import (
    CrisisAssessmentSchema,
    CrisisRiskResult,
    CrisisRiskService,
    assess_crisis_risk_with_llm,
    enforce_crisis_truth_table,
)

__all__ = [
    "CrisisAssessmentSchema",
    "CrisisGateResult",
    "CrisisInputGuardrailOutput",
    "CrisisRiskResult",
    "CrisisRiskService",
    "assess_crisis_gate",
    "assess_crisis_risk_with_llm",
    "build_crisis_classifier_prompt",
    "build_crisis_classifier_system_prompt",
    "build_crisis_gate_delta",
    "build_crisis_response_prompt",
    "build_crisis_response_system_prompt",
    "crisis_input_guardrail",
    "enforce_crisis_truth_table",
    "run_crisis_input_guardrail",
]
