"""Guard docs and eval labels against stale runtime ownership wording."""

from __future__ import annotations

from pathlib import Path

import pytest


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "apps").is_dir() and (parent / "eval").is_dir():
            return parent
    raise AssertionError("Could not locate repository root")


STALE_PHRASES = {
    "eval/README.md": (
        "save-preference and forget-by-query dispatch",
        "save-preference dispatch",
        "forget-by-query dispatch",
    ),
    "eval/datasets/behavior_matrix.jsonl": (
        "memory_save_preference_via_triage_dispatch",
        "memory_forget_by_query_dispatch",
    ),
    "apps/backend/agent/README.md": (
        "agent.runtime.text.OpenAITextRuntime",
        "agent/runtime/text.py",
        "agent/runtime/agents/",
        "agent/runtime/tools/",
    ),
    "apps/docs/docs/agent/graph.mdx": (
        "## Therapeutic subgraph",
        "The dispatcher picks exactly one therapeutic response style per turn",
        "The dispatcher is **LLM-owned**",
        "agent/runtime/text.py",
        "agent/runtime/agents/",
        "agent/runtime/tools/",
        "agent/memory/load_turn.py",
        "agent/runtime/guardrails/",
        "agent/runtime/tools/crisis.py",
        "agent/runtime/tools/grounded.py",
    ),
    "apps/docs/docs/intro.md": (
        "therapeutic subgraph",
        "memory_control_node",
        "turn_dispatch_node",
        "grounded_answer_node",
        "crisis_gate_node",
        "finalize_turn_node",
        "Graph nodes and runtime services",
    ),
    "apps/docs/docs/backend/overview.mdx": (
        "Explicit LangGraph text-agent graph",
        "first node",
        "graph nodes in identical order",
        "agent/runtime/tools/crisis.py",
        "agent/runtime/tools/grounded.py",
        "agent/gates/memory_control/",
    ),
    "apps/docs/docs/backend/runtime.mdx": (
        "Both layers run identical graph nodes in identical order.",
        "finalize_turn_node",
        "Graph nodes and runtime side effects",
        "agent/graph.py",
        "agent/persistence.py",
        "agent/audit/session_feedback.py",
        "agent/audit/postgres_session_feedback.py",
        "agent/audit/sqlite_session_feedback.py",
    ),
    "apps/docs/docs/memory/privacy.mdx": (
        "Memory control runs in the graph",
        "first-class graph traffic",
        "turn_dispatch_node",
        "therapeutic subgraph",
        "memory_control_node",
        "same dispatch node",
        "Both subsystems live under `agent/audit/`",
    ),
    "apps/docs/docs/agent/state.mdx": (
        "flows through every node",
        "Nodes read from it",
        "internal to the graph",
        "finalize_turn_node",
        "therapeutic subgraph completes",
        "agent/graph.py",
    ),
    "apps/docs/src/components/StateFields.tsx": (
        "resolved into prompt behavior by the graph",
        "guided_exercise_node",
        "memory_control_node",
        "crisis_gate_node",
        "crisis_log_node",
        "therapeutic_dispatch_node",
        "turn_dispatch_node",
        "grounded_answer_node",
        "crisis_resource_lookup_node",
        "Whichever node wins the route",
        "graph nodes + runtime",
    ),
    "apps/docs/src/components/AgentGraph.tsx": (
        "Therapeutic subgraph expansion",
        "Therapeutic responses — dispatcher picks exactly one per turn",
        "The runtime owns dispatch",
        "LLM dispatcher",
    ),
    "apps/docs/src/components/GraphTopology.tsx": (
        "therapeutic_subgraph active",
        "graph END",
    ),
    "apps/docs/src/components/ScenarioReplay.tsx": (
        "therapeutic subgraph nodes",
        "therapeutic_subgraph",
        "crisis_resource_lookup_node",
        "crisis_response_node",
        "crisis_log_node",
        "dispatcher's decision rule",
    ),
    "apps/docs/src/components/TherapyApproach.tsx": (
        "dispatched per turn by the subgraph",
        "therapeutic subgraph",
        "LLM dispatcher",
    ),
    "apps/docs/src/components/TurnPipeline.tsx": (
        "crisis_gate_node",
        "crisis_resource_lookup_node",
        "crisis_response_node",
        "crisis_log_node",
        "turn_dispatch_node",
        "memory_control_node",
        "grounded_answer_node",
        "therapeutic_subgraph",
        "finalize_turn_node",
        "runtime side effects after graph END",
        "Every I/O node",
    ),
    "apps/docs/docs/agent/context-management.mdx": (
        "agent/runtime/text.py",
        "crisis_gate_node",
        "wired into graph state",
        "runtime node",
        "separate therapeutic dispatcher",
    ),
    "apps/docs/docs/agent/tools.mdx": (
        "agent/runtime/text.py",
        "agent/runtime/tools/",
        "node-invoked",
        "graph-registered",
    ),
    "apps/docs/docs/agent/prompt-assembly.mdx": (
        "selected at runtime by the graph",
        "agent/runtime/agents/",
        "agent/runtime/guardrails/",
        "agent/memory/load_turn.py",
        "wired into the graph",
    ),
    "apps/docs/docs/memory/overview.mdx": (
        "After the graph reaches `END`",
        "agent/audit/session_feedback.py",
        "agent/memory/load_turn.py",
        "agent/persistence.py",
        "### Graph integration",
    ),
    "apps/docs/docs/observability/session-feedback.md": (
        "agent/audit/session_feedback.py",
        "agent/audit/postgres_session_feedback.py",
        "agent/audit/sqlite_session_feedback.py",
        "agent/persistence.py",
    ),
    "apps/docs/docs/observability/overview.md": (
        "graph END",
        "graph nodes",
        "graph state",
        "finalize_turn_node",
    ),
    "apps/docs/docs/philosophy/crisis-gate.mdx": (
        "agent/runtime/guardrails/",
        "agent/runtime/tools/",
    ),
    "apps/docs/docs/philosophy/approach.mdx": (
        "first node in the graph",
        "therapeutic subgraph",
        "LLM dispatcher",
    ),
    "apps/docs/docs/philosophy/graph-vs-react.mdx": (
        "compiled directed graph",
        "LangGraph",
        "Compiled graph",
        "agent/gates/",
        "crisis_gate_node",
        "finalize_turn_node",
        "graph END",
    ),
    "apps/docs/src/components/ToolRegistry.tsx": (
        "agent/runtime/tools/",
        "grounded_answer_node",
        "turn_dispatch_node",
        "node-invoked",
        "therapeutic subgraph",
    ),
    "apps/docs/src/components/PromptLayerStack.tsx": (
        "runtime/agents/",
        "graph-selected",
        "node-specific",
        "graph state",
    ),
    "apps/backend/agent/memory/README.md": ("agent/runtime/text.py",),
    "eval/runners/run_crisis_template_eval.py": ("agent.runtime.tools",),
    "eval/runners/run_crisis_resource_eval.py": ("agent.runtime.tools",),
}


@pytest.mark.parametrize(
    ("relative_path", "stale_phrases"),
    STALE_PHRASES.items(),
)
def test_docs_and_evals_use_current_runtime_vocabulary(
    relative_path: str,
    stale_phrases: tuple[str, ...],
) -> None:
    text = (_repo_root() / relative_path).read_text(encoding="utf-8")

    found = [phrase for phrase in stale_phrases if phrase in text]

    assert found == []
