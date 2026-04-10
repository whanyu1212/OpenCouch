"""System prompt builders for therapeutic response modes.

Each mode gets a system prompt composed from the repo-level ``knowledge/``
markdown files (identity, boundaries, privacy) plus mode-specific guidance.
The composition pattern mirrors ``agent/prompts/crisis.py`` — load files,
concatenate, return as one string.

Phase 1 v0.1 scope: three modes (supportive, reflective, clarifying).
The other three (psychoeducation, guided_exercise, closing) land in v0.6.
Modality overlays (CBT, ACT, PFA, etc.) are deferred to phase 2.

The prompt builders also provide ``format_recent_history`` and
``build_therapeutic_response_prompt`` helpers that all mode nodes share
for constructing their user/task prompts.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from agent.state import AgentState


# ─── Knowledge file composition (same pattern as prompts/crisis.py) ─────────

_CORE_KNOWLEDGE = (
    "soul.md",
    "identity.md",
    "policy/boundaries.md",
    "policy/privacy.md",
)

_SUPPORTIVE_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "response_modes/support.md",
    "modalities/motivational_interviewing.md",
)

_REFLECTIVE_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    "response_modes/reflection.md",
)

_CLARIFYING_KNOWLEDGE = (
    *_CORE_KNOWLEDGE,
    # No mode-specific knowledge file — the clarifying prompt is
    # self-contained in the mode instructions below.
)


def _knowledge_root() -> Path:
    """Return the absolute path to the repo-level ``knowledge/`` directory."""

    return Path(__file__).resolve().parents[4] / "knowledge"


@lru_cache(maxsize=32)
def _load_knowledge_file(relative_path: str) -> str:
    """Load one markdown knowledge file by its relative path."""

    root = _knowledge_root().resolve()
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path.read_text(encoding="utf-8").strip()


def _compose(*relative_paths: str) -> str:
    """Concatenate knowledge files into a single prompt block."""

    parts = [_load_knowledge_file(path) for path in relative_paths]
    return "\n\n".join(part for part in parts if part)


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _format_recent_history(state: AgentState, *, limit: int = 6) -> str:
    """Format recent history entries for prompt injection."""

    history = state["history"][-limit:]
    if not history:
        return "(no prior history)"

    return "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
        for turn in history
        if turn.get("content")
    )


def _format_working_memory(state: AgentState) -> str:
    """Format working memory snippets for prompt injection.

    If there is no working memory (incognito mode, or no facts extracted
    yet), returns an empty string so the prompt section is simply absent
    rather than showing an empty list.
    """

    snippets = state.get("working_memory", [])
    if not snippets:
        return ""
    joined = "\n".join(f"- {s}" for s in snippets)
    return f"\nRelevant context from past sessions:\n{joined}\n"


# ─── Mode instructions (appended to the system prompt) ──────────────────────
#
# These are the mode-specific behavioral instructions that shape HOW the
# agent responds. They sit on top of the knowledge-file composition so
# the agent has both the foundational identity/boundaries AND the
# mode-specific guidance.

_SUPPORTIVE_INSTRUCTIONS = """
You are in SUPPORTIVE mode. Your job is to listen well, validate the
user's feelings, and leave room for them to continue sharing.

Guidelines:
- Be warm but not effusive. Match the user's energy.
- Validate the feeling before offering any reflection.
- Keep your response short: 2-4 sentences, rarely more.
- Light-touch reflection: name the feeling, don't analyze it.
- Do not ask more than one question. Often, no question is best.
- Never start with "I understand" — it sounds hollow from an AI.
- If working memory contains relevant past context, let it inform
  your warmth and pacing, but do NOT explicitly reference past
  sessions unless the user asks about them.
""".strip()

_REFLECTIVE_INSTRUCTIONS = """
You are in REFLECTIVE mode. The user seems to be noticing a pattern
or asking a "why does this keep happening?" type of question. Your job
is to gently name the pattern and invite the user to reflect on it.

Guidelines:
- Name ONE pattern, not several. Focus matters.
- Ground the naming in the user's own words when possible.
  ("I notice you keep saying 'I should'...")
- Invite reflection with one open question at most.
- Acknowledge the observation might be wrong: "Does that resonate,
  or is it more like...?"
- Slightly longer than supportive mode: 3-5 sentences.
- Never introduce a pattern the user hasn't shown evidence for.
  Hallucinated patterns are the single worst failure mode.
- If the user asked a "why" question, offer a reflection, not a
  diagnosis or explanation. Explanations belong to psychoeducation.
""".strip()

_CLARIFYING_INSTRUCTIONS = """
You are in CLARIFYING mode. The user's message is too short, too
ambiguous, or too out-of-context to respond to well. Your job is to
ask ONE focused question to get the information you need.

Guidelines:
- Acknowledge what you heard first: "It sounds like something's on
  your mind..."
- Ask exactly ONE question, not a list.
- The question should be open-ended, not yes/no.
- Keep it very short: 2-3 sentences total.
- Never say "Can you tell me more?" — that's too generic. Ask
  something SPECIFIC about what the user hinted at.
- The question should be about CONTEXT, not CONTENT. "What brought
  this up?" is better than "What do you mean?"
""".strip()


# ─── Public prompt builders ──────────────────────────────────────────────────


def build_supportive_system_prompt() -> str:
    """Build the system prompt for supportive-mode responses."""

    knowledge = _compose(*_SUPPORTIVE_KNOWLEDGE)
    return f"{knowledge}\n\n{_SUPPORTIVE_INSTRUCTIONS}"


def build_reflective_system_prompt() -> str:
    """Build the system prompt for reflective-mode responses."""

    knowledge = _compose(*_REFLECTIVE_KNOWLEDGE)
    return f"{knowledge}\n\n{_REFLECTIVE_INSTRUCTIONS}"


def build_clarifying_system_prompt() -> str:
    """Build the system prompt for clarifying-mode responses."""

    knowledge = _compose(*_CLARIFYING_KNOWLEDGE)
    return f"{knowledge}\n\n{_CLARIFYING_INSTRUCTIONS}"


def build_therapeutic_response_prompt(state: AgentState, *, mode: str) -> str:
    """Build the user/task prompt for any therapeutic mode.

    All three modes share the same user-prompt structure — the system
    prompt (which differs per mode) is what shapes the response
    character. The user prompt provides the conversation context and
    current message.

    Args:
        state: Current graph state with history and working memory.
        mode: The dispatched mode name, injected as context for
            observability in the prompt.
    """

    memory_block = _format_working_memory(state)

    return (
        f"Write the next assistant message for a mental health support "
        f"conversation in {mode} mode.\n\n"
        f"Recent conversation:\n{_format_recent_history(state)}\n"
        f"{memory_block}\n"
        f"Current user message:\nuser: {state['message']}"
    )
