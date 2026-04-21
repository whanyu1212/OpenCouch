"""Prompt builders and shared prompt-assembly helpers for the agent graph.

This module owns the shared infrastructure that both ``agent/prompts/crisis.py``
and ``agent/therapeutic/prompts.py`` use to compose system prompts from
markdown source files.

IMPORTANT — definition ordering: ``CORE_SOURCES``, ``load_prompt_source``,
``compose_sources``, and ``format_recent_history`` MUST be defined BEFORE the
``from agent.prompts.crisis import ...`` line below. ``crisis.py`` imports
from this module at load time, creating a circular reference that Python
resolves only because the names it needs are already bound by the time
``crisis.py`` executes its import. Moving the crisis imports above these
definitions will break the import chain.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


# ─── Shared constants ────────────────────────────────────────────────────────
#
# The core prompt sources loaded by every system prompt (crisis, therapeutic,
# all modes). Defined here so crisis.py and therapeutic/prompts.py share a
# single source of truth instead of duplicating the tuple.

CORE_SOURCES = (
    "soul.md",
    "identity.md",
    "policy/boundaries.md",
    "policy/privacy.md",
)


# ─── Shared helpers ──────────────────────────────────────────────────────────


def prompt_sources_root() -> Path:
    """Return the absolute path to the prompt sources directory.

    All prompt-source markdown files live under
    ``apps/backend/agent/prompts/sources/``. This function is the single
    source of truth for that location — both ``agent/prompts/crisis.py``
    and ``agent/therapeutic/prompts.py`` import it instead of computing
    the root independently.
    """

    return Path(__file__).resolve().parent / "sources"


@lru_cache(maxsize=64)
def load_prompt_source(relative_path: str) -> str:
    """Load one prompt-source markdown file by its path relative to the sources root.

    Raises ValueError if the resolved path escapes the sources root
    (directory traversal protection).
    """

    root = prompt_sources_root().resolve()
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path.read_text(encoding="utf-8").strip()


def compose_sources(*relative_paths: str) -> str:
    """Concatenate prompt source files into a single prompt block."""

    parts = [load_prompt_source(path) for path in relative_paths]
    return "\n\n".join(part for part in parts if part)


def format_recent_history(state: dict[str, Any], *, limit: int = 6) -> str:
    """Format recent history entries for prompt injection.

    Used by both crisis and therapeutic prompt builders to inject a
    window of recent conversation into the user/task prompt.
    """

    history = state.get("history", [])[-limit:]
    if not history:
        return "(no prior history)"

    return "\n".join(
        f"{turn.get('role', 'unknown')}: {turn.get('content', '').strip()}"
        for turn in history
        if turn.get("content")
    )


# ─── Submodule re-exports (MUST come after shared definitions) ─────��─────────

from agent.prompts.crisis import (  # noqa: E402
    build_crisis_classifier_prompt,
    build_crisis_classifier_system_prompt,
    build_crisis_response_prompt,
    build_crisis_response_system_prompt,
)

__all__ = [
    # Shared
    "CORE_SOURCES",
    "prompt_sources_root",
    "load_prompt_source",
    "compose_sources",
    "format_recent_history",
    # Crisis builders
    "build_crisis_classifier_prompt",
    "build_crisis_classifier_system_prompt",
    "build_crisis_response_prompt",
    "build_crisis_response_system_prompt",
]
