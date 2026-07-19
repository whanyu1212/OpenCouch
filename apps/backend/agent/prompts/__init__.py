"""Shared prompt-source loading and formatting utilities.

This package is the home of the *shared* prompt infrastructure used by
every prompt builder in the codebase: the markdown source loader, the
composition helper, the recent-history formatter, and the ``CORE_SOURCES``
tuple referenced by every system prompt. Feature-specific builders
(therapeutic responses, crisis replies, grounded lookup) live next to
their consumers — not here.

The ``sources/`` subdirectory contains the markdown knowledge files
that ``load_prompt_source`` reads. It is loaded relative to this
package's ``__file__`` so that builders importing from any consumer
module resolve to the same on-disk knowledge base.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.runtime.session.state import format_recent_history as _format_recent_history

CORE_SOURCES = (
    "core_identity.md",
    "policy/boundaries.md",
    "policy/privacy.md",
)


def prompt_sources_root() -> Path:
    """Return the absolute path to the prompt sources directory.

    Returns:
        Absolute prompt source directory path.
    """

    return Path(__file__).resolve().parent / "sources"


@lru_cache(maxsize=64)
def load_prompt_source(relative_path: str) -> str:
    """Load one prompt-source markdown file by relative path.

    Args:
        relative_path: Path under the prompt sources root.

    Returns:
        File contents with leading and trailing whitespace removed.
    """

    root = prompt_sources_root().resolve()
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path.read_text(encoding="utf-8").strip()


def compose_sources(*relative_paths: str) -> str:
    """Concatenate prompt source files into a single prompt block.

    Args:
        relative_paths: Prompt source paths under the sources root.

    Returns:
        Combined prompt text separated by blank lines.
    """

    parts = [load_prompt_source(path) for path in relative_paths]
    return "\n\n".join(part for part in parts if part)


def format_recent_history(state: Mapping[str, Any], *, limit: int = 6) -> str:
    """Format recent history entries for prompt injection.

    Args:
        state: Current runtime state or state-like mapping.
        limit: Maximum number of recent history turns to include.

    Returns:
        Formatted conversation history, or a no-history placeholder.
    """

    return _format_recent_history(state, limit=limit)


__all__ = [
    "CORE_SOURCES",
    "prompt_sources_root",
    "load_prompt_source",
    "compose_sources",
    "format_recent_history",
]
