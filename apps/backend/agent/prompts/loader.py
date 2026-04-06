"""Load and compose markdown prompt knowledge from the repo-level `knowledge/` tree."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def get_knowledge_root() -> Path:
    """Return the absolute path to the repo-level knowledge directory.

    Returns:
        The absolute path to the `knowledge/` directory.
    """

    return Path(__file__).resolve().parents[4] / "knowledge"


def _resolve_knowledge_path(relative_path: str) -> Path:
    """Resolve a relative knowledge path and ensure it stays within `knowledge/`."""

    root = get_knowledge_root().resolve()
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path


@lru_cache(maxsize=128)
def load_knowledge_file(relative_path: str) -> str:
    """Load one markdown file from the knowledge directory.

    Args:
        relative_path: Relative path under the repo-level `knowledge/` directory.

    Returns:
        The stripped markdown contents of the requested file.
    """

    path = _resolve_knowledge_path(relative_path)
    return path.read_text(encoding="utf-8").strip()


def compose_knowledge_sections(*relative_paths: str) -> str:
    """Compose multiple knowledge files into one prompt section.

    Args:
        *relative_paths: Relative knowledge-file paths to concatenate.

    Returns:
        One combined prompt section built from the requested files.
    """

    parts = [load_knowledge_file(path) for path in relative_paths]
    return "\n\n".join(part for part in parts if part)
