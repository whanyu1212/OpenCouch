"""Regression tests for memory-layer dependency direction."""

from __future__ import annotations

import ast
from pathlib import Path


_MEMORY_ROOT = Path(__file__).resolve().parents[3] / "agent" / "memory"
_FORBIDDEN_IMPORTS = ("agent.runtime", "agent.state")


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return [
        imported
        for imported in imports
        if any(
            imported == forbidden or imported.startswith(f"{forbidden}.")
            for forbidden in _FORBIDDEN_IMPORTS
        )
    ]


def test_memory_does_not_import_runtime_or_runtime_state() -> None:
    violations = {
        path.relative_to(_MEMORY_ROOT): _forbidden_imports(path)
        for path in sorted(_MEMORY_ROOT.rglob("*.py"))
        if not path.name.startswith("._") and _forbidden_imports(path)
    }

    assert not violations, f"memory dependency-direction violations: {violations}"
