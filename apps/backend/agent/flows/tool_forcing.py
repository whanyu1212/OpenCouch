"""Shared tool-forcing prompt directive for single-required-tool routes.

Several routes (crisis resource lookup, grounded lookup, guided-exercise skill
load) force the model to call exactly one tool before answering. They each
emitted the same three-line directive inline, with one latent inconsistency:
crisis hardcoded ``Required tool arguments: {}`` while the others rendered
arguments via ``json.dumps``. Centralizing the directive removes the
duplication and normalizes the empty-arguments rendering (``json.dumps({})``
also yields ``{}``), without changing any route's emitted text.

The directive is deliberately scoped to the *shared* skeleton only — each
route appends its own answering instructions, since those genuinely differ
(crisis prioritizes safety; grounded restricts to ``response_text``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def force_tool_directive(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Return the shared "call exactly one required tool" directive.

    The returned block ends with a trailing newline so callers can append
    their route-specific answering instructions directly. ``arguments`` is
    rendered with sorted keys for deterministic output.
    """

    rendered_arguments = json.dumps(dict(arguments), sort_keys=True)
    return (
        f"Required tool: {tool_name}\n"
        f"Required tool arguments: {rendered_arguments}\n"
        "Call the required tool exactly once before answering. "
    )
