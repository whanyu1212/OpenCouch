"""Shared tool-forcing prompt directive for single-required-tool prompts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def force_tool_directive(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Return the shared "call exactly one required tool" directive.

    The returned block ends with a trailing space so callers can append
    their route-specific answering instructions directly. ``arguments`` is
    rendered with sorted keys for deterministic output.
    """

    rendered_arguments = json.dumps(dict(arguments), sort_keys=True)
    return (
        f"Required tool: {tool_name}\n"
        f"Required tool arguments: {rendered_arguments}\n"
        "Call the required tool exactly once before answering. "
    )


__all__ = ["force_tool_directive"]
