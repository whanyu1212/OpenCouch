"""Generic system-prompt composition helpers for response modes and modalities."""

from __future__ import annotations

from agent.prompts.catalog import (
    ALLOWED_MODALITIES,
    MODE_FILES,
    MODALITY_FILES,
    Modality,
    ResponseMode,
)
from agent.prompts.core import build_core_system_prompt
from agent.prompts.loader import compose_knowledge_sections


def _validate_modalities(mode: ResponseMode, modalities: tuple[Modality, ...]) -> None:
    """Validate that the selected modalities are allowed for the chosen mode."""

    allowed = set(ALLOWED_MODALITIES[mode])
    invalid = [modality for modality in modalities if modality not in allowed]
    if invalid:
        raise ValueError(
            f"Modalities {invalid} are not allowed for mode '{mode}'. "
            f"Allowed modalities: {sorted(allowed)}"
        )


def build_mode_prompt(mode: ResponseMode) -> str:
    """Build the knowledge-backed prompt fragment for a response mode.

    Args:
        mode: Response mode whose knowledge files should be composed.

    Returns:
        The composed prompt fragment for the selected response mode.
    """

    return compose_knowledge_sections(*MODE_FILES[mode])


def build_modality_prompt(*modalities: Modality) -> str:
    """Build the combined knowledge-backed prompt fragment for selected modalities.

    Args:
        *modalities: Modality overlays to compose into one prompt fragment.

    Returns:
        The combined prompt fragment for the selected modalities.
    """

    paths: list[str] = []
    for modality in modalities:
        paths.extend(MODALITY_FILES[modality])
    return compose_knowledge_sections(*paths)


def build_system_prompt(
    *,
    mode: ResponseMode,
    modalities: tuple[Modality, ...] = (),
) -> str:
    """Build a full system prompt from the core layer plus mode/modality overlays.

    Args:
        mode: Response mode for the system prompt.
        modalities: Optional modality overlays to append.

    Returns:
        The complete system prompt for the requested mode and modalities.
    """

    _validate_modalities(mode, modalities)

    parts = [
        build_core_system_prompt(),
        build_mode_prompt(mode),
    ]
    if modalities:
        parts.append(build_modality_prompt(*modalities))

    return "\n\n".join(part for part in parts if part)
