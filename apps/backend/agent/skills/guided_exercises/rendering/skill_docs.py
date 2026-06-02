"""Filesystem skill-doc helpers for guided exercises.

The Python exercise registry remains the runtime source of truth. These helpers
validate standards-aligned ``SKILL.md`` documentation against that registry so
exercise content can move toward OpenAI/Anthropic skill packaging without
changing guided-exercise execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.skills.guided_exercises.types import ExerciseDefinition

_CATALOG_DIR = Path(__file__).resolve().parents[1] / "catalog"
_SKILL_DOC_FILENAME = "SKILL.md"
_VALID_CHANNELS = frozenset({"text", "voice"})


@dataclass(frozen=True)
class GuidedExerciseSkillDoc:
    """Parsed guided-exercise skill document."""

    name: str
    description: str
    version: int | None
    category: str | None
    channels: tuple[str, ...]
    path: Path
    body: str


def iter_guided_exercise_skill_docs(
    *,
    catalog_dir: Path = _CATALOG_DIR,
) -> tuple[GuidedExerciseSkillDoc, ...]:
    """Return parsed guided-exercise skill docs in directory order."""

    if not catalog_dir.exists():
        return ()

    docs: list[GuidedExerciseSkillDoc] = []
    for skill_dir in sorted(path for path in catalog_dir.iterdir() if path.is_dir()):
        skill_doc_path = skill_dir / _SKILL_DOC_FILENAME
        if not skill_doc_path.exists():
            continue
        docs.append(_parse_skill_doc(skill_doc_path))
    return tuple(docs)


def get_guided_exercise_skill_doc(
    exercise_type: str,
    *,
    catalog_dir: Path = _CATALOG_DIR,
) -> GuidedExerciseSkillDoc | None:
    """Return a parsed skill doc by exercise id, when one exists."""

    for doc in iter_guided_exercise_skill_docs(catalog_dir=catalog_dir):
        if doc.name == exercise_type:
            return doc
    return None


def validate_guided_exercise_skill_docs(
    definitions: tuple[ExerciseDefinition, ...],
    *,
    catalog_dir: Path = _CATALOG_DIR,
) -> None:
    """Validate filesystem skill docs against runtime exercise definitions."""

    definitions_by_id = {definition.id: definition for definition in definitions}
    seen_names: set[str] = set()
    for doc in iter_guided_exercise_skill_docs(catalog_dir=catalog_dir):
        if doc.name in seen_names:
            raise ValueError(f"Duplicate guided exercise skill doc name {doc.name!r}.")
        seen_names.add(doc.name)

        definition = definitions_by_id.get(doc.name)
        if definition is None:
            raise ValueError(
                f"Guided exercise skill doc {doc.path} references unknown "
                f"exercise {doc.name!r}."
            )

        if doc.version is not None and doc.version != definition.version:
            raise ValueError(
                f"Guided exercise skill doc {doc.name!r} version {doc.version} "
                f"does not match registry version {definition.version}."
            )

        if doc.category is not None and doc.category != definition.category:
            raise ValueError(
                f"Guided exercise skill doc {doc.name!r} category {doc.category!r} "
                f"does not match registry category {definition.category!r}."
            )

        invalid_channels = set(doc.channels) - _VALID_CHANNELS
        if invalid_channels:
            raise ValueError(
                f"Guided exercise skill doc {doc.name!r} has unsupported channels "
                f"{sorted(invalid_channels)!r}."
            )

        registry_channels = set(definition.channels)
        if definition.voice_supported:
            registry_channels.add("voice")
        if doc.channels and not set(doc.channels).issubset(registry_channels):
            raise ValueError(
                f"Guided exercise skill doc {doc.name!r} channels {doc.channels!r} "
                f"are not compatible with registry channels "
                f"{tuple(sorted(registry_channels))!r}."
            )


def _parse_skill_doc(path: Path) -> GuidedExerciseSkillDoc:
    raw_text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(raw_text, path=path)
    metadata = _parse_frontmatter(frontmatter, path=path)

    name = _required_string(metadata, "name", path=path)
    description = _required_string(metadata, "description", path=path)
    version = _optional_int(metadata, "version", path=path)
    category = _optional_string(metadata, "category", path=path)
    channels = _optional_string_list(metadata, "channels", path=path)

    return GuidedExerciseSkillDoc(
        name=name,
        description=description,
        version=version,
        category=category,
        channels=channels,
        path=path,
        body=body.strip(),
    )


def _split_frontmatter(raw_text: str, *, path: Path) -> tuple[str, str]:
    if not raw_text.startswith("---\n"):
        raise ValueError(f"Guided exercise skill doc {path} is missing frontmatter.")

    try:
        _, frontmatter, body = raw_text.split("---\n", 2)
    except ValueError as exc:
        raise ValueError(
            f"Guided exercise skill doc {path} has invalid frontmatter."
        ) from exc

    if not frontmatter.strip():
        raise ValueError(f"Guided exercise skill doc {path} has empty frontmatter.")
    if not body.strip():
        raise ValueError(f"Guided exercise skill doc {path} has empty body.")
    return frontmatter, body


def _parse_frontmatter(frontmatter: str, *, path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in frontmatter.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("- "):
            if current_list_key is None:
                raise ValueError(
                    f"Guided exercise skill doc {path} has list item without key."
                )
            values.setdefault(current_list_key, []).append(line[2:].strip())
            continue

        current_list_key = None
        if ":" not in line:
            raise ValueError(
                f"Guided exercise skill doc {path} has invalid frontmatter line "
                f"{raw_line!r}."
            )

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Guided exercise skill doc {path} has empty key.")

        if value:
            values[key] = _unquote(value)
        else:
            values[key] = []
            current_list_key = key

    return values


def _required_string(metadata: dict[str, Any], key: str, *, path: Path) -> str:
    value = _optional_string(metadata, key, path=path)
    if value is None:
        raise ValueError(f"Guided exercise skill doc {path} requires {key!r}.")
    return value


def _optional_string(metadata: dict[str, Any], key: str, *, path: Path) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Guided exercise skill doc {path} field {key!r} must be a string."
        )
    return value.strip()


def _optional_int(metadata: dict[str, Any], key: str, *, path: Path) -> int | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Guided exercise skill doc {path} field {key!r} must be an integer."
        ) from exc


def _optional_string_list(
    metadata: dict[str, Any],
    key: str,
    *,
    path: Path,
) -> tuple[str, ...]:
    value = metadata.get(key)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(
            f"Guided exercise skill doc {path} field {key!r} must be a list."
        )
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Guided exercise skill doc {path} field {key!r} has invalid item."
            )
        items.append(item.strip())
    return tuple(items)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


__all__ = [
    "GuidedExerciseSkillDoc",
    "get_guided_exercise_skill_doc",
    "iter_guided_exercise_skill_docs",
    "validate_guided_exercise_skill_docs",
]
