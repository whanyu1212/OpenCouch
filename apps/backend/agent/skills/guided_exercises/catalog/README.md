# Guided exercise skill docs

This catalog contains OpenAI/Anthropic-style `SKILL.md` documentation for guided exercises.

For now, the Python `ExerciseDefinition` registry remains the runtime source of truth for exercise selection, steps, channels, and state transitions. These docs provide a standards-aligned packaging layer for reviewability and future prompt rendering. Tests validate that each documented skill stays compatible with the Python registry.

Each skill directory should contain exactly one `SKILL.md` file with frontmatter including at least:

- `name`: must match a registered exercise id
- `description`: concise routing/use description
- `version`: should match the registry definition version
- `category`: should match the registry category
- `channels`: supported delivery channels
