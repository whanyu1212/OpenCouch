# Slash Command Follow-Up Plan
Date: 2026-05-24
POC: AdaL
TL;DR: After shipping `/summary` and `/export`, the next highest-value native TUI slash commands are deterministic search, developer-facing diagnostics, and small quality-of-life toggles. Build them in small PRs, keep command execution app-native, and continue using shared command metadata for help/completion/model awareness.

## Context
We have now shipped:
- prompt-aware slash command suggestions
- `/summary [short|full]`
- `/export <md|json|txt> [filename]`

We intentionally deferred:
- `/search ...`
- richer debug / workflow commands
- small no-arg toggle polish

The main product direction remains:
- slash commands are native TUI commands
- the LLM can suggest them, but does not parse or execute them
- the command registry remains the source of truth for help, completion, and prompt awareness

## Recommended next commands

### PR 2 — `/search history|memory|all <query>`
Highest-value next feature.

#### User-facing syntax
- `/search history <query>`
- `/search memory <query>`
- `/search all <query>`

#### Why
Users need a way to quickly recover prior discussion or stored memory without manually scanning:
- long active thread recap
- cross-check whether something is already in memory
- debug retrieval behavior

#### V1 scope
Keep the first version deterministic and narrow:
- `history`: search current thread transcript only
- `memory`: search stored memory records only
- `all`: combine both with source labels

#### Implementation notes
- Add metadata in `opencouch_cli/commands.py`
- Add handler in `opencouch_cli/app.py`
- Reuse transcript projection for history search results
- Reuse existing memory listing / retrieval helpers where possible
- Return labeled snippets, not full dumps

#### Output shape
Example:
- `[history] user: I need help with sleep after travel`
- `[memory/fact] #3: user prefers shorter responses`

#### Risks
- weak relevance if naive substring match only
- scope creep into embeddings / semantic retrieval

#### Recommendation
Ship V1 as substring / lightweight fuzzy search first.
Only add semantic retrieval after real usage proves need.

---

### PR 3 — `/logs` and `/doctor` expansion
Developer-facing polish for debugging local runs.

#### Candidate syntax
- `/logs tail [n]`
- `/logs errors`
- `/doctor`
- `/doctor verbose`

#### Why
Useful for local dogfooding and debugging runtime behavior.

#### Recommendation
Keep this behind explicit developer-oriented semantics.
Do not let these commands become a dumping ground for unrelated diagnostics.

---

### PR 4 — Quick toggles / quality-of-life polish
Small but high-feel improvements:
- `/trace` with no args already reports status; consider toggle behavior
- `/theme` with no args could cycle or show current
- `/ui` with no args could toggle compact/full
- `/memory recall` with no arg could show current state or toggle

#### Recommendation
Only add no-arg toggle behavior where it is obvious and low-surprise.

## Not recommended yet
Avoid these until the simpler native commands are proven useful:
- `/rename <name>` (too expensive relative to current value)
- natural-language-to-command execution
- embedding-backed global search
- `/copy` clipboard integration
- broad command-router refactor
- heavy alias expansion in prompt awareness
- command families reorganization (`/thread ...`, `/session ...`) unless command count grows much more

## Architectural rules to keep
1. `SLASH_COMMANDS` stays the single source of truth
2. completion and help should come from metadata, not hand-maintained lists
3. model-visible command awareness should remain generated from the registry
4. command execution stays deterministic in the TUI
5. commands that invoke LLM work should still be parsed natively first

## Suggested delivery order
1. `/search history|memory|all <query>`
2. `/logs ...` / doctor polish
3. small no-arg toggle UX improvements

## Acceptance criteria for PR 2
- `/search history <query>` searches the active thread transcript
- `/search memory <query>` searches stored memory records
- `/search all <query>` combines both with source labels
- autocomplete surfaces nested search subcommands
- help reflects exact syntax
- prompt awareness includes the new commands automatically
- no command is LLM-parsed for execution
