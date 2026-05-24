# Slash Command Spec — `/search history|memory|all <query>`
Date: 2026-05-24
POC: AdaL
TL;DR: Implement `/search` as a deterministic native TUI command with three subcommands: `history`, `memory`, and `all`. V1 should use substring / lightweight fuzzy matching only, return labeled snippets, and reuse existing transcript and memory helpers. Do not introduce embeddings, broad retrieval refactors, or LLM-parsed execution in this phase.

## Goal
Add a high-value native search command that helps users quickly recover:
- prior discussion in the active thread
- persisted memory entries
- both combined in one labeled result set

This should improve long-session usability without changing the current architectural model:
- slash commands stay app-native
- help/completion/model awareness stay registry-driven
- the LLM may suggest `/search`, but does not execute it

## User-facing command surface

### Public commands
- `/search history <query>`
- `/search memory <query>`
- `/search all <query>`

### Optional future expansion, not in V1
- `/search history --limit 10 <query>`
- `/search thread <thread-id> <query>`
- `/search semantic <query>` via embeddings
- `/search sessions <query>` across past threads

## Why this command next
This is the best remaining UX gain after `/summary` and `/export` because it supports:
- returning to earlier parts of a long active conversation
- checking whether something is already stored in memory
- debugging whether memory extraction / retrieval is working
- reducing friction before using `/summary` or `/export`

## V1 product behavior

### `/search history <query>`
Search only the active thread transcript.

#### Data source
Use the exact existing active-thread transcript path:
- `runtime.get_state(session.thread_id)`
- `get_transcript(...)`
- `session_conversation_from_transcript(...)`

This keeps `/search history` aligned with the same canonical conversation projection already used by transcript-backed CLI features.

#### Matching strategy
- case-insensitive substring match on visible text
- optional lightweight ranking boost for earlier exact phrase matches
- no embeddings
- no LLM

#### Result format
Return a compact, labeled list. Example:

- `[history] user: I need help with sleep after travel`
- `[history] assistant: Try anchoring wake time first`

#### Limits
- default top 8 matches
- if more exist, show “and N more matches” footer rather than dumping everything

---

### `/search memory <query>`
Search only persisted memory content for the active owner scope.

#### Scope
Search the same memory scope the CLI already uses:
- user-scoped when `user_id` is present
- otherwise thread-scoped

#### Data sources
Use the exact existing helper paths already used by the CLI:
- semantic facts: `_collect_records_by_kind(..., kind="semantic", owner_id=session.owner_id())`
- episodic session arcs: `_collect_records_by_kind(..., kind="episodic", owner_id=session.owner_id())`
- procedural rules: `aget_procedural_profile(runtime.memory_store, user_id=session.owner_id())`

This matters because procedural memory is stored as a single profile document per owner, not as individual records like semantic and episodic memory.

#### Search targets
V1 should cover:
- semantic facts
- episodic session arcs
- procedural rules

#### Matching strategy
- case-insensitive substring search over meaningful text fields
- no embeddings in V1
- no semantic ranking in V1

#### Output format
Results must be source-labeled, concise, and human-readable. Example:

- `[memory/fact] #3: user prefers shorter responses`
- `[memory/rule] #1: don't suggest meditation again`
- `[memory/session] #2: difficult week after work conflict`

#### Important constraint
Do not dump full raw stored objects in V1.
Return short snippets only.

---

### `/search all <query>`
Search both transcript history and memory, then show a merged result list.

#### Behavior
- run history search
- run memory search
- combine results into one ordered list
- keep source labels visible

#### Ordering
V1 recommendation:
1. exact or strong history matches first
2. exact or strong memory matches next
3. preserve deterministic ordering within each bucket

Alternative acceptable behavior:
- history results first, then memory results

Do not overengineer ranking in V1.

## Input validation

### Required query
All search commands require a non-empty query.

#### Invalid examples
- `/search`
- `/search history`
- `/search memory`
- `/search all`

#### Error behavior
Show exact usage guidance:
- `Usage: /search history <query>`
- `Usage: /search memory <query>`
- `Usage: /search all <query>`

### Unknown subcommands
If the user types:
- `/search foo hello`

Return:
- `Unknown /search subcommand. Available: history, memory, all`

## Command metadata plan

## `opencouch_cli/commands.py`
Add:

- `("/search",)` with usage `/search <history|memory|all> <query>`
- `("/search", "history")`
- `("/search", "memory")`
- `("/search", "all")`

### Category
Use `session` or `memory`.

Recommendation:
- use `session` for the top-level `/search`
- child entries can also remain `session` for simplicity

Reason:
This command spans both transcript and memory, and is user workflow-oriented rather than a pure memory-management operation.

### Help text
Suggested metadata:

- `/search <history|memory|all> <query>` — Search the active transcript, stored memory, or both
- `/search history <query>` — Search the active thread transcript
- `/search memory <query>` — Search stored memory for the active owner scope
- `/search all <query>` — Search both transcript history and stored memory

### Completion expectations
Autocomplete should surface:
- `history`
- `memory`
- `all`

No special completer changes should be necessary unless child suggestions render awkwardly.

## Handler design

## `opencouch_cli/app.py`
Add one focused helper:

- `_handle_search_command(session, runtime, args) -> bool`

### Dispatch hook
In `handle_command(...)`, add:

- `/search` → `_handle_search_command(...)`

### Handler parsing
Behavior:
- `args[0]` selects mode: `history`, `memory`, or `all`
- remaining args joined into raw query string
- empty query rejected with usage warning

### Why join remaining args
Search queries naturally contain spaces:
- `/search history panic after meeting`
- `/search memory shorter responses`

So the parser should use:
- `" ".join(args[1:]).strip()`

## Internal helper structure

Recommended helpers inside `opencouch_cli/app.py`:

- `_search_history_conversation(...)`
- `_search_memory_records(...)`
- `_search_procedural_rules(...)`
- `_render_search_results(...)`

For V1, keep these helpers in `app.py` near the existing memory collection helpers to minimize blast radius.
If the logic grows later, transcript and memory search pieces can move into a shared runtime/session module.

## History search implementation

### Source
Use the exact existing active-thread state retrieval path:
- fetch active state from runtime
- extract transcript with `get_transcript(...)`
- project to canonical visible conversation form with `session_conversation_from_transcript(...)`

### Search unit
Search at the message level:
- user message content
- assistant message content

### Result payload shape
Conceptually:

- source: `"history"`
- role: `"user"` or `"assistant"`
- snippet: matched content excerpt
- rank: integer / tuple for deterministic sorting

### Snippet behavior
For V1:
- if message is short, show full content
- if message is long, truncate around the first match

Example truncation:
- `"...sleep after travel has been rough this week..."`

### Ranking
Deterministic simple rules:
1. exact substring present
2. lower first-match index ranks higher
3. shorter snippet/message may rank slightly higher
4. stable original order as tie-breaker

This should be enough for V1.

## Memory search implementation

### Reuse target
Do not create a new memory abstraction if existing CLI helpers already gather records.

Use the exact current reuse seams:
- semantic + episodic records via `_collect_records_by_kind(...)`
- procedural rules via `aget_procedural_profile(...)`

Do not route procedural search through `_collect_records_by_kind(...)`, because procedural rules are stored as one profile document per owner rather than one record per rule.

### Search unit per memory type
#### Semantic facts
Search likely text fields such as:
- predicate/object summary where available
- evidence quote
- normalized textual rendering if already used elsewhere

#### Episodic session arcs
Search likely:
- summary
- themes / main concern text
- other human-readable summary fields already displayed

#### Procedural rules
Search:
- rule text
- evidence text

### Result payload shape
Conceptually:

- source: `"memory/fact"` | `"memory/session"` | `"memory/rule"`
- index: display index if available
- snippet: human-readable matched text
- rank: deterministic sortable key

### Important UX rule
Results must look like the memory UI, not like raw database rows.

## Result rendering

### Shared renderer
Add a small search-specific renderer:
- `_render_search_results(query, results, mode, truncated_count=0)`

Do **not** reuse `_render_semantic_records_table`, `_render_episodic_records_table`, or `_render_procedural_rules_table` as the primary search UI.
Those table helpers are optimized for browsing full lists, while `/search` should render a compact mixed-source result set.

### Empty state
Examples:
- `No history matches for "sleep after travel".`
- `No memory matches for "shorter responses".`
- `No matches for "sleep after travel".`

### Non-empty state
Render a simple panel or table with:
- source
- snippet

Example rows:
- `history | user: I need help with sleep after travel`
- `memory/fact | #3: user prefers shorter responses`

### Footer hints
Helpful but optional:
- after history search: `Want the full conversation? Use /export md`
- after broad search: `Use /memory list to inspect stored records directly`

Keep hints minimal.

## Prompt-awareness behavior
No special prompt-builder change should be required beyond the existing registry-driven command list.

Once `/search` metadata is added to `SLASH_COMMANDS`, the assistant should automatically learn about:
- `/search history <query>`
- `/search memory <query>`
- `/search all <query>`

## Testing plan

## Integration / CLI tests
Add tests in:
- `tests/integration/cli/test_opencouch_cli.py`

### Registry / help / completion
- `/search <history|memory|all> <query>` appears in help
- completion for `/search ` includes `history`, `memory`, `all`

### Dispatch / validation
- `/search` warns with usage
- `/search history` warns with usage
- `/search memory` warns with usage
- `/search foo hello` warns with unknown subcommand message

### History search behavior
- finds a matching user message
- finds a matching assistant message
- returns empty-state message when no matches exist

### Memory search behavior
- finds a matching procedural rule
- finds a matching semantic fact if easy to set up with existing helpers
- returns empty-state message when no matches exist

### All-search behavior
- returns both history and memory matches
- preserves source labels

## Broader regression checks
Re-run at minimum:
- CLI integration suite
- therapeutic prompt tests
- session history boundary tests
- any memory-list / procedural CLI tests touched by reused helpers

## Risks and mitigations

### Risk 1: memory search implementation balloons
Because semantic, episodic, and procedural memory use different storage shapes.

#### Mitigation
Ship V1 with:
- semantic + episodic search through `_collect_records_by_kind(...)`
- procedural search through `aget_procedural_profile(...)`
- human-readable snippets only
- no attempt to unify all memory kinds behind a new generic abstraction

### Risk 2: too much result noise
Especially for long transcripts or very short queries.

#### Mitigation
- cap results at 8 by default
- require non-empty query
- prefer snippet rendering over full dumps

### Risk 3: semantic-search expectations
Users may assume “search” means smart retrieval.

#### Mitigation
Keep V1 deterministic and document it implicitly via behavior.
Do not promise semantic relevance yet.

## Explicit non-goals for V1
- embedding-backed search
- cross-thread global transcript search
- LLM ranking or summarization of search results
- natural-language command execution
- advanced query syntax / filters / flags

## Acceptance criteria
V1 is complete when:

1. `/search history <query>` searches the active transcript
2. `/search memory <query>` searches stored memory content
3. `/search all <query>` combines both
4. autocomplete surfaces `history`, `memory`, and `all`
5. `/help` shows the exact public syntax
6. model awareness includes the new commands automatically through the registry
7. tests cover command validation, completions, history search, and at least one memory search path
8. no part of the search command relies on LLM parsing or execution

## Recommended implementation order
1. add command metadata
2. add CLI dispatch + validation
3. implement history search
4. implement memory search
5. implement combined search
6. add tests
7. run focused tests
8. run broader regression sweep
