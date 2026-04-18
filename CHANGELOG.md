# Changelog

## 2026-04-18 — Voice Session Tab Persistence

### Web UI — voice session
- Fixed voice chat sessions being lost on in-app tab switches (Voice → Chat → Memory → back)
- Lifted WebSocket, AudioContext, MediaStream, and ScriptProcessorNode refs into module-level singletons so they survive component unmounts
- Added reactive voice state (connected, speaking, transcripts, error) to Zustand store for cross-page UI updates
- Added generation counter to scope `onclose`/`onended` callbacks — prevents stale sessions from tearing down newer ones
- Fixed AudioContext leak when microphone permission is denied
- Moved `visibilitychange` listener to AppShell so AudioContext resume works even when the voice page is not mounted
- Added pulsing "live" indicator in sidebar when a voice session is active on another page

### Housekeeping
- Scoped root `.gitignore` `lib/` rule to `/lib/` — was unintentionally shadowing `apps/web/src/lib/`
- Started tracking `CHANGELOG.md` in version control

## 2026-04-18 — Session Trajectory Eval + Safety Hardening

### Eval infrastructure
- Added unified session trajectory eval runner (`eval/runners/session_trajectory_eval.py`) supporting both inline-expect and checkpoint dataset schemas
- Added 21 long-trajectory cases covering modality selection, boundary enforcement, closing, venting, safety pivots, and mode transitions
- Added `--concurrency`, `--case`, `--verbose` flags for faster hybrid runs and single-case debugging
- Added 4 boundary enforcement cases: medication advice, repeated diagnosis pushing, legal advice, dependency framing
- Updated eval README to document unified runner and all flags

### Crisis gate
- Refactored to LLM-primary architecture — LLM classifies all non-override messages, regex is fallback only
- Fixed override precedence: imminent risk now checked before idiomatic-safe
- Scoped idiomatic-safe matching to current message only (prior idiom cannot suppress current threat)
- Added guard: idiomatic-safe skipped when clear self-harm or ambiguous patterns also match
- Added shadow monitoring: deterministic result logged alongside LLM for drift detection
- Hardened classifier prompt: conversation fencing, anti-injection instructions, adversarial examples (negation, quoted speech, sarcasm), confidence constraints
- Sanitized classifier reason output (length cap, control char removal, injection fence)
- Enforced strict level-to-flag truth table in normalize_crisis_assessment
- Tightened denial regex: requires explicit self-harm object after "not thinking about"
- Fixed denial false-positive: "no...ending it" now requires negation-of-intent phrasing
- Fixed post-safety-check gap: DISTRESS_PATTERNS now checked alongside AMBIGUOUS_PATTERNS
- Narrowed safety-check detection phrases to reduce false-context risk

### Therapeutic dispatcher
- Refactored to LLM-primary architecture — LLM handles all mode + modality classification
- Removed context-blind regex fast paths (reflective, confusion, closing) from primary path
- Demoted broad patterns to regex fallback only (kept narrowest forms: "why do I keep", "huh?", "what?")
- Added LLM-based mid-exercise exit detection — natural exit without regex traps
- Mid-exercise clarification preserves exercise state (repair turns don't exit)
- Tightened exercise exit patterns to unambiguous opt-out only
- Sharpened ACT vs CBT distinction in classifier prompt with concrete user-language examples
- Added exercise context injection into LLM prompt for continuation/exit decisions

### Agent changes
- Added OpenAI embedding provider (`text-embedding-3-large`) as default, Gemini as fallback
- Added extraction precision guard for early-session emerging patterns and negative self-beliefs
- Added context-aware exercise selection using recent conversation history
- Fixed modality persistence across exercise turns via routing merge reducer

### Eval coverage expansion
- Added 4 multi-turn crisis trajectory cases: gradual escalation, de-escalation after safety check, oscillating signals, post-crisis recovery (21 → 25 long trajectory cases)
- Added 7 small-talk gate boundary cases to extraction dataset: name-in-thanks, therapist mention, bare acknowledgments, 40-char boundary (26 → 33 extraction cases)
- Added deterministic fallback quality eval case: checks banned soul.md phrases in hardcoded responses (8 → 9 short trajectory cases)
- Fixed dotenv loading in retrieval eval runner so API keys from .env files are picked up
- Verified OpenAI embedding provider end-to-end: hybrid RRF achieves 14/17 recall@5 vs 6/17 token-only

### Deterministic fallback quality
- Fixed supportive fallback: removed "Thank you for sharing" and "I want you to know" (banned by soul.md)
- Fixed closing fallback: replaced "Thanks for sharing" with voice-compliant alternative

### Knowledge
- `soul.md`: added therapeutic grounding, cultural sensitivity, repair after missteps, boundary-setting voice, expanded antipatterns
- `identity.md`: full rewrite with product philosophy, capabilities, limits
- `boundaries.md`: expanded with redirection patterns, dependency framing, escalation guidance
