"""Pre-extractor small-talk gate.

Both ``extract_facts_node`` and ``extract_procedural_rules_node`` make
full LLM structured-output calls on every turn. For small-talk turns
("hi", "thanks", "ok got it") these calls burn ~3-5s of LLM round-trip
time and reliably return zero writes.

This module provides a single function :func:`is_small_talk` that the
two extractor nodes call **before** their LLM structured-output call.
When the function returns True, the node skips the LLM call entirely
and returns a diagnostics delta with
``reason="skipped: small_talk_gate"``.

Heuristic design — conservative by contract:

The gate uses two signals, both of which must be true:

1. **Short message.** Under :data:`MAX_SMALL_TALK_LENGTH` characters
   (default 40) after stripping whitespace. Most memory-worthy
   statements are longer because they contain names, context, or
   descriptions. "My sister Sarah visited this weekend" is 42 chars
   and would pass through.

2. **No content words.** After stopword filtering via
   :func:`agent.memory.text_tokens.tokenize_meaningful`, every
   remaining token must be in the :data:`SMALL_TALK_VOCABULARY` set.
   This catches "hello there", "thanks so much", "ok cool" while
   letting "hello Sarah" or "thanks for the meditation tip" through
   (those have a named entity or topical keyword that the extractor
   might care about).

**False-negative bias is intentional.** A false negative (letting a
small-talk turn through to the LLM) wastes ~3-5s of LLM cost but is
otherwise harmless — the extractor will correctly return zero facts.
A false positive (skipping a turn that had extractable content)
silently loses memory, which is bad and unrecoverable. The heuristic
is tuned to err heavily toward "let it through" — it only triggers
on messages that are unambiguously small talk by both length AND
vocabulary.

Why not use the LLM itself to decide? Because the whole point is to
**avoid** the LLM call. A classifier-style gate that requires a model
inference to decide whether to run the main inference defeats the
purpose. The heuristic is sub-millisecond and deterministic.

When to extend this gate:

- **Add a word to SMALL_TALK_VOCABULARY** when product usage surfaces
  a common small-talk token that's currently passing through. Verify
  the word is not also a topical signal for stored facts.
- **Raise MAX_SMALL_TALK_LENGTH** if common small-talk messages are
  consistently longer than 40 chars. Raising modestly is safe; large
  increases risk false positives on short declarative statements.
- **Do NOT add topic detection** (e.g., "if the message mentions
  a known entity, let it through"). That couples the gate to the
  memory store's content, which makes it non-deterministic and
  hard to test. The vocabulary check is enough.
"""

from __future__ import annotations

from agent.memory.text_tokens import tokenize_meaningful

# Maximum character length (after strip) for a message to be
# considered a small-talk candidate. Messages longer than this
# always pass through to the LLM extractor regardless of vocabulary.
#
# 40 chars is calibrated against common short acknowledgments:
# - "hi" (2), "hello" (5), "thanks" (6), "ok got it" (9),
#   "yeah that makes sense" (22), "appreciate it" (13) → all under 40
# - "my sister Sarah visited this weekend" (37) → under 40 but would
#   fail the vocabulary check (has "sister", "sarah", "visited",
#   "weekend" which are all outside the small-talk vocab)
# - "I have been feeling more anxious lately" (41) → over 40, passes
#   through on length alone
#
# The 40-char threshold is deliberately tight. It's better to let
# a 35-char small-talk message through (wasting one LLM call) than
# to skip a 35-char memory-worthy statement.
MAX_SMALL_TALK_LENGTH = 40

# Small-talk vocabulary — the set of meaningful tokens (after stopword
# filtering) that are unambiguously small talk. A message whose
# meaningful tokens are ALL in this set is classified as small talk.
# A message with even ONE token outside this set passes through.
#
# The set is intentionally small and boring. Every word here must be:
# - Clearly non-topical in a therapy context (no emotion words, no
#   coping words, no relationship words, no clinical terms)
# - Not a proper noun or identifier that a user might mention
# - Not a word that could be part of a memory-worthy statement
#
# Words like "good", "bad", "fine" are deliberately excluded because
# they carry emotional valence that the extractor might care about
# ("I'm fine" vs "I feel fine about it" are different signals).
# "yeah" and "ok" are included because they're pure acknowledgments
# with no extractable content.
SMALL_TALK_VOCABULARY: frozenset[str] = frozenset(
    {
        # greetings
        "hi",
        "hey",
        "hello",
        "howdy",
        # farewells (closing mode handles these, but the extractor
        # doesn't need to run on them)
        "bye",
        "goodbye",
        "later",
        # acknowledgments
        "ok",
        "okay",
        "yes",
        "no",
        "yeah",
        "yep",
        "nope",
        "sure",
        "right",
        "alright",
        "cool",
        "nice",
        "wow",
        "hm",
        "hmm",
        "huh",
        "ah",
        "oh",
        "uh",
        # gratitude (the extractor never extracts "user said thanks"
        # as a persistent fact)
        "thanks",
        "thank",
        "thx",
        "appreciate",
        # continuation signals
        "go",
        "ahead",
        "continue",
        "next",
        "ready",
        # affirmations
        "got",
        "makes",
        "sense",
        "sounds",
        "good",
        "great",
        "understood",
        "see",
        # fillers and intensifiers — carry no topical signal,
        # commonly appear in short acknowledgment phrases like
        # "thanks so much", "hello there", "really cool"
        "much",
        "really",
        "very",
        "so",
        "too",
        "there",
        "well",
        "just",
        "pretty",
    }
)


def is_small_talk(message: str) -> bool:
    """Return whether the message is unambiguously small talk.

    Args:
        message (str): User message to evaluate.

    Returns:
        bool: ``True`` when the message is short and all meaningful tokens are small-talk vocabulary.
    """

    stripped = message.strip()

    # Empty messages pass through — the node handles them upstream.
    if not stripped:
        return False

    # Length gate: messages over the threshold always pass through.
    if len(stripped) > MAX_SMALL_TALK_LENGTH:
        return False

    # Vocabulary gate: every meaningful token must be in the small-talk set.
    meaningful = tokenize_meaningful(stripped)

    # If there are no meaningful tokens at all (pure stopwords like
    # "I am so"), that's degenerate input the extractor would also
    # reject — but we let it through rather than making assumptions.
    # The cost is one wasted LLM call; the benefit is not having to
    # reason about edge cases here.
    if not meaningful:
        return False

    return meaningful.issubset(SMALL_TALK_VOCABULARY)
