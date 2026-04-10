"""Shared text tokenization helpers for the memory layer.

This module exists as a tiny, dependency-free seam so that both the hot-
path dedup helper (:mod:`agent.memory.dedup`) and the store's search
layer (:mod:`agent.memory.store`) can tokenize text the same way without
either importing the other. Having a single canonical tokenizer matters
because retrieval and dedup have to agree on what a "token" is — if they
drift, dedup will merge two facts that the search layer can't actually
distinguish, or vice versa.

Two functions live here:

- :func:`tokenize` — returns a frozenset of lowercased word tokens. Used
  by both dedup (to compute Jaccard similarity between two evidence
  quotes) and the store's search path (to compute query-recall against
  a haystack of serialized record fields).

- :func:`tokenize_meaningful` — the same tokenizer with a small stopword
  and length filter applied. Used only by the search path. Dedup does
  NOT use the stopword filter because for dedup-quality comparisons,
  even pronouns and articles carry signal ("I feel sad" vs "she feels
  sad" should be different facts, so dropping "I"/"she" would collapse
  them incorrectly).

Stopword philosophy: keep the list tiny and obviously-safe. The risk of
an over-eager stopword list is weird retrieval behavior that's hard to
debug later. The current list covers pronouns, articles, the two most
common copulas, and a handful of high-frequency connectives — all
things that don't meaningfully distinguish one therapy utterance from
another.
"""

from __future__ import annotations

import re

# Token extraction regex: lowercase word characters, ignoring punctuation.
# "I'm anxious!" and "im anxious" produce the same token set. Apostrophes
# split contractions ("I'm" → ["i", "m"]) which is imperfect but stable.
# NOTE: this is intentionally the same regex the dedup helper used
# internally — moved here so both paths share a canonical definition.
_WORD_RE = re.compile(r"\b[a-z0-9]+\b")

# Tiny stopword set used ONLY by the search path (not by dedup). These
# are high-frequency function words that add no retrieval signal: keeping
# them in the token set inflates recall scores and causes queries like
# "I worry about Sarah" to spuriously match "I worry about work" via the
# overlap on connective words.
#
# Keep this list small and boring. A 200-word stopword list is a recipe
# for surprising retrieval failures ("I can't remember" queries where
# "can" is stripped, etc.). The current set is pronouns, articles, the
# copula, and a few prepositions — all of which are safe to drop without
# changing the semantic thrust of a query.
_STOPWORDS: frozenset[str] = frozenset(
    {
        # pronouns
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "yours",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "we",
        "us",
        "our",
        "they",
        "them",
        "their",
        "it",
        "its",
        # articles
        "a",
        "an",
        "the",
        # copulas
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        # auxiliary verbs — content-free helpers that fill out tense
        # and aspect without carrying topical signal. Adding these keeps
        # queries like "Things have been tense with Sarah lately" from
        # spuriously matching on connectives instead of content words.
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        # common prepositions / connectives
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "about",
        "and",
        "or",
        "but",
        "so",
        "that",
        "this",
        "these",
        "those",
    }
)


def tokenize(text: str) -> frozenset[str]:
    """Return a case-insensitive word-token set for the given text.

    The returned set is deduplicated by construction (so "work work work"
    becomes ``{"work"}``) and lowercased. Punctuation, whitespace, and
    contraction apostrophes are all treated as token boundaries.

    This is the full token set — no stopword filtering, no length filter.
    Use :func:`tokenize_meaningful` if you want those applied.

    Returns an empty frozenset for empty or punctuation-only input so
    callers can safely compute set operations without a ``None`` guard.
    """

    return frozenset(_WORD_RE.findall(text.lower()))


def tokenize_meaningful(text: str) -> frozenset[str]:
    """Return a stopword- and length-filtered token set for retrieval scoring.

    Same as :func:`tokenize`, but additionally drops:

    - Any token in :data:`_STOPWORDS` (the tiny high-frequency function
      word list).
    - Single-character tokens (``"i"``, ``"a"``, ``"m"`` from contracted
      forms, etc.). Anything shorter than 2 characters is almost always
      noise for retrieval scoring.

    This is the right tokenizer for the **query side** of a search: we
    want retrieval scores to reflect topical overlap, not connective-
    word coincidence. A query like "I worry about Sarah" should have
    meaningful tokens ``{"worry", "sarah"}`` and score against only
    those — otherwise overlap on "i", "about" would make it match any
    record with an "I worry about <anything>" quote.

    For the **document side** (the haystack), you typically want the
    full :func:`tokenize` output: strip nothing, so any query token has
    a chance to land. The query filter is what matters; the document
    filter is a footgun.

    If ALL of the text's tokens get filtered out (e.g. the user typed
    "I am so"), returns an empty frozenset. Callers should treat an
    empty query-token set as "no meaningful query" and decide how to
    degrade — typically by falling back to "return everything" or
    "return nothing."
    """

    return frozenset(
        token
        for token in _WORD_RE.findall(text.lower())
        if len(token) >= 2 and token not in _STOPWORDS
    )
