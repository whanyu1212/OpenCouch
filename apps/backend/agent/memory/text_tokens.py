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

# Token extraction regex: Unicode word characters, ignoring punctuation.
# "I'm anxious!" and "im anxious" produce the same token set. Apostrophes
# split contractions ("I'm" → ["i", "m"]) which is imperfect but stable.
# NOTE: this is intentionally the same regex the dedup helper used
# internally — moved here so both paths share a canonical definition.
#
# Uses \w+ (Unicode-aware by default in Python 3) so accented Latin,
# Cyrillic, and CJK characters produce tokens instead of empty sets.
# CJK characters are post-processed into per-character tokens by
# _split_cjk because \b doesn't insert boundaries between consecutive
# CJK codepoints.
_WORD_RE = re.compile(r"\b\w+\b")

# CJK Unicode ranges: CJK Unified Ideographs (BMP + astral), CJK
# Compatibility Ideographs, Hangul Syllables, Katakana, Hiragana,
# and Bopomofo. Astral-plane extensions (B through H) are included
# so rare characters in names and classical texts tokenize correctly.
_CJK_RE = re.compile(
    r"["
    r"\u2E80-\u9FFF"  # CJK Radicals Supplement through CJK Unified Ideographs
    r"\uF900-\uFAFF"  # CJK Compatibility Ideographs
    r"\uAC00-\uD7AF"  # Hangul Syllables
    r"\u3040-\u309F"  # Hiragana
    r"\u30A0-\u30FF"  # Katakana
    r"\u3100-\u312F"  # Bopomofo
    r"\U00020000-\U0002A6DF"  # CJK Unified Ideographs Extension B
    r"\U0002A700-\U0002B73F"  # Extension C
    r"\U0002B740-\U0002B81F"  # Extension D
    r"\U0002B820-\U0002CEAF"  # Extension E
    r"\U0002CEB0-\U0002EBEF"  # Extension F
    r"\U00030000-\U0003134F"  # Extension G
    r"\U00031350-\U000323AF"  # Extension H
    r"]"
)


def _split_cjk(token: str) -> list[str]:
    """Split a token containing CJK characters into indexable pieces.

    Args:
        token (str): Raw token to split.

    Returns:
        list[str]: Per-character CJK tokens plus any preserved non-CJK runs.
    """

    if not _CJK_RE.search(token):
        return [token]

    parts: list[str] = []
    buf: list[str] = []
    for ch in token:
        if _CJK_RE.match(ch):
            if buf:
                parts.append("".join(buf))
                buf.clear()
            parts.append(ch)
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


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
    """Return the full case-insensitive token set for text.

    Args:
        text (str): Input text to tokenize.

    Returns:
        frozenset[str]: Deduplicated lowercase tokens with no stopword filtering.
    """

    tokens: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        tokens.extend(_split_cjk(raw))
    return frozenset(tokens)


def tokenize_meaningful(text: str) -> frozenset[str]:
    """Return the retrieval-oriented token set for text.

    Args:
        text (str): Input text to tokenize.

    Returns:
        frozenset[str]: Lowercase tokens after stopword and short-token filtering.
    """

    tokens: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        tokens.extend(_split_cjk(raw))
    return frozenset(
        token
        for token in tokens
        if (len(token) >= 2 or _CJK_RE.match(token)) and token not in _STOPWORDS
    )
