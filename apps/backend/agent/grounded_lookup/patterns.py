"""Regex patterns for grounded lookup routing policy."""

from __future__ import annotations

import re

LOOKUP_VERB_RE = re.compile(
    r"\b(look up|search(?: for| online| the web)?|web search|google|"
    r"check official|check current|"
    r"find (?:official|current|verified|local|nearby|resources|services|"
    r"clinics|directories))\b",
    re.I,
)
CHECK_IF_RE = re.compile(r"\bcan you check (?:if|whether)\b", re.I)
VERIFY_RE = re.compile(r"\bverify(?: whether| if| that)?\b", re.I)
CURRENT_INFO_RE = re.compile(
    r"\b(latest|current|up[- ]?to[- ]?date|still available|still works|"
    r"eligibility|official rules?|law|regulation|policy|price|cost|schedule)\b",
    re.I,
)
THERAPEUTIC_SUBJECTIVE_RE = re.compile(
    r"\b(being unreasonable|overreacting|bad person|wrong for feeling|"
    r"should i feel|why do i feel|what does it mean that i|"
    r"is it normal to feel|am i wrong|am i bad)\b",
    re.I,
)
AMBIGUOUS_LOOKUP_SIGNAL_RE = re.compile(
    r"\b(can you check|can you verify|verify whether|verify if|fact[- ]?check|"
    r"evidence[- ]?based|research|stud(?:y|ies)|clinical trials?|proven|"
    r"legit|reliable|source|sources|citation|citations|website|url|link|"
    r"wearables?|apps?|does .{0,40} work|is .{0,40} effective|"
    r"is .{0,40} safe)\b",
    re.I,
)
