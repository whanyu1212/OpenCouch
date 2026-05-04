"""Regex patterns for memory-control routing policy."""

from __future__ import annotations

import re

YES_RE = re.compile(
    r"^\s*(yes|yep|yeah|please do|do it|confirm|delete it)(?:,\s*delete it)?\s*[.!]?\s*$",
    re.I,
)
NO_RE = re.compile(
    r"^\s*(no|nope|cancel|never mind|don't|do not|stop)\s*[.!]?\s*$", re.I
)
INDEX_RE = re.compile(r"(?:#|number\s+)?(?P<index>\d+)")
AMBIGUOUS_MEMORY_CONTROL_SIGNAL_RE = re.compile(
    r"("
    r"\bcan you (?:remember|keep in mind|save|stop remembering|stop using|"
    r"stop bringing (?:this|that|it) up|stop bringing up)\b|"
    r"\bcould you (?:remember|keep in mind|save|stop remembering|stop using|"
    r"stop bringing (?:this|that|it) up|stop bringing up)\b|"
    r"\bplease (?:remember|keep in mind|save|stop remembering|stop using|"
    r"stop bringing (?:this|that|it) up|stop bringing up)\b|"
    r"^\s*keep in mind that\b|^\s*remember (?:this|that)\b|"
    r"^\s*save (?:this|that)\b|"
    r"\bdon't (?:remember|save|store|bring up|bring (?:this|that|it) up|"
    r"mention|use)\b|"
    r"\bdo not (?:remember|save|store|bring up|bring (?:this|that|it) up|"
    r"mention|use)\b|"
    r"\bstop (?:remembering|saving|storing|bringing up|"
    r"bringing (?:this|that|it) up|mentioning|using)\b|"
    r"\bforget (?:this|that|what i said|the thing|the memory|my|about)\b|"
    r"\bdelete (?:this|that|what i said|the thing|the memory|my|about)\b|"
    r"\bremove (?:this|that|what i said|the thing|the memory|my|about)\b|"
    r"\bwhat (?:do|have) you (?:remember|know|saved)\b"
    r")",
    re.I,
)
PREFERENCE_RULE_RE = re.compile(
    r"\b(prefer|preference|respond|repl(?:y|ies)|answer|ask|remind|bring up|"
    r"mention|use|avoid|tone|style|brief|short(?:er)?|concise|gentle|direct|"
    r"format|language|call me|address me)\b",
    re.I,
)
