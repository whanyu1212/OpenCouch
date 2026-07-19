"""Write policy for memory persistence.

Callers decide whether something is *candidate* memory. This module asks an
LLM-primary policy classifier whether that candidate should commit immediately,
wait for session-end review, require repeated evidence, or drop. Local code
only enforces hard safety/storage invariants; it does not provide
product-judgment fallback writes when the policy LLM is unavailable.
"""

from __future__ import annotations

import logging

from agent.memory.policy.clamps import (
    semantic_hard_policy_guard,
)
from agent.memory.policy.markers import (
    semantic_candidate_is_memory_control_request,
    semantic_candidate_is_turn_scoped,
    semantic_candidate_needs_repetition_guard,
    text_contains_memory_control_request,
)
from agent.memory.policy.thresholds import (
    should_commit_implicit_procedural_preference,
    should_commit_pattern,
)

logger = logging.getLogger(__name__)


# The per-candidate LLM write-policy classifiers
# (decide_semantic_candidate_llm_primary / decide_procedural_candidate_llm_primary)
# were removed when session-end consolidation moved to a whole-transcript
# extractor: the extractor holds candidates as commit_at_session_end directly and
# commit-time corroboration scoring re-gates durability, so the per-candidate
# classifier is no longer used. This module retains the hard-guard / marker /
# threshold helpers that ARE still consumed (e.g. by commit selection).


__all__ = [
    "semantic_candidate_is_memory_control_request",
    "semantic_candidate_is_turn_scoped",
    "semantic_candidate_needs_repetition_guard",
    "semantic_hard_policy_guard",
    "should_commit_implicit_procedural_preference",
    "should_commit_pattern",
    "text_contains_memory_control_request",
]
