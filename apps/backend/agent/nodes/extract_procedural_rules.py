"""Procedural rule writer node (v0.7).

Runs after the response generation node on every turn and writes zero
or more procedural rules to the user's ``ProceduralProfile`` in the
procedural namespace of the memory store. Parallels the design of
:func:`agent.nodes.extract_facts.run_extract_semantic_facts_node`
structurally — per-turn side effect, runs on both the crisis and
therapeutic branches, silently skips when no LLM client is available
or memory mode is INCOGNITO, always returns an empty state delta.

v0.7 design rules (locked via v0.7 scoping):

1. **Conservative writing.** Most turns produce zero rules. Rule writes
   are reserved for moments when the user explicitly asks the agent to
   change how it responds. The system prompt enforces this and uses
   second-person evidence-grounded phrasing; see
   ``agent/memory/procedural_prompts.py`` for the prompt design notes.

2. **Silent skip on incognito or no LLM.** Incognito mode is a privacy
   promise — no long-term writes to the memory store. The writer node
   no-ops without logging an error. Same contract as the semantic
   extractor.

3. **Failures degrade silently.** LLM errors, schema validation errors,
   and store write errors are all logged at WARNING level but never
   propagate. The writer is a side-effect node — a failure here must
   not fail the parent turn.

4. **Always return an empty state delta.** The write is a side effect;
   state isn't modified. Returning ``{}`` is the canonical "I touched
   nothing" signal for LangGraph delta-return nodes.

5. **No dedup in v0.7.** Duplicate rules (two store entries with
   identical rule text) are possible but rare because the node is
   conservative-by-default. A future v0.7.1 could add similarity-based
   dedup; for now the conservative prompt + the low expected rule
   rate make this acceptable. Users can manually remove duplicates
   via ``/memory forget rule <n>`` when it lands in Stage E.

Relationship with the semantic extractor:

Both nodes run after response generation, both are side-effect only,
both are conservative-by-default. They differ in:

- **Trigger philosophy.** Semantic writes on SHARED FACTS (user-
  volunteered persistent information). Procedural writes on DIRECT
  REQUESTS (user asking the agent to behave differently). The two
  extractors have different failure modes and different prompts,
  and they run as separate LLM calls rather than being combined.
- **Storage shape.** Semantic uses one-record-per-fact; procedural
  uses a single profile document per user at a fixed key. See
  :mod:`agent.memory.procedural` for the store helpers.
- **Graph placement.** Both are on the post-response spine. The
  procedural writer runs AFTER the semantic extractor in a serial
  chain: ``... → extract_semantic_facts_node →
  extract_procedural_rules_node → finalize_turn_node → END``. Serial
  ordering keeps log output deterministic and avoids parallel-node
  complexity that wouldn't buy anything (neither node affects user
  latency because both run after the response is already composed).
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.runtime import Runtime

from agent.memory.models import ProceduralExtractionResult
from agent.memory.modes import MemoryMode
from agent.memory.procedural import aadd_procedural_rule, build_procedural_rule
from agent.memory.procedural_prompts import (
    build_procedural_writer_system_prompt,
    build_procedural_writer_user_prompt,
)
from agent.runtime_context import WorkflowContext
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def run_extract_procedural_rules_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Write zero or more procedural rules from the current user turn.

    Runs after :func:`run_extract_semantic_facts_node` on both branches.
    Returns an empty state delta; the writes are side effects on the
    procedural namespace of the memory store.

    Silently skips when the runtime lacks an LLM client or is in
    incognito mode. All other failure modes (LLM errors, schema
    validation errors, store write errors) are logged at WARNING level
    with ``exc_info=True`` but never propagate.
    """

    # ── Early exits ─────────────────────────────────────────────────────
    llm_client = runtime.context.get("llm_client")
    memory_mode = runtime.context.get("memory_mode", MemoryMode.INCOGNITO)

    if llm_client is None:
        logger.debug("extract_procedural_rules_node: no llm_client; skipping")
        return {}
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "extract_procedural_rules_node: incognito mode; skipping (no "
            "writes to persistent memory in incognito)"
        )
        return {}

    store = runtime.context["memory_store"]
    owner_id = state.get("user_id") or state.get("session_id") or "local-default"

    # ── LLM structured-output writing ───────────────────────────────────
    try:
        result: ProceduralExtractionResult = await llm_client.generate_structured(
            prompt=build_procedural_writer_user_prompt(state),
            response_schema=ProceduralExtractionResult,
            system_instruction=build_procedural_writer_system_prompt(),
            temperature=0,
        )
    except Exception:
        logger.warning(
            "extract_procedural_rules_node: LLM structured-output call "
            "failed; skipping rule writes for this turn.",
            exc_info=True,
        )
        return {}

    # Log the writer's reason regardless of whether rules were produced —
    # it's a free observability signal for prompt tuning. INFO level (not
    # DEBUG) so dogfood sessions can see writer decisions without having
    # to rewire logging. The conservative writer rejects most turns, and
    # knowing *why* it rejected a turn is the fastest way to catch prompt
    # drift.
    logger.info(
        "extract_procedural_rules_node: %d rules, reason=%r",
        len(result.rules),
        result.reason,
    )

    if not result.rules:
        return {}

    # ── Per-draft write ────────────────────────────────────────────────
    # Each draft gets promoted to a ProceduralRule via build_procedural_rule
    # (which adds the added_at timestamp and source="explicit_user"), then
    # appended to the user's profile via aadd_procedural_rule. The helper
    # handles the load → mutate → put idiom internally; we don't touch
    # the store directly.
    #
    # Error handling is per-draft: a failure to write one rule does NOT
    # abandon the remaining drafts. This matches the semantic extractor's
    # per-candidate error isolation.
    written = 0
    for draft in result.rules:
        try:
            rule = build_procedural_rule(
                rule_text=draft.rule,
                evidence=draft.evidence,
                confidence=draft.confidence,
                source="explicit_user",
            )
            await aadd_procedural_rule(store, user_id=owner_id, rule=rule)
            written += 1
        except Exception:
            logger.warning(
                "extract_procedural_rules_node: failed to write draft %r; "
                "continuing with other drafts.",
                draft.rule[:60],
                exc_info=True,
            )

    logger.info(
        "extract_procedural_rules_node: turn complete — %d written, %d total drafts",
        written,
        len(result.rules),
    )
    return {}
