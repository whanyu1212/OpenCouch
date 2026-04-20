"""Procedural rule writer node with hot-path policy + session buffering.

Runs after the response generation node on every turn and writes zero
or more procedural rules to the user's ``ProceduralProfile`` in the
procedural namespace of the memory store. The node splits extraction from
immediate persistence: the LLM output first becomes a procedural
candidate, the deterministic policy decides whether it is safe and
durable enough to ``commit_now``, and only those candidates are written
to the profile. Held procedural candidates are buffered for later phases
rather than written immediately. Parallels the design of
:func:`agent.nodes.extract_facts.run_extract_semantic_facts_node`
structurally — per-turn side effect, runs on both the crisis and
therapeutic branches, silently skips when no LLM client is available
or memory mode is INCOGNITO, always returns an empty state delta.

Phase-1 design rules:

1. **Conservative writing.** Most turns produce zero rules. Rule writes
   are reserved for moments when the user either explicitly asks the
   agent to change how it responds or makes a clear durable agent-facing
   preference statement. The system prompt enforces this and uses
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

5. **Policy before persistence.** Extracted rule drafts become
   :class:`ProceduralCandidate` instances first. The deterministic
   policy can downgrade a candidate to ``commit_at_session_end`` or
   ``drop``. In phase 1 those non-``commit_now`` outcomes are
   diagnostics-only and do not write.

6. **Conservative reconciliation.** Phase D still keeps the node
   simple, but new writes now reconcile against the existing profile:
   exact duplicates are skipped and clearly stronger/conflicting rules
   can replace older wording. This avoids stale procedural guidance
   accumulating indefinitely without adding a full background job.

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
- **Graph placement.** Both are on the post-response spine, fanned out
  in parallel from ``finalize_turn_node``. Neither node affects user
  latency because both run after the response is already composed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langgraph.runtime import Runtime

from agent.memory.candidates import build_procedural_candidate
from agent.memory.models import ProceduralExtractionResult
from agent.memory.modes import MemoryMode
from agent.memory.procedural import aupsert_procedural_rule, build_procedural_rule
from agent.memory.procedural_prompts import (
    build_procedural_writer_system_prompt,
    build_procedural_writer_user_prompt,
)
from agent.memory.write_policy import decide_procedural_candidate
from agent.runtime_context import WorkflowContext
from agent.state import AgentState, resolve_owner_id

logger = logging.getLogger(__name__)


async def run_extract_procedural_rules_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Write zero or more procedural rules from the current user turn.

    Runs on the post-response spine after the reply has already been
    generated. Returns a state delta containing the per-turn diagnostics
    entry (timing + write count). The actual rule writes are side
    effects on the procedural namespace of the memory store.

    Silently skips when the runtime lacks an LLM client or is in
    incognito mode. All other failure modes (LLM errors, schema
    validation errors, store write errors) are logged at WARNING level
    with ``exc_info=True`` but never propagate.
    """

    # v0.8 observability: time the full writer call and report
    # write count via the diagnostics dict. Same shape as the
    # semantic extractor's _diagnostics_delta helper.
    start = time.monotonic()

    def _diagnostics_delta(
        *,
        procedural_writes: int = 0,
        procedural_candidates: int = 0,
        procedural_commit_now_candidates: int = 0,
        procedural_session_end_holds: int = 0,
        procedural_policy_drops: int = 0,
        reason: str = "",
    ) -> dict[str, Any]:
        """Return a state delta carrying just the writer's diagnostics."""

        return {
            "diagnostics": {
                "extract_procedural_ms": round((time.monotonic() - start) * 1000, 2),
                "procedural_writes": procedural_writes,
                "procedural_candidates": procedural_candidates,
                "procedural_commit_now_candidates": procedural_commit_now_candidates,
                "procedural_session_end_holds": procedural_session_end_holds,
                "procedural_policy_drops": procedural_policy_drops,
                "extract_procedural_reason": reason,
            }
        }

    # ── Early exits ─────────────────────────────────────────────────────

    # v0.9: crisis gate first — same rationale as extract_facts.
    route = state.get("routing", {}).get("route")
    if route == "crisis":
        logger.debug(
            "extract_procedural_rules_node: crisis path; skipping to avoid "
            "delaying crisis response delivery"
        )
        return _diagnostics_delta(reason="skipped: crisis_path")

    llm_client = runtime.context.llm_client
    memory_mode = runtime.context.memory_mode

    if llm_client is None:
        logger.debug("extract_procedural_rules_node: no llm_client; skipping")
        return _diagnostics_delta(reason="skipped: no llm_client")
    if memory_mode == MemoryMode.INCOGNITO:
        logger.debug(
            "extract_procedural_rules_node: incognito mode; skipping (no "
            "writes to persistent memory in incognito)"
        )
        return _diagnostics_delta(reason="skipped: incognito")

    store = runtime.context.memory_store
    session_buffer = runtime.context.session_memory_buffer
    owner_id = resolve_owner_id(state)

    from agent.memory.small_talk_gate import is_small_talk

    if is_small_talk(state["message"]):
        logger.debug(
            "extract_procedural_rules_node: small-talk gate triggered; "
            "skipping LLM call for message %r",
            state["message"][:40],
        )
        return _diagnostics_delta(reason="skipped: small_talk_gate")

    progress = state.get("progress", {})
    turn_count = int(progress.get("turn_count", 1))
    turn_index = max(0, turn_count - 1)

    # ── LLM structured-output writing ───────────────────────────────────
    try:
        result: ProceduralExtractionResult = await llm_client.generate_structured(
            prompt=build_procedural_writer_user_prompt(state),
            response_schema=ProceduralExtractionResult,
            system_instruction=build_procedural_writer_system_prompt(),
        )
    except Exception:
        logger.warning(
            "extract_procedural_rules_node: LLM structured-output call "
            "failed; skipping rule writes for this turn.",
            exc_info=True,
        )
        return _diagnostics_delta(reason="skipped: llm error")

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
        return _diagnostics_delta(reason=result.reason)

    # ── Build candidates and apply deterministic write policy ─────────
    immediate_candidates: list[tuple[Any, Any]] = []
    session_end_holds = 0
    policy_drops = 0

    for draft in result.rules:
        candidate = build_procedural_candidate(
            draft,
            message=state["message"],
            session_id=state.get("session_id") or owner_id,
            turn_index=turn_index,
        )
        decision = decide_procedural_candidate(candidate)

        if decision.action == "commit_now":
            immediate_candidates.append((candidate, decision))
        elif decision.action == "commit_at_session_end":
            session_end_holds += 1
            if session_buffer is not None:
                session_buffer.procedural_candidates.append(candidate)
        else:
            policy_drops += 1

    if not immediate_candidates:
        logger.info(
            "extract_procedural_rules_node: policy held all %d rules "
            "(session_end=%d, dropped=%d)",
            len(result.rules),
            session_end_holds,
            policy_drops,
        )
        return _diagnostics_delta(
            procedural_candidates=len(result.rules),
            procedural_session_end_holds=session_end_holds,
            procedural_policy_drops=policy_drops,
            reason=result.reason,
        )

    # ── Per-draft write ────────────────────────────────────────────────
    # Each surviving candidate gets promoted to a ProceduralRule via
    # build_procedural_rule (which adds the added_at timestamp and
    # source="explicit_user"), then upserted into the user's profile.
    # The helper handles the load → reconcile → put idiom internally;
    # we don't touch the store directly.
    #
    # Error handling is per-draft: a failure to write one rule does NOT
    # abandon the remaining drafts. This matches the semantic extractor's
    # per-candidate error isolation.
    written = 0
    for candidate, decision in immediate_candidates:
        draft = candidate.payload
        try:
            rule = build_procedural_rule(
                rule_text=draft.rule,
                evidence=draft.evidence,
                confidence=draft.confidence,
                source="explicit_user",
                write_timing="immediate",
                write_reason=decision.reason,
                policy_version=decision.policy_version,
            )
            upsert = await aupsert_procedural_rule(store, user_id=owner_id, rule=rule)
            if upsert.action != "skipped":
                written += 1
        except Exception:
            logger.warning(
                "extract_procedural_rules_node: failed to write draft %r; "
                "continuing with other drafts.",
                draft.rule[:60],
                exc_info=True,
            )

    logger.info(
        "extract_procedural_rules_node: turn complete — %d written, %d immediate, "
        "%d held_for_session, %d dropped",
        written,
        len(immediate_candidates),
        session_end_holds,
        policy_drops,
    )
    return _diagnostics_delta(
        procedural_writes=written,
        procedural_candidates=len(result.rules),
        procedural_commit_now_candidates=len(immediate_candidates),
        procedural_session_end_holds=session_end_holds,
        procedural_policy_drops=policy_drops,
        reason=result.reason,
    )
