"""Procedural rule writer node with hot-path policy and session buffering.

Runs after the response generation node on every turn and writes zero
or more procedural rules to the user's ``ProceduralProfile`` in the
procedural namespace of the memory store. The node splits extraction from
immediate persistence: the LLM output first becomes a procedural
candidate, the deterministic policy decides whether it is safe and
durable enough to ``commit_now``, and only those candidates are written
to the profile. Held procedural candidates are buffered for session-end review
rather than written immediately. Parallels the design of
:func:`agent.nodes.extract_facts.run_extract_semantic_facts_node`
structurally: per-turn side effect, runs on both the crisis and
therapeutic branches, silently skips when no LLM client is available
or memory mode is INCOGNITO, always returns an empty state delta.

Design rules:

1. **Conservative writing.** Most turns produce zero rules. Rule writes
   are reserved for moments when the user either explicitly asks the
   agent to change how it responds or makes a clear durable agent-facing
   preference statement. The system prompt enforces this and uses
   second-person evidence-grounded phrasing; see
   ``agent/memory/procedural_prompts.py`` for the prompt design notes.

2. **Silent skip on incognito or no LLM.** Incognito mode is a privacy
   promise: no long-term writes to the memory store. The writer node
   no-ops without logging an error. Same contract as the semantic
   extractor.

3. **Failures degrade silently.** LLM errors, schema validation errors,
   and store write errors are all logged at WARNING level but never
   propagate. The writer is a side-effect node; a failure here must
   not fail the parent turn.

4. **Always return an empty state delta.** The write is a side effect;
   state isn't modified. Returning ``{}`` is the canonical "I touched
   nothing" signal for LangGraph delta-return nodes.

5. **Policy before persistence.** Extracted rule drafts become
   :class:`ProceduralCandidate` instances first. The deterministic
   policy can downgrade a candidate to ``commit_at_session_end`` or
   ``drop``. Non-``commit_now`` outcomes are diagnostics-only on the
   hot path and can be reviewed at session end.

6. **Conservative reconciliation.** New writes reconcile against the existing profile:
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
from agent.memory.write_policy import decide_procedural_candidate_llm_primary
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

    Args:
        state: Current graph state after response generation.
        runtime: LangGraph runtime carrying memory dependencies.

    Returns:
        State delta containing procedural writer diagnostics.
    """

    # Time the full writer call and report write counts via diagnostics.
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
        """Return a state delta carrying just the writer's diagnostics.

        Args:
            procedural_writes: Number of procedural rules written.
            procedural_candidates: Total LLM candidate count.
            procedural_commit_now_candidates: Candidates eligible for immediate write.
            procedural_session_end_holds: Candidates buffered until session end.
            procedural_policy_drops: Candidates rejected by deterministic policy.
            reason: Human-readable writer or skip reason.

        Returns:
            State delta with diagnostics only.
        """

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

    # Skip extraction on high-priority operational routes for the same reason
    # as semantic extraction: these replies are not useful procedural material.
    route = state.get("route")
    if route in {"crisis", "memory_control", "grounded_lookup"}:
        skip_reason = (
            "crisis_path"
            if route == "crisis"
            else "memory_control_path"
            if route == "memory_control"
            else "grounded_lookup_path"
        )
        logger.debug(
            "extract_procedural_rules_node: %s; skipping extraction",
            skip_reason,
        )
        return _diagnostics_delta(reason=f"skipped: {skip_reason}")

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

    session_progress = state.get("session_progress", {})
    turn_count = int(session_progress.get("turn_count", 1))
    turn_index = max(0, turn_count - 1)

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

    # Log the reason regardless of whether rules were produced; the
    # conservative writer rejects most turns, and the reason is the fastest
    # way to catch prompt drift.
    logger.info(
        "extract_procedural_rules_node: %d rules, reason=%r",
        len(result.rules),
        result.reason,
    )

    if not result.rules:
        return _diagnostics_delta(reason=result.reason)

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
        decision = await decide_procedural_candidate_llm_primary(
            candidate,
            llm_client=llm_client,
        )

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

    # Each surviving candidate gets promoted to a ProceduralRule via
    # build_procedural_rule (which adds the added_at timestamp and
    # source="explicit_user"), then upserted into the user's profile.
    # The helper handles the load-reconcile-put idiom internally;
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
            upsert = await aupsert_procedural_rule(
                store,
                user_id=owner_id,
                rule=rule,
                llm_client=llm_client,
            )
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
