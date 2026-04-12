"""Phase-1 therapeutic subgraph signature sketch.

This file is a **design artifact**, not production code. It exists to
document the signatures, return shapes, and internal topology of the
therapeutic response subgraph that will be implemented alongside the
memory-layer phase-1 rebuild.

Why a sketch instead of an implementation?
- The therapeutic branch needs design closure on six mode definitions
  and one dispatch policy before any code is worth writing. Sketching
  surfaces the signature mismatches (between the subgraph and the
  parent ``build_agent_workflow``) before any data shapes get locked.
- It serves as the bridge between the top-level graph topology and
  the eventual per-mode implementation files in ``agent/therapeutic/``.
- It pairs with ``agent/memory/nodes_sketch.py``; the two sketches
  together describe the full phase-1 graph surface.

How to read this file:
- Each function has a **complete signature** (types, parameters, return)
  and a **detailed docstring**, but the body is just ``...``.
- Mode nodes are grouped by the dispatch keyword that routes to them.
- Every node declares which ``AgentState`` fields it reads and which
  it writes.
- The ``build_therapeutic_subgraph`` factory at the bottom shows how
  the nodes are assembled into a compiled ``StateGraph`` that the
  parent graph can embed as a single node.
- "Proposed subgraph topology" block at the bottom shows the flow.

This file is **not** imported by anything in production. It is safe to
modify, delete, or migrate piece-by-piece into concrete node files
when phase 1 implementation actually begins.

Status: design draft, locked against framing decisions (2026-04-10).
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.runtime_context import WorkflowContext
from agent.state import AgentState


# ─── Therapeutic modes: the six response-style options ──────────────────────
#
# Each mode is a distinct "way of responding" that the dispatcher routes to.
# The mode set is deliberately small (6 options) to keep dispatch reliable;
# more modes make LLM dispatch less consistent. Modality overlays (CBT, ACT,
# PFA, grief_support, etc.) are DEFERRED to phase 2 — phase 1 ships with
# mode selection only, no modality layering.
#
# Ordered by expected frequency:
#   1. supportive         (default, most turns)
#   2. reflective         (pattern recognition)
#   3. psychoeducation    (normalizing explanations)
#   4. guided_exercise    (grounding, breathing, steps)
#   5. closing            (winding down, tonal)
#   6. clarifying         (ambiguous message)

TherapeuticMode = Literal[
    "supportive",
    "reflective",
    "psychoeducation",
    "guided_exercise",
    "closing",
    "clarifying",
]

# Node names used inside the therapeutic subgraph. Keeping them as module-
# level constants makes the routing table in build_therapeutic_subgraph
# trivially greppable and guards against typos.
DISPATCH_NODE = "therapeutic_dispatch_node"
SUPPORTIVE_NODE = "supportive_response_node"
REFLECTIVE_NODE = "reflective_response_node"
PSYCHOEDUCATION_NODE = "psychoeducation_response_node"
GUIDED_EXERCISE_NODE = "guided_exercise_response_node"
CLOSING_NODE = "closing_response_node"
CLARIFYING_NODE = "clarifying_response_node"


# ─── Dispatch node ──────────────────────────────────────────────────────────


async def therapeutic_dispatch_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> Command[
    Literal[
        "supportive_response_node",
        "reflective_response_node",
        "psychoeducation_response_node",
        "guided_exercise_response_node",
        "closing_response_node",
        "clarifying_response_node",
    ]
]:
    """Select which therapeutic response mode should handle this turn.

    This node is the entry point of the therapeutic subgraph. It inspects
    the current user message, the retrieved working memory, and the
    session progress, then returns a ``Command(goto=...)`` that routes
    to one of the six mode-specific response nodes.

    Dispatch policy (phase 1):

    The dispatcher is a small structured-output LLM call (temperature 0,
    Pydantic ``DispatchDecision`` schema) that returns one of the six
    ``TherapeuticMode`` values plus a brief reasoning string. The LLM is
    given:
        - The current user message
        - The last ~6 turns of history
        - The top ~3 items from ``working_memory`` (if present)
        - A short summary of ``memory.active_concerns`` and
          ``memory.open_loops`` (if present)

    The prompt instructs the model to pick the single best mode. The
    default, when uncertain, is ``supportive``.

    A deterministic pre-filter can short-circuit the LLM call for a few
    obvious cases:
        - Very short message (<= 3 words) that's not a question  → supportive
        - Message that ends with "?" and contains "why" or "how"  → reflective
          or psychoeducation (let LLM decide between them)
        - Message contains "breathe", "ground me", "calm down"    → guided_exercise
        - Message contains "thanks", "I'm good", "got it"         → closing
          (low confidence; LLM override recommended)
        - Message contains "huh?", "what do you mean"             → clarifying

    These pre-filters are optimizations, not substitutes for the LLM call.
    When a pre-filter fires, the LLM is skipped; otherwise the LLM runs.

    Schema reference:
        Parent state: ``state["message"]``, ``state["history"]``,
            ``state["working_memory"]``, ``state["memory"]``
        Runtime: ``runtime.context["llm_client"]``

    Subgraph topology:
        START(subgraph) → therapeutic_dispatch_node → Command(goto=<mode>)

    Returns:
        ``Command(update={...}, goto="<mode>_response_node")`` where the
        update dict records the dispatch decision in
        ``state["routing"]["mode"]`` and ``state["routing"]["mode_source"]``.
        The mode nodes read ``routing.mode`` to know which mode they were
        dispatched as (for logging/observability; they don't use it to
        branch internally).

    Phase: 1

    Notes:
        - Failures in this node (LLM error, malformed output) fall back
          to ``supportive`` as the safe default. Log with
          ``logger.warning(..., exc_info=True)``.
        - The dispatcher's decision is ADVISORY from a cross-cutting
          standpoint: memory writes still run after ANY mode, the crisis
          gate still owns crisis routing, and the episodic summarizer
          still fires on session boundaries regardless of which mode was
          picked.
        - Dispatch to ``closing`` mode is the most load-bearing decision
          for user experience. See closing_response_node docstring for
          the ``closing_hint_shown`` hint mechanism and why this node
          does NOT directly terminate the session.
    """
    ...


# ─── Mode nodes ─────────────────────────────────────────────────────────────


async def supportive_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a supportive, validating response.

    The **default** therapeutic mode. Covers the majority of turns: the
    user is sharing, and the agent's job is to listen well, validate the
    feeling, offer light-touch reflection, and leave room for the user to
    continue.

    Response character:
        - Warm but not effusive
        - Validating without being pollyannaish
        - Short (target: 2-4 sentences, rarely more)
        - Light-touch reflection (name the feeling, don't analyze it)
        - No questions unless the user seems to want prompting
        - Never starts with "I understand" (a common LLM failure mode
          that feels hollow; the system prompt explicitly prohibits it)

    Schema reference:
        Parent state reads: ``message``, ``history``, ``working_memory``,
            ``memory.active_concerns``, ``memory.current_goal``, procedural
            rules (via retrieval.procedural injection)
        Writes: ``response.text``, ``response.kind``, ``routing.mode``,
            ``routing.mode_type``

    Subgraph topology:
        therapeutic_dispatch_node → supportive_response_node → END(subgraph)

    Phase: 1

    Returns:
        Delta dict with ``response`` and ``routing`` keys populated. Other
        state fields untouched (follows the delta-return pattern locked
        during the LangGraph best-practices refactor).

    Notes:
        - The supportive mode does NOT consult ``DID_NOT_HELP`` graph
          edges (phase 3+) because it doesn't suggest coping strategies.
          It's purely reflective.
        - If retrieval surfaced a relevant past concern and
          proactive_recall is ON, the response MAY reference the past
          concern ("Last time this came up, you mentioned..."). If
          proactive_recall is OFF (the default), the past concern shapes
          warmth and pacing but is not explicitly named.
    """
    ...


async def reflective_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a pattern-recognizing, reflective response.

    The user is describing something that looks like a pattern, or asking
    a "why does this keep happening?" type of question, or surfacing a
    recurring theme. The agent's job is to gently *name the pattern* and
    invite the user to reflect on it, without diagnosing or prescribing.

    Response character:
        - Names one pattern (not several — focus matters)
        - Grounds the naming in the user's own words when possible
          ("I notice you keep saying 'I should'...")
        - Invites reflection with one open question at most
        - Acknowledges the observation might be wrong ("Does that
          resonate, or is it more like...")
        - Longer than supportive mode (target: 3-5 sentences)

    Schema reference:
        Parent state reads: ``message``, ``history``,
            ``memory.active_concerns``, ``memory.open_loops``,
            ``working_memory`` (semantic facts about past patterns)
        Writes: ``response.text``, ``response.kind``, ``routing.mode``

    Subgraph topology:
        therapeutic_dispatch_node → reflective_response_node → END(subgraph)

    Phase: 1

    Returns:
        Delta dict with ``response`` and ``routing`` keys populated.

    Notes:
        - The reflective mode is the mode that benefits MOST from graph
          memory (phase 3+). When the graph store can return
          ``OVERLAPS_WITH`` edges between concerns, the dispatcher can
          surface cross-concern patterns that vector search alone would
          miss.
        - This mode should NEVER introduce a pattern the user hasn't
          already shown evidence for. Hallucinated patterns are the
          single worst failure mode for reflective responses.
        - If the user's message is a question ("why do I keep doing
          this?"), the response should offer a reflection, not a
          diagnosis or explanation. Explanations belong to psychoeducation.
    """
    ...


async def psychoeducation_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a normalizing, educational response.

    The user is showing confusion or surprise about their own reaction
    ("why am I crying over this?", "is it normal to feel both angry
    and relieved?"). The agent's job is to provide a short, accurate,
    non-lecturing explanation of why this reaction is common or
    well-understood in normal human psychology.

    Response character:
        - Brief factual framing, grounded in general psychology (not
          specific clinical diagnoses)
        - Uses "people often" or "it's common for" phrasing rather than
          "you are" (descriptive, not prescriptive)
        - Ends with something that brings it back to the user's
          specific situation — not generic "does that make sense?"
        - Never pathologizes — the mode's whole job is to NORMALIZE
        - Target length: 3-5 sentences

    Schema reference:
        Parent state reads: ``message``, ``history``, ``memory``,
            ``working_memory``
        Writes: ``response.text``, ``response.kind``, ``routing.mode``

    Subgraph topology:
        therapeutic_dispatch_node → psychoeducation_response_node → END(subgraph)

    Phase: 1

    Returns:
        Delta dict with ``response`` and ``routing`` keys populated.

    Notes:
        - This mode is the ONLY mode that may include mildly factual
          claims about psychology. The system prompt constrains these
          to well-established patterns and forbids specific clinical
          diagnoses. The prompt explicitly says "don't cite studies,
          don't name theories, don't diagnose."
        - If the explanation could reasonably be wrong for THIS user,
          the mode should hedge: "For a lot of people this happens
          because...; it may or may not fit your situation."
        - Avoid "should" and "need to" — psychoeducation informs; it
          doesn't prescribe.
    """
    ...


async def guided_exercise_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Walk the user through a grounding or regulation exercise.

    The user is acutely distressed or has explicitly asked for a
    technique to calm down. The agent's job is to offer ONE specific,
    simple exercise and walk the user through it step by step.

    Response character:
        - Picks one exercise (not several — choice paralysis is bad
          during distress)
        - Names the exercise ("Let's try a quick grounding technique
          called 5-4-3-2-1")
        - Gives the first step only, waits for the user to do it
        - Short, clear, present-tense instructions
        - Never longer than 5 sentences on the first step
        - Follow-up turns continue the exercise step by step

    Exercise selection logic:
        - Default: 5-4-3-2-1 grounding (sensory, universally applicable)
        - If retrieved memory shows a coping strategy marked
          ``HELPED_WITH`` for a similar concern: suggest that one
        - If retrieved memory shows a strategy marked ``DID_NOT_HELP``
          for this concern: AVOID suggesting it (critical for trust)
        - If the user explicitly asks for breathing: 4-7-8 breathing
        - If the user mentions racing thoughts: labeling exercise
          ("notice the thought, name it, let it pass")

    Schema reference:
        Parent state reads: ``message``, ``history``, ``working_memory``,
            (phase 3+) graph-store ``USES`` and ``DID_NOT_HELP`` edges
        Writes: ``response.text``, ``response.kind``, ``routing.mode``

    Subgraph topology:
        therapeutic_dispatch_node → guided_exercise_response_node → END(subgraph)

    Phase: 1

    Returns:
        Delta dict with ``response`` and ``routing`` keys populated.

    Notes:
        - The exercise-selection logic's "avoid DID_NOT_HELP" step is
          the #1 reason graph memory is worth building. Without it,
          the agent will repeatedly suggest strategies the user has
          said don't work, which is deeply frustrating.
        - Multi-turn exercises (5-4-3-2-1 takes ~5 turns) require the
          mode to track which step the user is on. The cheapest
          implementation: stash the step number in
          ``state["progress"]["stage_reason"]`` or a dedicated
          ``progress.exercise_step`` field. Deferred to phase 1
          implementation.
        - If the user seems to disengage mid-exercise, the next turn's
          dispatcher should route to supportive mode, NOT guided_exercise
          again. Let the dispatcher handle the continuation decision.
    """
    ...


async def closing_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a winding-down, closing-toned response.

    The dispatcher detected signals that the user is ready to close or
    that there's a natural lull after a productive turn. This mode
    generates a response with **tonal** closing — it acknowledges the
    arc of the conversation, leaves an open door, and offers a gentle
    farewell — WITHOUT structurally terminating the session.

    **Closing is tonal, NOT structural.** This mode does not:
        - Trigger episodic summarization
        - Modify ``progress.stage``
        - Flag the session as ended
        - Prevent the user from continuing

    Session termination is owned by the runtime layer:
        - Explicit ``/end`` command
        - 20-minute inactivity timeout
        - Runtime shutdown

    The closing mode's ONLY structural side effect is setting
    ``state["response"]["closing_hint_shown"] = True``, which is a hint
    to the runtime that it may want to shorten the inactivity timeout
    for this session (e.g., from 20 minutes to 5 minutes). The runtime
    can honor or ignore the hint.

    Response character:
        - Brief summary of the arc if memory supports it ("It sounds
          like you came in feeling X and now you're feeling Y")
        - Open door ("We can pick this back up whenever you want")
        - Gentle farewell without sounding like a dismissal
        - Target: 2-4 sentences
        - Never: "It was nice talking to you" (sounds transactional)

    Schema reference:
        Parent state reads: ``message``, ``history``,
            ``memory.active_concerns``, ``memory.open_loops``,
            ``progress.turn_count``
        Writes: ``response.text``, ``response.kind``, ``routing.mode``,
            ``response.closing_hint_shown`` (new field; see below)

    Subgraph topology:
        therapeutic_dispatch_node → closing_response_node → END(subgraph)

    Phase: 1

    Returns:
        Delta dict with ``response`` (including the new
        ``closing_hint_shown`` field), ``routing`` keys populated.

    Notes:
        - The ``closing_hint_shown`` field needs to be added to
          ``ResponseState`` in ``agent/state.py`` when this node is
          implemented. It's a ``NotRequired[bool]`` with default false.
        - If the dispatcher routes to closing but the user has
          ``open_loops`` that haven't been addressed, the response
          should acknowledge the unaddressed threads rather than ignore
          them ("You mentioned earlier that work has been tough — we
          can come back to that whenever you want").
        - This mode should NEVER run when the dispatcher is uncertain.
          False-positive closings ("oh, I thought you were done") are
          user-trust-damaging in a way that other false-positive mode
          choices aren't.
    """
    ...


async def clarifying_response_node(
    state: AgentState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Generate a single clarifying question when the message is ambiguous.

    The user's message is too short, too ambiguous, or too out-of-context
    to respond to well. Rather than guess and produce a bad response, the
    agent asks ONE focused question to get the information it needs.

    Response character:
        - Acknowledges what it heard first ("It sounds like something's
          on your mind...")
        - Asks exactly ONE question — not a list
        - The question is open-ended, not yes/no
        - Short — target: 2-3 sentences total
        - Never: "Can you tell me more?" (too generic; the whole
          point of this mode is to ask something SPECIFIC)

    When to route here (dispatcher's call):
        - Message is <= 5 words and doesn't fit a pre-filter
        - Message contains pronouns ("it", "that", "this") with no
          clear antecedent in recent history
        - Message contradicts something from earlier in the session
          without context
        - The dispatcher's own confidence is low across all other modes

    Schema reference:
        Parent state reads: ``message``, ``history``, ``working_memory``
        Writes: ``response.text``, ``response.kind``, ``routing.mode``

    Subgraph topology:
        therapeutic_dispatch_node → clarifying_response_node → END(subgraph)

    Phase: 1

    Returns:
        Delta dict with ``response`` and ``routing`` keys populated.

    Notes:
        - This is the mode I was least sure about including in the
          phase-1 set. If implementation reveals that the supportive
          mode can handle most ambiguous messages by asking gentle
          questions, this mode may get absorbed into supportive in
          phase 2.
        - The clarifying mode should NOT run multiple turns in a row.
          If the user's clarifying-response is still ambiguous, the
          dispatcher should route to supportive and let the agent make
          its best guess rather than repeatedly asking for more
          information (that pattern feels interrogative).
        - The question the agent asks should almost always be about
          CONTEXT, not CONTENT. "What brought this up?" is better than
          "What do you mean?"
    """
    ...


# ─── Subgraph assembly ──────────────────────────────────────────────────────


def build_therapeutic_subgraph() -> CompiledStateGraph:
    """Build and compile the therapeutic response subgraph.

    Returns a compiled ``StateGraph`` that can be registered as a single
    node in the parent ``build_agent_workflow`` via:

        parent.add_node("therapeutic_subgraph", build_therapeutic_subgraph())

    The subgraph shares the parent's ``AgentState`` schema, so no
    wrapper function is needed — LangGraph propagates state into and
    out of the subgraph automatically.

    Internal topology:

        START(subgraph)
          → therapeutic_dispatch_node
            → Command(goto=<one of the six mode nodes>)
          → [supportive_response_node
             | reflective_response_node
             | psychoeducation_response_node
             | guided_exercise_response_node
             | closing_response_node
             | clarifying_response_node]
          → END(subgraph)

    The dispatch node encodes its routing decision via
    ``Command(goto=...)``, so no conditional edges are needed — the
    pattern is identical to the top-level ``crisis_gate_node``.

    Context schema:
        The subgraph uses the same ``WorkflowContext`` as the parent,
        which means the dispatch and response nodes have access to
        ``llm_client``, ``memory_store`` (phase 1 memory rebuild),
        ``memory_mode``, and any other runtime deps declared in
        ``agent/runtime_context.py``.

    Phase: 1 (implementation), 2+ (additional modalities layered in)

    Returns:
        A ``CompiledStateGraph`` ready to register as a node in the
        parent graph.

    Notes:
        - Subgraph compilation is CHEAP — no significant overhead
          compared to registering nodes directly. The cognitive benefit
          (top-level graph stays small) is the main reason to use a
          subgraph.
        - The subgraph does NOT have its own checkpointer. LangGraph
          propagates the parent's checkpointer into the subgraph
          automatically when the subgraph is added as a node.
        - The subgraph does NOT do memory writes. Those live at the
          top level (``extract_semantic_facts_node`` etc.) and run
          AFTER the subgraph returns. This keeps memory concerns out
          of the therapeutic package.
    """
    subgraph = StateGraph(AgentState, context_schema=WorkflowContext)

    subgraph.add_node(DISPATCH_NODE, therapeutic_dispatch_node)
    subgraph.add_node(SUPPORTIVE_NODE, supportive_response_node)
    subgraph.add_node(REFLECTIVE_NODE, reflective_response_node)
    subgraph.add_node(PSYCHOEDUCATION_NODE, psychoeducation_response_node)
    subgraph.add_node(GUIDED_EXERCISE_NODE, guided_exercise_response_node)
    subgraph.add_node(CLOSING_NODE, closing_response_node)
    subgraph.add_node(CLARIFYING_NODE, clarifying_response_node)

    subgraph.add_edge(START, DISPATCH_NODE)
    # therapeutic_dispatch_node returns Command(goto=<mode>); no
    # conditional edge needed. The mode nodes all terminate at END.
    subgraph.add_edge(SUPPORTIVE_NODE, END)
    subgraph.add_edge(REFLECTIVE_NODE, END)
    subgraph.add_edge(PSYCHOEDUCATION_NODE, END)
    subgraph.add_edge(GUIDED_EXERCISE_NODE, END)
    subgraph.add_edge(CLOSING_NODE, END)
    subgraph.add_edge(CLARIFYING_NODE, END)

    return subgraph.compile()


# ─── Proposed parent-graph topology ─────────────────────────────────────────
#
# After this sketch is implemented, the top-level graph (agent/graph.py)
# gains ONE new node: the therapeutic subgraph. The memory-writeback nodes
# from the memory phase-1 rebuild sit after both the crisis branch and the
# therapeutic subgraph.
#
# Target parent topology:
#
#     START
#       → load_long_term_memory_node    (from memory/nodes_sketch.py)
#       → crisis_gate_node               (unchanged; Command routes here)
#       → [CRISIS BRANCH]
#            crisis_response_node
#              → crisis_log_node          (memory/nodes_sketch.py)
#              → extract_semantic_facts_node
#              → END
#       → [THERAPEUTIC BRANCH]
#            therapeutic_subgraph        (this file; compiled as one node)
#              → extract_semantic_facts_node
#              → explicit_procedural_writer_node
#              → END
#
# Parent-graph concrete wiring (in build_agent_workflow):
#
#     workflow.add_node("load_memory_node", load_long_term_memory_node)
#     workflow.add_node("crisis_gate_node", run_crisis_gate_node)
#     workflow.add_node("crisis_response_node", run_crisis_response_node)
#     workflow.add_node("crisis_log_node", crisis_log_node)
#     workflow.add_node("therapeutic_subgraph", build_therapeutic_subgraph())
#     workflow.add_node("extract_semantic_facts_node", extract_semantic_facts_node)
#     workflow.add_node("explicit_procedural_writer_node", explicit_procedural_writer_node)
#
#     workflow.add_edge(START, "load_memory_node")
#     workflow.add_edge("load_memory_node", "crisis_gate_node")
#     # crisis_gate_node Command-routes to either crisis_response_node or
#     # therapeutic_subgraph
#     workflow.add_edge("crisis_response_node", "crisis_log_node")
#     workflow.add_edge("crisis_log_node", "extract_semantic_facts_node")
#     workflow.add_edge("therapeutic_subgraph", "extract_semantic_facts_node")
#     workflow.add_edge("extract_semantic_facts_node", "explicit_procedural_writer_node")
#     workflow.add_edge("explicit_procedural_writer_node", END)
#
# Parent-graph node count: 7 (vs. 13+ if mode nodes were flat). The
# subgraph hides 7 therapeutic-specific nodes behind a single top-level
# box, which keeps the top-level file readable and the LangSmith
# visualization uncluttered.
#
# Notes:
# - crisis_gate_node needs its Command return-type annotation updated
#   from ``Literal["crisis_response_node", "__end__"]`` to
#   ``Literal["crisis_response_node", "therapeutic_subgraph"]`` when
#   this topology lands. The therapeutic branch no longer terminates
#   at END directly.
# - The memory-writeback nodes (extract_semantic_facts_node,
#   explicit_procedural_writer_node) run AFTER both branches. They're
#   cross-cutting and live at the top level, not inside either
#   subgraph or branch.
# - crisis_log_node runs ONLY on the crisis branch. It's not a
#   cross-cutting node; it's crisis-specific.
# - The parallel fan-out of extract_semantic_facts_node and
#   explicit_procedural_writer_node can be expressed as two separate
#   edges from extract_semantic_facts_node → END and
#   extract_semantic_facts_node → explicit_procedural_writer_node → END,
#   OR as two edges out of the response-producing nodes. LangGraph
#   supports parallel node execution; see phase-1 implementation for
#   the specific wiring.


# ─── Proposed AgentState changes ────────────────────────────────────────────
#
# The therapeutic subgraph as designed needs TWO new optional fields on
# ResponseState, both added when this sketch is implemented:
#
#     class ResponseState(TypedDict):
#         # ... existing fields ...
#
#         # Set by closing_response_node; hint to the runtime that it may
#         # want to shorten the inactivity timeout for this session.
#         closing_hint_shown: NotRequired[bool]
#
#         # Set by guided_exercise_response_node for multi-turn exercises.
#         # The step index within a running exercise (0-indexed). Reset
#         # to None when the exercise is abandoned or completed.
#         exercise_step: NotRequired[int | None]
#
# Both fields are ``NotRequired`` so existing state constructions don't
# need to pass them explicitly. The existing tests and fixtures are
# unaffected.


# ─── Open implementation questions ──────────────────────────────────────────
#
# These are NOT design questions (the framing is locked). They are
# implementation questions that will be answered when real code is
# written, but worth flagging here so they're not forgotten:
#
# 1. Should the dispatch node's pre-filters be deterministic regexes or
#    a small fast model call? Regexes are cheaper and more predictable;
#    a fast model call is more flexible. Lean toward regexes for the
#    obvious cases (breathe/ground me/calm down) and model call for
#    the rest.
#
# 2. Multi-turn guided_exercise: where does the exercise_step counter
#    live? In state (``progress.exercise_step``) or in the procedural
#    rules (persisted via memory)? The former is session-scoped; the
#    latter survives across sessions. Session-scoped is simpler for v1.
#
# 3. The system prompts for each mode are where the most iteration will
#    happen. They should probably live in a separate file (not here)
#    — maybe ``agent/therapeutic/prompts.py`` — so prompt tuning doesn't
#    require touching node code. Flag for phase-1 implementation.
#
# 4. Should the dispatcher's decision be stored in ``routing.mode_source``
#    as "therapeutic_dispatch"? That's consistent with how crisis_gate_node
#    sets mode_source to "crisis_gate". It means the source field tells
#    you not just which node picked the mode but which subgraph the
#    decision came from.
#
# 5. What about the cross-subgraph escape for mid-response crisis
#    detection? If a therapeutic mode node reads the latest user message
#    and detects imminent risk, it should return
#    ``Command(graph=Command.PARENT, goto="crisis_response_node")``.
#    The docs support this pattern; it should be documented in each
#    mode's docstring but NOT implemented in phase 1 (it's a rare path
#    that's hard to test). Defer to phase 2.
#
# 6. The closing mode's "closing_hint_shown" field is read by the
#    runtime (PersistentAgentRuntime.run_turn or similar). How does the
#    runtime know to check it? Probably: after each turn, the runtime
#    inspects the final state's closing_hint_shown flag and updates
#    its internal "timeout for this session" to a shorter value. Flag
#    for phase 1 runtime-layer implementation.
#
# 7. Whether to make the subgraph's dispatch decision visible in
#    LangSmith traces as a distinct span. It should be — LangGraph
#    handles this automatically as long as the subgraph is registered
#    with a name. The name "therapeutic_subgraph" should be descriptive
#    enough.
