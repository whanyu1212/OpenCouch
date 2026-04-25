# UX Tooling Enhancements

Reference backlog for user-experience improvements that add graph nodes,
function tools, or tool-routing patterns to the OpenCouch agent.

## Current Status

| Enhancement | Status | Purpose |
| --- | --- | --- |
| `crisis_resource_lookup_node` | Implemented | Modular crisis resource lookup before crisis response. Handles location-aware crisis resources without bloating `crisis_response_node`. |
| Conversational memory control | Implemented | Lets users manage memory in normal chat: list memory, recall on/off, save preferences, and forget/delete memories with confirmation. Implemented as `memory_control_gate_node` plus `memory_control_node`. |
| Exercise state hardening | Deferred | More explicit exercise state machine for start, continue, side-turn, exit, and completion handling. Deferred because current guided-exercise coverage is small and too much rigidity may reduce conversational flexibility. |
| Non-crisis `local_resource_lookup_node` | Not started; partially covered by grounded lookup | Finds support resources for non-imminent needs such as therapists, clinics, grief groups, domestic violence support, financial/legal aid, and peer support. Should require user-provided location or clear consent. General explicit lookups can route through grounded factual lookup, but a curated local-resource node is still separate. |
| In-session recap/takeaway node | Addressed via `closing` mode | User-triggered wrap-up takeaways such as "what is the main takeaway?" now stay in `closing` mode. We avoided a separate recap node because session-end summarization already exists and closing-mode prompt/eval coverage is sufficient for now. |
| Handoff export node | Not started | Generates a user-controlled summary for a therapist, doctor, coach, or personal notes. Should avoid diagnosis and clearly label uncertainty. |
| Mood/check-in tracking | UI-owned; agent-consumed | Opt-in structured check-ins for mood rating, stress level, sleep, triggers, and coping attempts. This should be captured through UI/API controls, then surfaced to the agent as retrieved context when relevant. Avoid an agent-routing node unless chat-based logging becomes a clear requirement. |
| Grounded factual lookup / web-search routing | Implemented | Handles explicit current or factual lookup requests with provider-native search grounding instead of guessing. Implemented as `grounded_lookup_gate_node` plus `grounded_answer_node`; ordinary therapy turns bypass it. |
| Reminder/follow-up planning | Deferred | Lets users set follow-up intentions or reminders. Not worth implementing until there is an actual reminder, calendar, or notification backend. |
| Session plan node | Addressed via opening prompt behavior | Low-content openings may get one optional orientation question about what the user wants from the session. Avoided a graph node because the behavior is conversational and should not interrupt emotionally loaded openings. |
| Generic tool-call subgraph architecture | Conceptual only | Future shared LLM tool-call loop for lower-stakes tools. High-stakes flows such as crisis and memory should continue using explicit nodes. |

## Suggested Priority

1. Handoff export node.
2. Non-crisis local resource lookup, if we want curated resource behavior beyond general grounded search.
3. Reminder/follow-up planning after backend support exists.
4. Exercise state hardening if guided exercises expand significantly.
5. Generic tool-call subgraph only if several low-stakes tools need shared orchestration.

## Implemented Or Addressed

- `crisis_resource_lookup_node`.
- Conversational memory control.
- In-session wrap-up takeaways, handled through `closing` mode.
- Grounded factual lookup / web-search routing.
- Mood/check-in tracking, reclassified as UI-owned with agent consumption.
- Session plan behavior, handled as optional low-content opening guidance.

## Architecture Notes

Use explicit graph nodes when the action has safety, privacy, persistence, or
auditing implications. This keeps state ownership clear and avoids hidden tool
side effects inside therapeutic response generation.

Use function tools or a shared tool-call subgraph only for lower-stakes,
bounded actions where failed or skipped tool calls do not affect safety-critical
behavior. Candidate examples include factual lookup, optional summaries, or
future lightweight planning helpers.

Resource lookup nodes should never infer location from IP by default. Ask for
or use a user-provided location, and keep crisis lookup separate from
non-crisis local resources.

## Session Summary Boundary

`run_summarize_session` is implemented as a session-end persistence
path. It runs when the runtime ends a session, writes an episodic arc, and the
CLI renders the saved summary so the user can see what will be remembered.

Conversation-time recap requests such as "what have we figured out so far?",
"what is the main takeaway?", or "can you summarize this before I go?" are
handled through `closing` mode. They should remain user-facing and
non-persistent by default unless the user also asks to save them.
