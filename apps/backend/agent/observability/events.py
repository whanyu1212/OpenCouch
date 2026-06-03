"""Canonical span and event names for agent observability."""

from __future__ import annotations

RUNTIME_TEXT_TURN = "runtime.text_turn"
RUNTIME_VOICE_SESSION = "runtime.voice_session"
ROUTING_DECISION = "routing.decision"
MEMORY_READ = "memory.read"
MEMORY_WRITE_POLICY = "memory.write_policy"
SDK_OPENAI_CALL = "sdk.openai.call"
TOOL_DISPATCH = "tool.dispatch"
SAFETY_ASSESS = "safety.assess"
AUDIT_CRISIS_LOG_APPEND = "audit.crisis_log.append"
GUIDED_EXERCISE_STEP = "guided_exercise.step"
