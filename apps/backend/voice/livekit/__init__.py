"""LiveKit agentic voice implementation (Option C).

Uses LiveKit Agents SDK with native Agent, AgentTask, and
@function_tool primitives. Runs as a separate worker process
alongside the FastAPI server.

Coexists with the Option B WebSocket proxy in voice/realtime.py
for A/B testing.
"""
