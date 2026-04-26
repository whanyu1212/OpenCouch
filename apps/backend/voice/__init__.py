"""Voice chat module — two coexisting implementations for A/B testing.

Option B (voice/realtime.py, voice/api.py):
    Direct WebSocket proxy between the browser and the OpenAI Realtime
    API. Simple, low-latency, but no tool calling or agent routing.

Option C (voice/livekit/):
    LiveKit Agents SDK with native Agent, AgentTask, and @function_tool
    primitives. Supports multi-agent handoffs, structured grounding
    exercises, and per-turn memory retrieval. Runs as a separate
    worker process.
"""
