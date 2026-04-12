"""Voice chat module — OpenAI Realtime direct integration.

Option B implementation: direct WebSocket proxy between the browser
and the OpenAI Realtime API. The Realtime model handles STT, response
generation, and TTS natively. Our crisis gate runs as a synchronous
pre-check on each user transcript before Realtime generates a response.

Previous Option A (LiveKit + LangGraph LLMAdapter) is in voice/deprecated/.
"""
