import logging

from fastapi import APIRouter

from api.routes import chat, health, memory, threads

logger = logging.getLogger(__name__)

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(threads.router)
api_router.include_router(memory.router)

# Voice routes depend on livekit/openai packages that live in the
# optional [voice] dependency group.  Import them conditionally so the
# API server starts without torch/livekit when only core deps are
# installed.
try:
    from agent.voice.api import router as livekit_voice_router

    api_router.include_router(livekit_voice_router)
except ImportError:
    logger.info("agent.voice.api routes not available (voice extras not installed)")
