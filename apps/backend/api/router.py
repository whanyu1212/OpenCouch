from fastapi import APIRouter

from api.routes import chat, health, memory, threads
from voice.api import router as voice_router
from voice.livekit.api import router as livekit_voice_router

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(threads.router)
api_router.include_router(memory.router)
api_router.include_router(voice_router)
api_router.include_router(livekit_voice_router)
