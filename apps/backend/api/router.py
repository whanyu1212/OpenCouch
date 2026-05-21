from fastapi import APIRouter

from api.routes import chat, health, memory, threads

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(threads.router)
api_router.include_router(memory.router)
