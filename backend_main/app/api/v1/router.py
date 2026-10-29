from fastapi import APIRouter

from app.api.v1.routes import governance, guild, health, missions, realtime, tools

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(tools.router)
api_router.include_router(guild.router)
api_router.include_router(missions.router)
api_router.include_router(realtime.router)
api_router.include_router(governance.router)
