"""Liveness and readiness probes (for Docker / Kubernetes / load balancers)."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.deps import ContainerDep

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(container: ContainerDep) -> JSONResponse:
    checks = {
        "storage": await container.storage.ping(),
        "event_bus": await container.event_bus.ping(),
        "llm_configured": container.llm.available,
    }
    healthy = checks["storage"] and checks["event_bus"]
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={"status": "ok" if healthy else "degraded", "checks": checks, "llm_routes": container.llm.describe()},
    )
