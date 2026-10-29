"""The guild roster and each mission's task force (who Maya hired, why, and with what mandate)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.agents.guild import clearance
from app.api.deps import ContainerDep
from app.core.errors import NotFoundError, ValidationFailed

router = APIRouter(tags=["guild"])


class HireBody(BaseModel):
    skills: list[str] = Field(min_length=1, description="What the work needs, most important first")
    reason: str = Field(min_length=3, max_length=500)


class ReleaseBody(BaseModel):
    reason: str = Field(default="released by operator", max_length=500)


@router.get("/agents")
async def roster(container: ContainerDep) -> dict[str, Any]:
    return {"agents": [
        {"agent_id": d.agent_id, "name": d.name, "team": d.team, "role": d.role, "skills": d.skills,
         "tools": d.tools, "badge": d.badge, "char_idx": d.char_idx, "always_on": d.always_on,
         "clearance": clearance(d), "default_mandate": d.default_mandate}
        for d in container.guild.all()
    ]}


@router.get("/missions/{mission_id}/hires")
async def mission_hires(mission_id: str, container: ContainerDep) -> dict[str, Any]:
    mission = await container.board.get(mission_id)
    return {"mission_id": mission_id, "active": mission.hired_agents,
            "hires": [h.model_dump(mode="json") for h in mission.hires]}


@router.post("/missions/{mission_id}/hires")
async def hire(mission_id: str, body: HireBody, container: ContainerDep) -> dict[str, Any]:
    """Ask Maya for a specialist (an operator override; Atlas does this on its own during a mission)."""
    try:
        records = await container.hiring.hire_for_skills(mission_id, body.skills, body.reason)
    except ValueError as exc:
        raise ValidationFailed(str(exc)) from None
    if not records:
        raise NotFoundError(f"No agent in the guild covers {', '.join(body.skills)}")
    return {"hired": [r.model_dump(mode="json") for r in records]}


@router.post("/missions/{mission_id}/hires/{agent}/release")
async def release(mission_id: str, agent: str, body: ReleaseBody, container: ContainerDep) -> dict[str, Any]:
    record = await container.hiring.release(mission_id, agent, body.reason)
    return {"released": record.model_dump(mode="json")}
