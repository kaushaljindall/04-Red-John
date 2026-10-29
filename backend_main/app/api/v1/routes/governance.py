"""
Human oversight endpoints: answer approval requests and inspect the audit trail.

`POST /realtime/approvals/{id}/respond` is kept because the Mycel control-room
approval modal already calls it.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.deps import ContainerDep
from app.services.apply import APPLY_TOOL
from app.domain.governance import ApprovalStatus

router = APIRouter(tags=["governance"])


class ApprovalDecision(BaseModel):
    approved: bool
    note: str | None = Field(default=None, max_length=1000)


@router.get("/missions/{mission_id}/approvals")
async def list_approvals(mission_id: str, container: ContainerDep, status: ApprovalStatus | None = None) -> dict[str, Any]:
    approvals = await container.storage.approvals.list(mission_id, status=status)
    return {"mission_id": mission_id, "approvals": [a.model_dump(mode="json") for a in approvals]}


@router.post("/approvals/{approval_id}/respond")
@router.post("/realtime/approvals/{approval_id}/respond", include_in_schema=False)
async def respond_to_approval(approval_id: str, body: ApprovalDecision, container: ContainerDep) -> dict[str, Any]:
    # Phase 10 replaces "operator" with the authenticated user.
    decided = await container.approvals.resolve(approval_id, body.approved, by="operator", note=body.note)
    response = {"status": "ok", "approval_id": approval_id, "approved": decided.approved,
                "approval": decided.model_dump(mode="json")}
    if decided.tool == APPLY_TOOL:
        # Nobody is waiting on this approval: the decision itself triggers (or cancels) the apply.
        response["application"] = (await container.apply.on_decision(decided)).model_dump(mode="json")
    return response


@router.get("/missions/{mission_id}/audit")
async def mission_audit(
    mission_id: str,
    container: ContainerDep,
    agent: str | None = None,
    limit: int = Query(500, ge=1, le=5000),
) -> dict[str, Any]:
    entries = await container.storage.audit.list(mission_id, agent=agent, limit=limit)
    by_outcome: dict[str, int] = {}
    for entry in entries:
        by_outcome[entry.outcome] = by_outcome.get(entry.outcome, 0) + 1
    return {"mission_id": mission_id, "count": len(entries), "by_outcome": by_outcome,
            "entries": [e.model_dump(mode="json") for e in entries]}
