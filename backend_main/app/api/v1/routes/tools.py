"""The tool catalogue: what agents can do, with each tool's risk and category."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import ContainerDep

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def list_tools(container: ContainerDep) -> dict[str, Any]:
    catalog = container.tools.catalog()
    return {"count": len(catalog), "tools": catalog}
