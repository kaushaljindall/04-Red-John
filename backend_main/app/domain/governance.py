"""
Governance models: human approvals, the audit trail, and mission limits.
These are what make DevPilot's autonomy *bounded*.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.core.ids import new_id, now_ms


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"     # nobody answered in time — treated as a denial


class Approval(BaseModel):
    approval_id: str = Field(default_factory=lambda: new_id("apr"))
    mission_id: str
    agent: str
    tool: str
    action_class: str = "AMBER"
    category: str = "generic"
    reason: str
    args_preview: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: int = Field(default_factory=now_ms)
    expires_at: int
    decided_at: int | None = None
    decided_by: str | None = None
    note: str | None = None

    @property
    def approved(self) -> bool:
        return self.status == ApprovalStatus.APPROVED

    def request_payload(self) -> dict[str, Any]:
        """Shape of Mycel's `approval_request` event, so the existing modal renders it."""
        return {
            "approval_id": self.approval_id,
            "agent": self.agent,
            "tool": self.tool,
            "intent": f"{self.agent} wants to call '{self.tool}'.",
            "risk": self.action_class,
            "reason": self.reason,
            "category": self.category,
            "args_preview": self.args_preview,
            "expires_at": self.expires_at,
        }

    def response_payload(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "approved": self.approved,
            "status": self.status,
            "agent": self.agent,
            "tool": self.tool,
            "class": self.action_class,
            "decided_by": self.decided_by,
            "note": self.note,
        }


class AuditOutcome(StrEnum):
    OK = "ok"
    FAILED = "failed"               # ran, but the tool reported failure (e.g. tests failed)
    BLOCKED = "blocked"             # policy said RED
    DENIED = "denied"               # AMBER and the human said no (or did not answer)
    INVALID = "invalid"             # unknown tool or bad arguments
    REFUSED_LOOP = "refused_loop"   # same failing call repeated too often
    REFUSED_BUDGET = "refused_budget"


class AuditEntry(BaseModel):
    entry_id: str = Field(default_factory=lambda: new_id("aud"))
    mission_id: str
    at: int = Field(default_factory=now_ms)
    agent: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    action_class: str | None = None
    outcome: AuditOutcome
    approved_by_human: bool = False
    error: str = ""
    display: str = ""
    duration_s: float = 0.0


class MissionLimits(BaseModel):
    """Hard ceilings for one mission. Hitting any of them stops further tool use."""

    max_tool_calls: int = 400
    max_llm_tokens: int = 2_000_000
    max_wall_clock_s: int = 45 * 60
    # The same call (same agent, tool and arguments) failing this many times is a loop.
    max_identical_failures: int = 3
    # One agent failing this many tool calls in a row has lost the plot.
    max_consecutive_failures: int = 8
