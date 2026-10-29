"""
Human approval gate for AMBER actions. Ported from Mycel `core/approval_gate.py`.

What changed for scale: Mycel parked each request in a process-local dict, so
the click had to land on the same process that was waiting. Here the request
is stored, the waiting agent listens on a bus topic for its approval id, and
`resolve()` — on any API replica — writes the decision and pings that topic.
The store is the source of truth; the bus only wakes the waiter.

Fail-safe: no answer before the deadline means EXPIRED, which is a denial.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.errors import ConflictError, NotFoundError
from app.core.ids import now_ms
from app.domain.events import EventKind, LogLevel
from app.domain.governance import Approval, ApprovalStatus
from app.infrastructure.bus.base import EventBus
from app.infrastructure.storage.base import ApprovalRepository
from app.security.policy import Verdict
from app.services.event_publisher import EventPublisher
from app.tools.registry import ToolContext, preview_args

log = logging.getLogger(__name__)


def approval_topic(approval_id: str) -> str:
    return f"approval.{approval_id}"


class ApprovalGate:
    def __init__(
        self,
        repo: ApprovalRepository,
        bus: EventBus,
        events: EventPublisher,
        *,
        timeout_s: float = 300,
        auto_approve_categories: set[str] | None = None,
    ) -> None:
        self._repo = repo
        self._bus = bus
        self._events = events
        self._timeout_s = timeout_s
        self._auto = auto_approve_categories or set()

    # ── Approver protocol (called by the tool registry) ──────────────────
    async def request(self, ctx: ToolContext, tool: str, args: dict[str, Any], verdict: Verdict) -> bool:
        if verdict.category in self._auto:
            await self._events.log(
                ctx.mission_id, f"{ctx.agent} → {tool}: auto-approved by configuration ({verdict.reason})", LogLevel.ARMOR
            )
            return True
        approval = Approval(
            mission_id=ctx.mission_id,
            agent=ctx.agent,
            tool=tool,
            action_class=verdict.action_class.value,
            category=verdict.category,
            reason=verdict.reason,
            args_preview=str(preview_args(args))[:600],
            expires_at=now_ms() + int(self._timeout_s * 1000),
        )
        decided = await self.wait_for_decision(approval)
        return decided.approved

    async def wait_for_decision(self, approval: Approval) -> Approval:
        # Listen before announcing, so a very fast click cannot be missed.
        async with self._bus.subscribe(approval_topic(approval.approval_id)) as decisions:
            await self._repo.create(approval)
            await self._events.publish(approval.mission_id, EventKind.APPROVAL_REQUEST, approval.request_payload())
            log.info("approval %s requested: %s → %s", approval.approval_id, approval.agent, approval.tool)
            try:
                await asyncio.wait_for(anext(decisions), timeout=self._timeout_s)
            except TimeoutError:
                await self._repo.decide(
                    approval.approval_id, ApprovalStatus.EXPIRED, by="system:timeout",
                    note=f"no answer within {self._timeout_s:.0f}s", at=now_ms(),
                )
        final = await self._repo.get(approval.approval_id) or approval
        await self._events.publish(final.mission_id, EventKind.APPROVAL_RESPONSE, final.response_payload())
        log.info("approval %s resolved: %s", final.approval_id, final.status)
        return final

    # ── operator side (called by the API) ────────────────────────────────
    async def resolve(self, approval_id: str, approved: bool, *, by: str, note: str | None = None) -> Approval:
        status = ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED
        decided = await self._repo.decide(approval_id, status, by=by, note=note, at=now_ms())
        if decided is None:
            existing = await self._repo.get(approval_id)
            if existing is None:
                raise NotFoundError(f"Approval {approval_id} not found")
            raise ConflictError(f"Approval {approval_id} is already {existing.status}",
                                details={"status": existing.status})
        await self._bus.publish(approval_topic(approval_id), {"approval_id": approval_id, "status": status})
        return decided

    async def expire_pending(self, mission_id: str, reason: str) -> int:
        """Deny everything still waiting for a mission (used when a mission is stopped)."""
        count = 0
        for approval in await self._repo.list(mission_id, status=ApprovalStatus.PENDING):
            decided = await self._repo.decide(approval.approval_id, ApprovalStatus.EXPIRED,
                                              by="system:stop", note=reason, at=now_ms())
            if decided is not None:
                count += 1
                await self._bus.publish(approval_topic(approval.approval_id),
                                        {"approval_id": approval.approval_id, "status": decided.status})
        return count
