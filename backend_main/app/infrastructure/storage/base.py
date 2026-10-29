"""Persistence interfaces. Each has an in-memory and a MongoDB implementation."""
from __future__ import annotations

from typing import Protocol

from app.domain.events import EventEnvelope
from app.domain.governance import Approval, ApprovalStatus, AuditEntry
from app.domain.mission import AgentMessage, Evidence, Mission, MissionStatus


class EventStore(Protocol):
    async def append(self, event: EventEnvelope) -> EventEnvelope:
        """Persist `event`, assigning its per-mission `seq`. Returns the stored event."""
        ...

    async def list(
        self, mission_id: str, *, after_seq: int = 0, limit: int = 5000
    ) -> list[EventEnvelope]: ...


class ApprovalRepository(Protocol):
    async def create(self, approval: Approval) -> Approval: ...

    async def get(self, approval_id: str) -> Approval | None: ...

    async def decide(
        self, approval_id: str, status: ApprovalStatus, *, by: str, note: str | None, at: int
    ) -> Approval | None:
        """Atomically move a PENDING approval to `status`. Returns None if it was not pending."""
        ...

    async def list(self, mission_id: str, *, status: ApprovalStatus | None = None) -> list[Approval]: ...


class AuditLog(Protocol):
    async def append(self, entry: AuditEntry) -> None: ...

    async def list(
        self, mission_id: str, *, agent: str | None = None, limit: int = 1000
    ) -> list[AuditEntry]: ...


class MissionRepository(Protocol):
    async def create(self, mission: Mission) -> Mission: ...

    async def get(self, mission_id: str) -> Mission | None: ...

    async def save(self, mission: Mission) -> Mission:
        """Save if nobody else saved since `mission.version` was loaded; raises ConcurrencyConflict otherwise.
        Returns the mission with its new version."""
        ...

    async def list(self, *, status: MissionStatus | None = None, limit: int = 50) -> list[Mission]: ...


class EvidenceRepository(Protocol):
    async def add(self, evidence: Evidence) -> Evidence: ...

    async def get_many(self, mission_id: str, evidence_ids: list[str]) -> list[Evidence]: ...

    async def list(
        self, mission_id: str, *, kind: str | None = None, task_id: str | None = None, limit: int = 500
    ) -> list[Evidence]: ...


class MessageRepository(Protocol):
    async def post(self, message: AgentMessage) -> AgentMessage:
        """Persist with the next per-mission `seq`."""
        ...

    async def list(self, mission_id: str, *, after_seq: int = 0, limit: int = 500) -> list[AgentMessage]: ...


class ConcurrencyConflict(Exception):
    """The mission was changed by someone else since it was loaded."""


class Storage(Protocol):
    """One storage backend: its lifecycle plus every repository it provides."""

    events: EventStore
    approvals: ApprovalRepository
    audit: AuditLog
    missions: MissionRepository
    evidence: EvidenceRepository
    messages: MessageRepository

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def ping(self) -> bool: ...
