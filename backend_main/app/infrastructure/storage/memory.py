"""In-memory storage for local runs and tests. Not shared between processes."""
from __future__ import annotations

import asyncio
from collections import defaultdict

from app.domain.events import EventEnvelope
from app.domain.governance import Approval, ApprovalStatus, AuditEntry
from app.domain.mission import AgentMessage, Evidence, Mission, MissionStatus
from app.infrastructure.storage.base import ConcurrencyConflict


class InMemoryEventStore:
    def __init__(self) -> None:
        self._events: dict[str, list[EventEnvelope]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def append(self, event: EventEnvelope) -> EventEnvelope:
        async with self._lock:
            stream = self._events[event.mission_id]
            stored = event.model_copy(update={"seq": len(stream) + 1})
            stream.append(stored)
            return stored

    async def list(
        self, mission_id: str, *, after_seq: int = 0, limit: int = 5000
    ) -> list[EventEnvelope]:
        stream = self._events.get(mission_id, [])
        return [e for e in stream if (e.seq or 0) > after_seq][:limit]


class InMemoryApprovalRepository:
    def __init__(self) -> None:
        self._items: dict[str, Approval] = {}

    async def create(self, approval: Approval) -> Approval:
        self._items[approval.approval_id] = approval
        return approval

    async def get(self, approval_id: str) -> Approval | None:
        return self._items.get(approval_id)

    async def decide(
        self, approval_id: str, status: ApprovalStatus, *, by: str, note: str | None, at: int
    ) -> Approval | None:
        # No await between the check and the write, so this is atomic on one event loop.
        current = self._items.get(approval_id)
        if current is None or current.status != ApprovalStatus.PENDING:
            return None
        decided = current.model_copy(update={"status": status, "decided_by": by, "note": note, "decided_at": at})
        self._items[approval_id] = decided
        return decided

    async def list(self, mission_id: str, *, status: ApprovalStatus | None = None) -> list[Approval]:
        return sorted(
            (a for a in self._items.values() if a.mission_id == mission_id and (status is None or a.status == status)),
            key=lambda a: a.created_at,
        )


class InMemoryAuditLog:
    def __init__(self) -> None:
        self._entries: dict[str, list[AuditEntry]] = defaultdict(list)

    async def append(self, entry: AuditEntry) -> None:
        self._entries[entry.mission_id].append(entry)

    async def list(self, mission_id: str, *, agent: str | None = None, limit: int = 1000) -> list[AuditEntry]:
        entries = [e for e in self._entries.get(mission_id, []) if agent is None or e.agent == agent]
        return entries[-limit:]


class InMemoryMissionRepository:
    def __init__(self) -> None:
        self._items: dict[str, Mission] = {}

    async def create(self, mission: Mission) -> Mission:
        if mission.mission_id in self._items:
            raise ConcurrencyConflict(f"mission {mission.mission_id} already exists")
        stored = mission.model_copy(deep=True, update={"version": 1})
        self._items[mission.mission_id] = stored
        return stored.model_copy(deep=True)

    async def get(self, mission_id: str) -> Mission | None:
        stored = self._items.get(mission_id)
        # Hand out copies: callers mutate what they load, and must save to make it real.
        return stored.model_copy(deep=True) if stored else None

    async def save(self, mission: Mission) -> Mission:
        current = self._items.get(mission.mission_id)
        if current is None or current.version != mission.version:
            raise ConcurrencyConflict(f"mission {mission.mission_id} changed since it was loaded")
        stored = mission.model_copy(deep=True, update={"version": mission.version + 1})
        self._items[mission.mission_id] = stored
        return stored.model_copy(deep=True)

    async def list(self, *, status: MissionStatus | None = None, limit: int = 50) -> list[Mission]:
        items = [m for m in self._items.values() if status is None or m.status == status]
        items.sort(key=lambda m: m.created_at, reverse=True)
        return [m.model_copy(deep=True) for m in items[:limit]]


class InMemoryEvidenceRepository:
    def __init__(self) -> None:
        self._items: dict[str, list[Evidence]] = defaultdict(list)

    async def add(self, evidence: Evidence) -> Evidence:
        self._items[evidence.mission_id].append(evidence)
        return evidence

    async def get_many(self, mission_id: str, evidence_ids: list[str]) -> list[Evidence]:
        wanted = set(evidence_ids)
        return [e for e in self._items.get(mission_id, []) if e.evidence_id in wanted]

    async def list(
        self, mission_id: str, *, kind: str | None = None, task_id: str | None = None, limit: int = 500
    ) -> list[Evidence]:
        items = [e for e in self._items.get(mission_id, [])
                 if (kind is None or e.kind == kind) and (task_id is None or e.task_id == task_id)]
        return items[-limit:]


class InMemoryMessageRepository:
    def __init__(self) -> None:
        self._items: dict[str, list[AgentMessage]] = defaultdict(list)

    async def post(self, message: AgentMessage) -> AgentMessage:
        stream = self._items[message.mission_id]
        stored = message.model_copy(update={"seq": len(stream) + 1})
        stream.append(stored)
        return stored

    async def list(self, mission_id: str, *, after_seq: int = 0, limit: int = 500) -> list[AgentMessage]:
        return [m for m in self._items.get(mission_id, []) if (m.seq or 0) > after_seq][:limit]


class InMemoryStorage:
    def __init__(self) -> None:
        self.events = InMemoryEventStore()
        self.approvals = InMemoryApprovalRepository()
        self.audit = InMemoryAuditLog()
        self.missions = InMemoryMissionRepository()
        self.evidence = InMemoryEvidenceRepository()
        self.messages = InMemoryMessageRepository()

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def ping(self) -> bool:
        return True
