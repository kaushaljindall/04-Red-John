"""
MongoDB storage. One client, shared by every repository.

- Event and message `seq` come from atomic per-mission counters, so concurrent
  writers (several workers on one mission) never collide.
- Approval decisions use a conditional update (`status == pending`), so two
  operators clicking at once, or a click racing a timeout, resolve to one answer.
- Missions use optimistic concurrency (`version`): a save only lands if nobody
  else saved since the mission was loaded.
"""
from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.domain.events import EventEnvelope
from app.domain.governance import Approval, ApprovalStatus, AuditEntry
from app.domain.mission import AgentMessage, Evidence, Mission, MissionStatus
from app.infrastructure.storage.base import ConcurrencyConflict

log = logging.getLogger(__name__)


class _Db:
    def __init__(self, url: str, database: str) -> None:
        self.url = url
        self.name = database
        self.client: AsyncIOMotorClient | None = None

    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self.client is None:
            raise RuntimeError("MongoDB used before start()")
        return self.client[self.name]


async def _next_seq(conn: _Db, key: str) -> int:
    counter = await conn.db.counters.find_one_and_update(
        {"_id": key}, {"$inc": {"value": 1}}, upsert=True, return_document=ReturnDocument.AFTER
    )
    return int(counter["value"])


class MongoEventStore:
    def __init__(self, conn: _Db) -> None:
        self._conn = conn

    async def ensure_indexes(self) -> None:
        await self._conn.db.mission_events.create_index([("mission_id", ASCENDING), ("seq", ASCENDING)], unique=True)

    async def append(self, event: EventEnvelope) -> EventEnvelope:
        stored = event.model_copy(update={"seq": await _next_seq(self._conn, f"events:{event.mission_id}")})
        await self._conn.db.mission_events.insert_one(stored.model_dump(mode="json"))
        return stored

    async def list(self, mission_id: str, *, after_seq: int = 0, limit: int = 5000) -> list[EventEnvelope]:
        cursor = (
            self._conn.db.mission_events.find({"mission_id": mission_id, "seq": {"$gt": after_seq}}, {"_id": 0})
            .sort("seq", ASCENDING)
            .limit(limit)
        )
        return [EventEnvelope.model_validate(doc) async for doc in cursor]


class MongoApprovalRepository:
    def __init__(self, conn: _Db) -> None:
        self._conn = conn

    async def ensure_indexes(self) -> None:
        await self._conn.db.approvals.create_index("approval_id", unique=True)
        await self._conn.db.approvals.create_index([("mission_id", ASCENDING), ("created_at", ASCENDING)])

    async def create(self, approval: Approval) -> Approval:
        await self._conn.db.approvals.insert_one(approval.model_dump(mode="json"))
        return approval

    async def get(self, approval_id: str) -> Approval | None:
        doc = await self._conn.db.approvals.find_one({"approval_id": approval_id}, {"_id": 0})
        return Approval.model_validate(doc) if doc else None

    async def decide(
        self, approval_id: str, status: ApprovalStatus, *, by: str, note: str | None, at: int
    ) -> Approval | None:
        doc = await self._conn.db.approvals.find_one_and_update(
            {"approval_id": approval_id, "status": ApprovalStatus.PENDING.value},
            {"$set": {"status": status.value, "decided_by": by, "note": note, "decided_at": at}},
            projection={"_id": 0},
            return_document=ReturnDocument.AFTER,
        )
        return Approval.model_validate(doc) if doc else None

    async def list(self, mission_id: str, *, status: ApprovalStatus | None = None) -> list[Approval]:
        query: dict = {"mission_id": mission_id}
        if status is not None:
            query["status"] = status.value
        cursor = self._conn.db.approvals.find(query, {"_id": 0}).sort("created_at", ASCENDING)
        return [Approval.model_validate(doc) async for doc in cursor]


class MongoAuditLog:
    def __init__(self, conn: _Db) -> None:
        self._conn = conn

    async def ensure_indexes(self) -> None:
        await self._conn.db.audit_log.create_index([("mission_id", ASCENDING), ("at", ASCENDING)])

    async def append(self, entry: AuditEntry) -> None:
        await self._conn.db.audit_log.insert_one(entry.model_dump(mode="json"))

    async def list(self, mission_id: str, *, agent: str | None = None, limit: int = 1000) -> list[AuditEntry]:
        query: dict = {"mission_id": mission_id}
        if agent:
            query["agent"] = agent
        cursor = self._conn.db.audit_log.find(query, {"_id": 0}).sort("at", DESCENDING).limit(limit)
        entries = [AuditEntry.model_validate(doc) async for doc in cursor]
        return entries[::-1]


class MongoMissionRepository:
    def __init__(self, conn: _Db) -> None:
        self._conn = conn

    async def ensure_indexes(self) -> None:
        await self._conn.db.missions.create_index("mission_id", unique=True)
        await self._conn.db.missions.create_index([("status", ASCENDING), ("created_at", DESCENDING)])

    async def create(self, mission: Mission) -> Mission:
        stored = mission.model_copy(update={"version": 1})
        try:
            await self._conn.db.missions.insert_one(stored.model_dump(mode="json"))
        except DuplicateKeyError:
            raise ConcurrencyConflict(f"mission {mission.mission_id} already exists") from None
        return stored

    async def get(self, mission_id: str) -> Mission | None:
        doc = await self._conn.db.missions.find_one({"mission_id": mission_id}, {"_id": 0})
        return Mission.model_validate(doc) if doc else None

    async def save(self, mission: Mission) -> Mission:
        stored = mission.model_copy(update={"version": mission.version + 1})
        result = await self._conn.db.missions.replace_one(
            {"mission_id": mission.mission_id, "version": mission.version}, stored.model_dump(mode="json")
        )
        if result.matched_count == 0:
            raise ConcurrencyConflict(f"mission {mission.mission_id} changed since it was loaded")
        return stored

    async def list(self, *, status: MissionStatus | None = None, limit: int = 50) -> list[Mission]:
        query = {"status": status.value} if status else {}
        cursor = self._conn.db.missions.find(query, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
        return [Mission.model_validate(doc) async for doc in cursor]


class MongoEvidenceRepository:
    def __init__(self, conn: _Db) -> None:
        self._conn = conn

    async def ensure_indexes(self) -> None:
        await self._conn.db.evidence.create_index("evidence_id", unique=True)
        await self._conn.db.evidence.create_index([("mission_id", ASCENDING), ("at", ASCENDING)])

    async def add(self, evidence: Evidence) -> Evidence:
        await self._conn.db.evidence.insert_one(evidence.model_dump(mode="json"))
        return evidence

    async def get_many(self, mission_id: str, evidence_ids: list[str]) -> list[Evidence]:
        cursor = self._conn.db.evidence.find({"mission_id": mission_id, "evidence_id": {"$in": evidence_ids}}, {"_id": 0})
        return [Evidence.model_validate(doc) async for doc in cursor]

    async def list(
        self, mission_id: str, *, kind: str | None = None, task_id: str | None = None, limit: int = 500
    ) -> list[Evidence]:
        query: dict = {"mission_id": mission_id}
        if kind:
            query["kind"] = kind
        if task_id:
            query["task_id"] = task_id
        cursor = self._conn.db.evidence.find(query, {"_id": 0}).sort("at", DESCENDING).limit(limit)
        return [Evidence.model_validate(doc) async for doc in cursor][::-1]


class MongoMessageRepository:
    def __init__(self, conn: _Db) -> None:
        self._conn = conn

    async def ensure_indexes(self) -> None:
        await self._conn.db.agent_messages.create_index([("mission_id", ASCENDING), ("seq", ASCENDING)], unique=True)

    async def post(self, message: AgentMessage) -> AgentMessage:
        stored = message.model_copy(update={"seq": await _next_seq(self._conn, f"messages:{message.mission_id}")})
        await self._conn.db.agent_messages.insert_one(stored.model_dump(mode="json"))
        return stored

    async def list(self, mission_id: str, *, after_seq: int = 0, limit: int = 500) -> list[AgentMessage]:
        cursor = (
            self._conn.db.agent_messages.find({"mission_id": mission_id, "seq": {"$gt": after_seq}}, {"_id": 0})
            .sort("seq", ASCENDING)
            .limit(limit)
        )
        return [AgentMessage.model_validate(doc) async for doc in cursor]


class MongoStorage:
    def __init__(self, url: str, database: str) -> None:
        self._conn = _Db(url, database)
        self.events = MongoEventStore(self._conn)
        self.approvals = MongoApprovalRepository(self._conn)
        self.audit = MongoAuditLog(self._conn)
        self.missions = MongoMissionRepository(self._conn)
        self.evidence = MongoEvidenceRepository(self._conn)
        self.messages = MongoMessageRepository(self._conn)

    async def start(self) -> None:
        self._conn.client = AsyncIOMotorClient(self._conn.url, serverSelectionTimeoutMS=5000, tz_aware=True)
        await self._conn.client.admin.command("ping")
        for repo in (self.events, self.approvals, self.audit, self.missions, self.evidence, self.messages):
            await repo.ensure_indexes()
        log.info("mongo storage connected (db=%s)", self._conn.name)

    async def stop(self) -> None:
        if self._conn.client is not None:
            self._conn.client.close()
            self._conn.client = None

    async def ping(self) -> bool:
        try:
            await self._conn.client.admin.command("ping")  # type: ignore[union-attr]
            return True
        except Exception:
            return False
