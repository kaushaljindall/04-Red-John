"""
One contract, every storage backend. The in-memory backend always runs; MongoDB runs when
DEVPILOT_TEST_MONGO_URL points at a disposable server (each run uses its own database and drops it).
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from app.core.ids import now_ms
from app.domain.events import EventEnvelope, EventKind
from app.domain.governance import Approval, ApprovalStatus, AuditEntry, AuditOutcome
from app.domain.mission import AgentMessage, Evidence, Finding, Mission, MissionStatus
from app.infrastructure.storage.base import ConcurrencyConflict
from app.infrastructure.storage.memory import InMemoryStorage

MONGO_URL = os.environ.get("DEVPILOT_TEST_MONGO_URL")


@pytest.fixture(params=["memory", "mongo"])
async def storage(request):
    if request.param == "memory":
        yield InMemoryStorage()
        return
    if not MONGO_URL:
        pytest.skip("set DEVPILOT_TEST_MONGO_URL to run against a real MongoDB")
    from app.infrastructure.storage.mongo import MongoStorage

    name = f"devpilot_test_{uuid.uuid4().hex[:8]}"
    store = MongoStorage(MONGO_URL, name)
    await store.start()
    try:
        yield store
    finally:
        await store._conn.client.drop_database(name)
        await store.stop()


async def test_events_are_sequenced_per_mission_even_when_written_concurrently(storage):
    await asyncio.gather(*(storage.events.append(EventEnvelope(mission_id="m1", kind=EventKind.LOG, data={"i": i}))
                           for i in range(20)))
    await storage.events.append(EventEnvelope(mission_id="m2", kind=EventKind.LOG))
    events = await storage.events.list("m1")
    assert [e.seq for e in events] == list(range(1, 21))
    assert [e.seq for e in await storage.events.list("m1", after_seq=18)] == [19, 20]
    assert [e.seq for e in await storage.events.list("m2")] == [1]


async def test_approvals_decide_exactly_once(storage):
    approval = await storage.approvals.create(Approval(mission_id="m1", agent="Takeshi", tool="run_command",
                                                       reason="install", expires_at=now_ms() + 60_000))
    first, second = await asyncio.gather(
        storage.approvals.decide(approval.approval_id, ApprovalStatus.APPROVED, by="a", note=None, at=now_ms()),
        storage.approvals.decide(approval.approval_id, ApprovalStatus.DENIED, by="b", note=None, at=now_ms()))
    winners = [d for d in (first, second) if d is not None]
    assert len(winners) == 1
    stored = await storage.approvals.get(approval.approval_id)
    assert stored.status == winners[0].status and stored.decided_by == winners[0].decided_by
    assert [a.approval_id for a in await storage.approvals.list("m1", status=stored.status)] == [approval.approval_id]
    assert await storage.approvals.list("m1", status=ApprovalStatus.PENDING) == []


async def test_missions_use_optimistic_concurrency(storage):
    mission = await storage.missions.create(Mission(title="t", request="r", repo_path="/r"))
    assert mission.version == 1
    with pytest.raises(ConcurrencyConflict):
        await storage.missions.create(mission)
    a = await storage.missions.get(mission.mission_id)
    b = await storage.missions.get(mission.mission_id)
    a.add_finding(Finding(agent="A", kind="note", title="from a"))
    saved = await storage.missions.save(a)
    assert saved.version == 2
    b.add_finding(Finding(agent="B", kind="note", title="from b"))
    with pytest.raises(ConcurrencyConflict):
        await storage.missions.save(b)                 # b was loaded before a saved
    reloaded = await storage.missions.get(mission.mission_id)
    assert [f.title for f in reloaded.findings] == ["from a"]
    reloaded.transition(MissionStatus.PLANNING)
    await storage.missions.save(reloaded)
    assert [m.mission_id for m in await storage.missions.list(status=MissionStatus.PLANNING)] == [mission.mission_id]
    assert await storage.missions.get("msn_missing") is None


async def test_evidence_messages_and_audit(storage):
    e1 = await storage.evidence.add(Evidence(mission_id="m1", kind="test_run", agent="Aiko", tool="run_tests",
                                             passed=False, task_id="t1"))
    e2 = await storage.evidence.add(Evidence(mission_id="m1", kind="diff", agent="Haruto", tool="git_diff"))
    assert {e.evidence_id for e in await storage.evidence.get_many("m1", [e1.evidence_id, "evd_x"])} == {e1.evidence_id}
    assert [e.evidence_id for e in await storage.evidence.list("m1", kind="test_run", task_id="t1")] == [e1.evidence_id]
    assert len(await storage.evidence.list("m1")) == 2 and e2.passed is None

    posted = await asyncio.gather(*(storage.messages.post(AgentMessage(mission_id="m1", sender="A", subject=str(i)))
                                    for i in range(5)))
    assert sorted(m.seq for m in posted) == [1, 2, 3, 4, 5]
    assert [m.seq for m in await storage.messages.list("m1", after_seq=3)] == [4, 5]

    for agent in ("Takeshi", "Aiko", "Takeshi"):
        await storage.audit.append(AuditEntry(mission_id="m1", agent=agent, tool="read_file", outcome=AuditOutcome.OK))
    assert [a.agent for a in await storage.audit.list("m1", agent="Takeshi")] == ["Takeshi", "Takeshi"]
    assert len(await storage.audit.list("m1", limit=2)) == 2
    assert await storage.ping()
