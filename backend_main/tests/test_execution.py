"""
Phase 8: queue, leases, workers, crash recovery, and applying to the repository behind a human.
Redis tests run only when DEVPILOT_TEST_REDIS_URL points at a disposable Redis (they flush their keys).
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import uuid

import pytest

from app.container import Container
from app.core.config import BusBackend, ExecutionMode
from app.core.errors import ConflictError, ValidationFailed
from app.domain.mission import ApplyStatus, DecisionType, MissionStatus, TaskStatus
from app.infrastructure.jobs.memory import InMemoryJobQueue, InMemoryMissionLease
from app.services.runner import MissionExecutor
from app.tools.registry import AllowAllApprover
from app.worker import MissionWorker
from tests.fake_llm import gateway
from tests.test_orchestrator import RoleScripted, everyone

REDIS_URL = os.environ.get("DEVPILOT_TEST_REDIS_URL")


# ── queue and lease contracts (run against every backend) ───────────────────
async def _queue_contract(queue) -> None:
    job = await queue.enqueue("msn_a")
    got = await queue.dequeue("w1", timeout_s=1)
    assert got.mission_id == "msn_a" and got.deliveries == 1
    assert await queue.dequeue("w2", timeout_s=0.2) is None          # in flight with w1
    await asyncio.sleep(0.4)                                           # w1 goes silent past claim_after_s
    reclaimed = await queue.dequeue("w2", timeout_s=0.2)
    assert reclaimed.mission_id == "msn_a" and reclaimed.deliveries >= 2
    await queue.ack(reclaimed)
    assert await queue.dequeue("w3", timeout_s=0.2) is None
    second = await queue.enqueue("msn_b")
    taken = await queue.dequeue("w1", timeout_s=1)
    await queue.release(taken)                                         # e.g. worker shutting down
    again = await queue.dequeue("w2", timeout_s=1)
    assert again.mission_id == "msn_b"
    await queue.ack(again)
    assert job.job_id and second.job_id


async def _lease_contract(lease) -> None:
    assert await lease.acquire("msn_x", "a", ttl_s=0.3)
    assert not await lease.acquire("msn_x", "b", ttl_s=0.3)
    assert await lease.acquire("msn_x", "a", ttl_s=0.3)                # re-entrant for the holder
    assert await lease.holder("msn_x") == "a"
    assert not await lease.renew("msn_x", "b", ttl_s=0.3)              # cannot extend someone else's
    await lease.release("msn_x", "b")                                  # nor drop it
    assert await lease.holder("msn_x") == "a"
    await asyncio.sleep(0.4)                                           # holder died: lease expires
    assert await lease.holder("msn_x") is None
    assert await lease.acquire("msn_x", "b", ttl_s=5)
    await lease.release("msn_x", "b")
    assert await lease.holder("msn_x") is None


async def test_memory_queue_and_lease():
    await _queue_contract(InMemoryJobQueue(claim_after_s=0.3))
    await _lease_contract(InMemoryMissionLease())


@pytest.mark.skipif(not REDIS_URL, reason="set DEVPILOT_TEST_REDIS_URL to run against a real Redis")
async def test_redis_queue_and_lease():
    from app.infrastructure.jobs.redis import RedisJobQueue, RedisMissionLease

    tag = uuid.uuid4().hex[:8]
    queue = RedisJobQueue(REDIS_URL, stream=f"devpilot:test:{tag}", claim_after_s=0.3)
    lease = RedisMissionLease(REDIS_URL, prefix=f"devpilot:test:{tag}:lease:")
    await queue.start()
    await lease.start()
    try:
        await _queue_contract(queue)
        await _lease_contract(lease)
    finally:
        await queue.redis.delete(f"devpilot:test:{tag}")
        await queue.stop()
        await lease.stop()


# ── executor ────────────────────────────────────────────────────────────────
class SlowOrchestrator:
    def __init__(self) -> None:
        self.runs = 0
        self.cancelled = False

    async def run(self, mission_id):
        self.runs += 1
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            self.cancelled = True
            raise

        class Done:
            status = "completed"

        return Done()


async def test_a_mission_never_runs_twice_at_once():
    lease, orchestrator = InMemoryMissionLease(), SlowOrchestrator()
    first = MissionExecutor(orchestrator, lease, "worker-1", ttl_s=5)
    second = MissionExecutor(orchestrator, lease, "worker-2", ttl_s=5)
    results = await asyncio.gather(first.execute("msn_1"), second.execute("msn_1"))
    assert sorted(results) == [False, True] and orchestrator.runs == 1
    assert await lease.holder("msn_1") is None


async def test_losing_the_lease_stops_the_run():
    lease, orchestrator = InMemoryMissionLease(), SlowOrchestrator()
    executor = MissionExecutor(orchestrator, lease, "worker-1", ttl_s=0.3)
    run = asyncio.create_task(executor.execute("msn_1"))
    await asyncio.sleep(0.05)
    lease._leases["msn_1"] = ("worker-2", 10**12)    # another worker took over (we stalled)
    assert await run is True
    assert orchestrator.cancelled and await lease.holder("msn_1") == "worker-2"


# ── workers ─────────────────────────────────────────────────────────────────
async def _queue_container(settings, scripts=None) -> Container:
    settings.hiring_llm_mandates = False
    settings.llm_retry_backoff_s = 0
    settings.execution_mode = ExecutionMode.QUEUE
    container = Container.build(settings, llm=gateway(RoleScripted(scripts or everyone())))
    container.runtime._approver = AllowAllApprover()   # agents' AMBER calls pass; the real gate stays for apply
    await container.start()
    return container


async def _drain(worker: MissionWorker, container: Container, mission_id: str, timeout_s: float = 60) -> None:
    runner = asyncio.create_task(worker.run())
    try:
        for _ in range(int(timeout_s * 10)):
            mission = await container.board.get(mission_id)
            if mission.is_terminal and not worker._tasks:
                break
            await asyncio.sleep(0.1)
    finally:
        worker.stop()
        await runner


async def test_queue_mode_api_enqueues_and_a_worker_completes_the_mission(settings, sample_repo):
    container = await _queue_container(settings)
    mission = await container.board.create_mission("Fix add() so the test suite passes", str(sample_repo))
    await container.runner.start(mission.mission_id)            # what POST /missions/{id}/run does in queue mode
    assert (await container.board.get(mission.mission_id)).status == MissionStatus.CREATED  # nothing ran in the API

    await _drain(MissionWorker(container, concurrency=2, worker_id="w-test"), container, mission.mission_id)
    final = await container.board.get(mission.mission_id)
    assert final.status == MissionStatus.COMPLETED, final.status_reason
    assert await container.lease.holder(mission.mission_id) is None
    assert await container.job_queue.dequeue("w-other", timeout_s=0.1) is None   # acknowledged
    assert await container.lease.acquire(mission.mission_id, "worker-elsewhere", 5)
    with pytest.raises(ConflictError):                           # held elsewhere: the API refuses a second run
        await container.runner.start(mission.mission_id)
    await container.stop()


async def test_a_crashed_run_resumes_where_it_stopped(settings, sample_repo):
    container = await _queue_container(settings)
    mission = await container.board.create_mission("Fix add() so the test suite passes", str(sample_repo))
    mid = mission.mission_id
    # Simulate a worker that died mid-task: Atlas had opened the mission and started analysis.
    await container.orchestrator._open(mission)
    analysis = (await container.board.get(mid)).task_by_origin("bootstrap:requirements")
    await container.board.assign_task(mid, analysis.id, "Kenji", by="Atlas")
    await container.board.update_task(mid, analysis.id, TaskStatus.IN_PROGRESS, by="Atlas")

    final = await container.orchestrator.run(mid)
    assert final.status == MissionStatus.COMPLETED, final.status_reason
    retry = [d.reason for d in final.decisions if d.decision == DecisionType.RETRY]
    assert any("interrupted run" in r and "Analyse the request" in r for r in retry)
    await container.stop()


async def test_a_mission_that_keeps_killing_workers_is_failed(settings, sample_repo):
    container = await _queue_container(settings)
    settings.worker_max_deliveries = 2
    mission = await container.board.create_mission("Fix add()", str(sample_repo))
    job = await container.job_queue.enqueue(mission.mission_id)
    worker = MissionWorker(container, concurrency=1, worker_id="w-test")
    await worker._handle(job.model_copy(update={"deliveries": 3}))
    final = await container.board.get(mission.mission_id)
    assert final.status == MissionStatus.FAILED and "interrupted 2 time(s)" in final.status_reason
    await container.stop()


# ── applying to the repository ──────────────────────────────────────────────
async def _completed(settings, sample_repo) -> tuple[Container, str]:
    settings.execution_mode = ExecutionMode.INLINE
    container = await _queue_container(settings)
    mission = await container.board.create_mission("Fix add() so the test suite passes", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)
    assert final.status == MissionStatus.COMPLETED, final.status_reason
    return container, mission.mission_id


async def test_apply_waits_for_a_human_then_writes_only_the_change(settings, sample_repo):
    container, mid = await _completed(settings, sample_repo)
    original = (sample_repo / "calc.py").read_bytes()

    patch = await container.apply.patch(mid)
    assert "-    return a - b" in patch and "+    return a + b" in patch
    approval = await container.apply.request(mid, by="operator")
    assert approval.action_class == "AMBER" and approval.category == "apply" and "calc.py" in approval.args_preview
    assert (sample_repo / "calc.py").read_bytes() == original            # nothing written yet
    with pytest.raises(ConflictError):
        await container.apply.request(mid, by="operator")                 # one request at a time

    decided = await container.approvals.resolve(approval.approval_id, True, by="kaushal")
    record = await container.apply.on_decision(decided)
    assert record.status == ApplyStatus.APPLIED and record.files == ["calc.py"] and record.decided_by == "kaushal"
    assert b"return a + b" in (sample_repo / "calc.py").read_bytes()
    assert (container.settings.data_dir / "applied" / mid / "backup" / "calc.py").read_bytes() == original
    assert not (sample_repo / ".git").exists()                             # nothing committed, no repo created
    audit = [a for a in await container.storage.audit.list(mid) if a.tool == "apply_changes"]
    assert audit[-1].approved_by_human and audit[-1].outcome == "ok"
    await container.stop()


async def test_apply_rejected_or_conflicting_writes_nothing(settings, sample_repo):
    container, mid = await _completed(settings, sample_repo)
    approval = await container.apply.request(mid, by="operator")
    rejected = await container.apply.on_decision(
        await container.approvals.resolve(approval.approval_id, False, by="kaushal", note="not today"))
    assert rejected.status == ApplyStatus.REJECTED and b"a - b" in (sample_repo / "calc.py").read_bytes()

    # Someone edits the file after the mission started: applying must not overwrite their work.
    (sample_repo / "calc.py").write_text((sample_repo / "calc.py").read_text() + "\n# local edit\n")
    approval = await container.apply.request(mid, by="operator")
    failed = await container.apply.on_decision(await container.approvals.resolve(approval.approval_id, True, by="k"))
    assert failed.status == ApplyStatus.FAILED and "calc.py" in failed.error
    assert (sample_repo / "calc.py").read_text().endswith("# local edit\n")
    await container.stop()


async def test_apply_guards(settings, sample_repo):
    container = await _queue_container(settings)
    mission = await container.board.create_mission("Fix add()", str(sample_repo))
    with pytest.raises(ConflictError, match="completed"):
        await container.apply.request(mission.mission_id, by="operator")
    await container.stop()


async def test_line_endings_survive_on_windows_style_checkouts(settings, tmp_path):
    """core.autocrlf=true (the Windows default) must not make every file look changed at apply time."""
    repo = tmp_path / "crlf"
    repo.mkdir()
    (repo / "calc.py").write_bytes(b"def add(a, b):\r\n    return a - b\r\n")
    (repo / "pytest.ini").write_bytes(b"[pytest]\r\npythonpath = .\r\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_calc.py").write_bytes(
        b"from calc import add\r\n\r\n\r\ndef test_add():\r\n    assert add(2, 3) == 5\r\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "core.autocrlf=true", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"],
                   cwd=repo, check=True)
    subprocess.run(["git", "-c", "core.autocrlf=true", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm",
                    "init"], cwd=repo, check=True)
    container, mid = await _completed(settings, repo)
    approval = await container.apply.request(mid, by="operator")
    record = await container.apply.on_decision(await container.approvals.resolve(approval.approval_id, True, by="k"))
    assert record.status == ApplyStatus.APPLIED, record.error
    assert (repo / "calc.py").read_bytes() == b"def add(a, b):\r\n    return a + b\r\n"
    await container.stop()


@pytest.mark.skipif(not REDIS_URL, reason="set DEVPILOT_TEST_REDIS_URL to run against a real Redis")
async def test_redis_event_bus_round_trip():
    from app.infrastructure.bus.redis import RedisEventBus

    bus = RedisEventBus(REDIS_URL, channel_prefix=f"devpilot:test:{uuid.uuid4().hex[:8]}:")
    await bus.start()
    try:
        async with bus.subscribe("mission.m1") as live:
            await bus.publish("mission.m1", {"seq": 1, "kind": "log"})
            await bus.publish("mission.other", {"seq": 9})
            assert await asyncio.wait_for(anext(live), timeout=5) == {"seq": 1, "kind": "log"}
        assert await bus.ping()
    finally:
        await bus.stop()
