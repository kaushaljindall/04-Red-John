import asyncio

import pytest

from app.core.errors import NotFoundError, ValidationFailed
from app.domain.events import EventKind
from app.domain.mission import (
    AgentMessage,
    CriterionStatus,
    Decision,
    Finding,
    MessageKind,
    MissionStatus,
    OpenIssue,
    Task,
    TaskStatus,
)
from app.infrastructure.bus.memory import InMemoryEventBus
from app.infrastructure.storage.memory import InMemoryStorage
from app.security.sandbox import WorkspaceManager
from app.services.event_publisher import EventPublisher
from app.services.governance import Governor
from app.services.mission_board import MissionBoard
from app.tools.registry import AllowAllApprover, ToolContext


@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def board(storage, settings):
    events = EventPublisher(storage.events, InMemoryEventBus())
    return MissionBoard(storage, events, WorkspaceManager(settings.workspaces_dir))


@pytest.fixture
async def mission(board, sample_repo):
    m = await board.create_mission("Fix add() so the tests pass", str(sample_repo))
    await board.set_requirements(m.mission_id, ["add() returns the sum"], ["tests/test_calc.py passes"], by="Kenji")
    return await board.get(m.mission_id)


async def test_create_mission_makes_workspace(board, storage, mission):
    assert mission.workspace_path and len(mission.baseline_commit) == 40
    assert mission.version == 2  # created, then requirements saved
    kinds = [e.kind for e in await storage.events.list(mission.mission_id)]
    assert kinds == [EventKind.MISSION_CREATED, EventKind.PLAN_UPDATED]
    with pytest.raises(NotFoundError):
        await board.get("msn_missing")
    with pytest.raises(ValidationFailed):
        await board.create_mission("   ", "x")


async def test_parallel_agents_do_not_lose_updates(board, mission):
    # Eight agents report findings at the same moment; every one must land.
    async def report(i: int) -> None:
        await board.add_finding(mission.mission_id, Finding(agent=f"agent{i}", kind="note", title=f"note {i}"))

    await asyncio.gather(*(report(i) for i in range(8)))
    stored = await board.get(mission.mission_id)
    assert len(stored.findings) == 8
    assert stored.version == mission.version + 8


async def test_domain_errors_become_validation_errors(board, mission):
    with pytest.raises(ValidationFailed, match="unknown task"):
        await board.add_tasks(mission.mission_id, [Task(title="x", depends_on=["tsk_nope"])], by="Ryo")


async def test_feedback_loop_with_evidence_gated_verification(board, storage, mission, settings, registry):
    mid = mission.mission_id
    [fix] = await board.add_tasks(mid, [Task(title="Fix add()", kind="fix")], by="Ryo")
    await board.assign_task(mid, fix.id, "Takeshi", by="Atlas")
    await board.update_task(mid, fix.id, TaskStatus.IN_PROGRESS, by="Takeshi")

    workspace = WorkspaceManager(settings.workspaces_dir).get(mid)
    governor = Governor(mid, registry, storage.audit, board._events, board=board)
    ctx = ToolContext(mission_id=mid, agent="Aiko", workspace=workspace, settings=settings,
                      approver=AllowAllApprover(), task_id=fix.id)

    failing = await governor.execute("run_tests", {}, ctx)
    failing_id = failing.meta["evidence_id"]
    assert f"evidence_id" in failing.to_model_text()
    [stored] = await board.evidence(mid, kind="test_run")
    assert stored.passed is False and stored.task_id == fix.id and "1 failed" in stored.summary

    # Aiko hands the failure to the developer; the developer reads the inbox.
    await board.post_message(AgentMessage(mission_id=mid, sender="Aiko", to="Takeshi", kind=MessageKind.FINDING,
                                          subject="test_add fails", evidence_ids=[failing_id]))
    [msg] = await board.inbox(mid, "Takeshi")
    assert msg.evidence_ids == [failing_id] and not await board.inbox(mid, "Aiko")

    criterion = (await board.get(mid)).acceptance_criteria[0]
    with pytest.raises(ValidationFailed, match="failing evidence"):
        await board.verify_criterion(mid, criterion.id, CriterionStatus.MET, by="Mei", evidence_ids=[failing_id])
    with pytest.raises(ValidationFailed, match="Unknown evidence"):
        await board.verify_criterion(mid, criterion.id, CriterionStatus.MET, by="Mei", evidence_ids=["evd_fake"])

    ctx.agent = "Takeshi"
    assert (await governor.execute("edit_file", {"path": "calc.py", "old": "return a - b", "new": "return a + b"}, ctx)).ok
    ctx.agent = "Mei"
    passing = await governor.execute("run_tests", {}, ctx)
    await board.verify_criterion(mid, criterion.id, CriterionStatus.MET, by="Mei",
                                 evidence_ids=[passing.meta["evidence_id"]])

    await board.update_task(mid, fix.id, TaskStatus.DONE, by="Takeshi", summary="add() fixed")
    await board.set_status(mid, MissionStatus.PLANNING, by="Atlas")
    await board.set_status(mid, MissionStatus.RUNNING, by="Atlas")
    done = await board.set_status(mid, MissionStatus.COMPLETED, by="Atlas", reason="all criteria met")
    assert done.status == MissionStatus.COMPLETED

    kinds = [e.kind for e in await storage.events.list(mid)]
    assert kinds.count(EventKind.EVIDENCE) == 2
    assert EventKind.CRITERION in kinds and EventKind.MESSAGE in kinds and kinds[-1] == EventKind.COMPLETE


async def test_issues_and_decisions(board, mission):
    mid = mission.mission_id
    issue = await board.raise_issue(mid, OpenIssue(raised_by="Kenji", kind="clarification", needs_human=True,
                                                   text="Should mul() accept floats?"))
    assert len((await board.get(mid)).open_issues) == 1
    await board.resolve_issue(mid, issue.id, by="operator", resolution="Yes, floats too")
    with pytest.raises(ValidationFailed, match="already resolved"):
        await board.resolve_issue(mid, issue.id, by="operator", resolution="again")
    await board.record_decision(mid, Decision(by="Atlas", decision="replan", reason="new requirement: floats"))
    assert (await board.get(mid)).decisions[-1].decision == "replan"
