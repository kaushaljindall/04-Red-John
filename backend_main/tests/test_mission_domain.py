import pytest

from app.domain.mission import (
    AgentMessage,
    CriterionStatus,
    Mission,
    MissionError,
    MissionStatus,
    Task,
    TaskStatus,
)


def make_mission() -> Mission:
    m = Mission(title="t", request="add a multiply endpoint", repo_path="/r")
    m.set_requirements(["Support multiplication"], ["GET /mul returns a*b", "Existing tests still pass"])
    return m


def test_dag_ready_tasks_and_cycles():
    m = make_mission()
    recon = m.add_task(Task(title="recon", kind="analysis"))
    impl = m.add_task(Task(title="implement", depends_on=[recon.id]))
    test = m.add_task(Task(title="test", kind="test", depends_on=[impl.id]))
    review = m.add_task(Task(title="review", kind="review", depends_on=[impl.id]))

    assert [t.id for t in m.ready_tasks()] == [recon.id]
    with pytest.raises(MissionError, match="unknown task"):
        m.add_task(Task(title="x", depends_on=["tsk_nope"]))
    with pytest.raises(MissionError, match="cycle"):
        m.set_dependencies(recon.id, [test.id])
    assert m.task(recon.id).depends_on == []  # rejected change was rolled back

    with pytest.raises(MissionError, match="waiting on"):
        m.update_task(impl.id, TaskStatus.IN_PROGRESS)
    m.update_task(recon.id, TaskStatus.IN_PROGRESS)
    m.update_task(recon.id, TaskStatus.DONE, summary="pytest, flat layout")
    m.update_task(impl.id, TaskStatus.IN_PROGRESS)
    m.update_task(impl.id, TaskStatus.DONE)
    # Test and review can now run in parallel.
    assert {t.id for t in m.ready_tasks()} == {test.id, review.id}


def test_reopening_a_task_reruns_its_dependants():
    m = make_mission()
    impl = m.add_task(Task(title="implement"))
    test = m.add_task(Task(title="test", depends_on=[impl.id]))
    for task in (impl, test):
        m.update_task(task.id, TaskStatus.IN_PROGRESS)
        m.update_task(task.id, TaskStatus.DONE)
    m.update_task(impl.id, TaskStatus.PENDING)  # review found a bug
    assert m.task(test.id).status == TaskStatus.PENDING


def test_attempt_limit():
    m = make_mission()
    fix = m.add_task(Task(title="fix", kind="fix", max_attempts=2))
    for _ in range(2):
        m.update_task(fix.id, TaskStatus.IN_PROGRESS)
        m.update_task(fix.id, TaskStatus.FAILED)
        m.update_task(fix.id, TaskStatus.PENDING)
    with pytest.raises(MissionError, match="attempts"):
        m.update_task(fix.id, TaskStatus.IN_PROGRESS)


def test_completion_needs_every_criterion_met_with_evidence():
    m = make_mission()
    m.transition(MissionStatus.PLANNING)
    m.transition(MissionStatus.RUNNING)
    with pytest.raises(MissionError, match="evidence"):
        m.set_criterion(m.acceptance_criteria[0].id, CriterionStatus.MET, by="Mei", evidence_ids=[])
    with pytest.raises(MissionError, match="cannot complete"):
        m.transition(MissionStatus.COMPLETED)
    for c in m.acceptance_criteria:
        m.set_criterion(c.id, CriterionStatus.MET, by="Mei", evidence_ids=["evd_1"])
    m.transition(MissionStatus.COMPLETED)
    with pytest.raises(MissionError, match="no longer change"):
        m.add_task(Task(title="late"))
    with pytest.raises(MissionError, match="cannot go"):
        m.transition(MissionStatus.RUNNING)


def test_requirements_keep_verified_criteria():
    m = make_mission()
    first = m.acceptance_criteria[0]
    m.set_criterion(first.id, CriterionStatus.MET, by="Mei", evidence_ids=["evd_1"])
    m.set_requirements(["Support multiplication"], ["GET /mul returns a*b", "Handles non-numbers with 400"])
    assert m.acceptance_criteria[0].id == first.id and m.acceptance_criteria[0].status == CriterionStatus.MET
    assert m.acceptance_criteria[1].status == CriterionStatus.PENDING


def test_message_addressing():
    msg = AgentMessage(mission_id="m", sender="Aiko", to="@forge", subject="tests failing")
    assert msg.addressed_to("Takeshi", {"forge"}) and not msg.addressed_to("Haruto", {"shadow"})
    assert AgentMessage(mission_id="m", sender="Haruto", to="Takeshi", subject="x").addressed_to("takeshi")
    assert AgentMessage(mission_id="m", sender="Atlas", subject="x").addressed_to("anyone")
