"""
Whole missions, run by Atlas, with each agent's LLM replies scripted per agent.
Tools, tests, diffs and scans are real; only the model's choices are scripted.
"""
from __future__ import annotations

import json
import re

import pytest

from app.container import Container
from app.core.errors import ExternalServiceError
from app.domain.events import EventKind
from app.domain.mission import (
    CriterionStatus,
    DecisionType,
    FindingStatus,
    MissionStatus,
    Task,
    TaskKind,
    TaskStatus,
)
from app.infrastructure.llm.types import ChatRequest, ChatResponse
from app.services.planning import enforce_gates, parse_plan
from app.tools.registry import AllowAllApprover
from tests.fake_llm import call, finish, gateway, reply

FIX_PLAN = [{"title": "Fix add()", "kind": "fix", "description": "add() subtracts instead of adding",
             "required_skills": ["python"]}]


class RoleScripted:
    """One script per agent (by the `caller` on each request); records every request per agent."""

    name = "scripted"

    def __init__(self, scripts: dict[str, list]) -> None:
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.requests: dict[str, list[ChatRequest]] = {}

    @property
    def available(self) -> bool:
        return True

    async def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        self.requests.setdefault(request.caller, []).append(request.model_copy(deep=True))
        steps = self.scripts.get(request.caller)
        if not steps:
            raise ExternalServiceError(f"no script left for {request.caller}")
        step = steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(request) if callable(step) else step

    async def aclose(self) -> None:
        return None


# ── agent behaviours ────────────────────────────────────────────────────────
def kenji(*criteria: str) -> list:
    return [finish(summary="Criteria set", objectives=["add() returns the sum"],
                   acceptance_criteria=list(criteria or ("The full test suite passes", "mul() behaviour is unchanged")))]


def hana() -> list:
    return [reply(call("run_tests")),
            finish(summary="pytest; flat layout", test_command="python -m pytest", key_files=["calc.py"],
                   conventions=["plain functions, no classes"], risks=["test_add fails at baseline"])]


def ryo(plan=FIX_PLAN) -> list:
    return [finish(summary="Planned", tasks=plan)]


def dev_fix() -> list:
    return [reply(call("edit_file", path="calc.py", old="return a - b", new="return a + b")),
            reply(call("run_tests")), finish(summary="Fixed add()")]


def dev_noop(summary="Looked at it; should be fine") -> list:
    return [finish(summary=summary)]


def aiko() -> list:
    return [reply(call("run_tests")), finish(summary="Ran the suite", passed=True)]  # the claim is ignored; evidence decides


def approve(tool: str) -> list:
    return [reply(call(tool)), finish(summary="Looks good", approved=True)]


def _json_tool_result(request: ChatRequest) -> dict:
    return json.loads(request.messages[-1]["content"])


def mei() -> list:
    def verify_all(request: ChatRequest) -> ChatResponse:
        criteria = re.findall(r"- (ac_[0-9a-f]+) \[", request.messages[1]["content"])
        evidence = _json_tool_result(request)["content"]["evidence"]
        passing = [e["evidence_id"] for e in evidence if e["passed"]][-1:]
        return reply(*[call("verify_criterion", criterion_id=c, status="met", evidence_ids=passing,
                            note="covered by the passing suite") for c in criteria])

    return [reply(call("list_evidence", kind="test_run")), verify_all, finish(summary="Verified", verdicts=[])]


def everyone(**overrides: list) -> dict[str, list]:
    scripts = {"Kenji": kenji(), "Hana": hana(), "Ryo": ryo(), "Takeshi": dev_fix(), "Aiko": aiko(),
               "Haruto": approve("git_diff"), "Kaito": approve("secret_scan"), "Mei": mei()}
    scripts.update(overrides)
    return scripts


@pytest.fixture
def make(settings):
    settings.hiring_llm_mandates = False
    settings.llm_retry_backoff_s = 0

    async def build(scripts: dict[str, list], **overrides) -> tuple[Container, RoleScripted]:
        for key, value in overrides.items():
            setattr(settings, key, value)
        provider = RoleScripted(scripts)
        container = Container.build(settings, llm=gateway(provider))
        container.approvals = AllowAllApprover()  # no human in these tests
        container.runtime._approver = container.approvals
        await container.start()
        return container, provider

    return build


async def _events(container: Container, mid: str, kind: EventKind) -> list[dict]:
    return [e.data for e in await container.storage.events.list(mid) if e.kind == kind]


async def _decisions(container: Container, mid: str) -> list[tuple[str, str]]:
    return [(d.decision, d.reason) for d in (await container.board.get(mid)).decisions]


# ── scenarios ───────────────────────────────────────────────────────────────
async def test_end_to_end_mission_completes_with_evidence(make, sample_repo):
    container, provider = await make(everyone())
    mission = await container.board.create_mission("Fix add() so the test suite passes", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)

    assert final.status == MissionStatus.COMPLETED, final.status_reason
    assert all(c.status == CriterionStatus.MET and c.evidence_ids for c in final.acceptance_criteria)
    kinds = {t.kind for t in final.tasks}
    assert {TaskKind.FIX, TaskKind.TEST, TaskKind.REVIEW, TaskKind.SECURITY, TaskKind.VERIFY} <= kinds
    gates = [t.title for t in final.tasks if (t.origin or "").startswith("gate:")]
    assert gates == ["Run the test suite", "Review the change", "Security inspection", "Verify the acceptance criteria"]
    assert {t.title: t.assignee for t in final.tasks} | {} == {
        "Analyse the request": "Kenji", "Reconnoitre the codebase": "Hana", "Plan the work": "Ryo",
        "Fix add()": "Takeshi", "Run the test suite": "Aiko", "Review the change": "Haruto",
        "Security inspection": "Kaito", "Verify the acceptance criteria": "Mei"}

    hires = await _events(container, mission.mission_id, EventKind.HIRE)
    assert [h["agent"] for h in hires] == ["Kenji", "Hana", "Ryo", "Takeshi", "Aiko", "Haruto", "Kaito", "Mei"]
    assert final.hired_agents == []  # everyone released at the end
    logs = [e["text"] for e in await _events(container, mission.mission_id, EventKind.LOG)]
    assert any(t.startswith("Atlas dispatching in parallel: Kenji") for t in logs)
    assert any("Aiko → Run the test suite" in t and "Haruto → Review the change" in t for t in logs)

    report = final.report
    assert "Acceptance criteria verified:** 2 of 2" in report and "✅ met" in report
    assert "1 file changed" in report and "Nothing unresolved" in report
    assert (container.workspaces.get(mission.mission_id).root / "calc.py").read_text().count("a + b") == 1
    assert "a - b" in (sample_repo / "calc.py").read_text()  # the real repo is untouched
    await container.stop()


async def test_failing_tests_loop_back_and_escalate_to_the_debugger(make, sample_repo):
    """The PS feedback loop: new evidence (failing tests) changes who does what next."""
    scripts = everyone(
        Takeshi=dev_noop() * 3,                                  # first fix and two repair rounds change nothing
        Daichi=dev_fix(),
        Aiko=aiko() * 4, Haruto=approve("git_diff") * 4, Kaito=approve("secret_scan") * 4,
    )
    container, provider = await make(scripts)
    mission = await container.board.create_mission("Fix add() so the test suite passes", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)

    assert final.status == MissionStatus.COMPLETED, final.status_reason
    repair = final.task_by_origin(f"repair:{next(t.id for t in final.tasks if t.title == 'Run the test suite')}")
    assert repair.assignee == "Daichi" and repair.attempts == 3
    decisions = await _decisions(container, mission.mission_id)
    replans = [r for d, r in decisions if d == DecisionType.REPLAN]
    assert len(replans) == 3 and "Tests fail → added" in replans[0] and "(round 3)" in replans[2]
    assert "Evidence: evd_" in replans[0]
    logs = [e["text"] for e in await _events(container, mission.mission_id, EventKind.LOG)]
    assert any("failed 2 time(s) under Takeshi" in t for t in logs)
    # The repair briefing carried the real failure output.
    takeshi_repair_brief = provider.requests["Takeshi"][1].messages[1]["content"]
    assert "Tests fail" in takeshi_repair_brief and "test_add" in takeshi_repair_brief
    runs = await container.board.evidence(mission.mission_id, kind="test_run")
    assert [e.passed for e in runs if e.agent == "Aiko"] == [False, False, False, True]
    await container.stop()


async def test_ambiguity_pauses_for_a_human_then_resumes(make, sample_repo):
    ask = [reply(call("raise_issue", kind="clarification", needs_human=True,
                      text="Should add() also accept strings?")),
           finish(status="blocked", summary="Waiting for the human")]
    container, provider = await make(everyone(Kenji=ask + kenji()))
    mission = await container.board.create_mission("Make add() work properly", str(sample_repo))
    mid = mission.mission_id

    paused = await container.orchestrator.run(mid)
    assert paused.status == MissionStatus.AWAITING_HUMAN
    assert paused.task_by_origin("bootstrap:requirements").status == TaskStatus.BLOCKED
    assert paused.task_by_origin("bootstrap:recon").status == TaskStatus.DONE  # parallel work was not lost
    [issue] = paused.open_issues
    await container.board.resolve_issue(mid, issue.id, by="operator", resolution="No, numbers only")

    final = await container.orchestrator.run(mid)
    assert final.status == MissionStatus.COMPLETED, final.status_reason
    second_brief = provider.requests["Kenji"][-1].messages[1]["content"]
    assert "Should add() also accept strings? → No, numbers only" in second_brief
    assert [d for d, _ in await _decisions(container, mid)].count(DecisionType.CLARIFY) == 1
    await container.stop()


async def test_review_critique_dispute_and_arbitration(make, sample_repo):
    def critique() -> list:
        return [reply(call("record_finding", kind="review", severity="medium", title="add() lacks a docstring",
                           file="calc.py", line=4)),
                finish(summary="One issue", approved=False)]

    def dispute(request: ChatRequest) -> ChatResponse:
        finding = re.search(r"(fnd_[0-9a-f]+)", request.messages[1]["content"]).group(1)
        return reply(call("update_finding", finding_id=finding, status="disputed",
                          resolution="No function in calc.py has a docstring; the project does not use them."))

    container, provider = await make(everyone(
        Takeshi=dev_fix() + [dispute, finish(summary="Disputed the finding with evidence")],
        Haruto=critique() + [finish(summary="Developer disputes", approved=False)] + approve("git_diff"),
        Emi=[reply(call("read_file", path="calc.py")),
             finish(summary="Ruled", ruling="developer", reason="No docstrings anywhere in the module")],
        Aiko=aiko() * 2, Kaito=approve("secret_scan") * 2,
    ))
    mission = await container.board.create_mission("Fix add() so the test suite passes", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)

    assert final.status == MissionStatus.COMPLETED, final.status_reason
    [finding] = [f for f in final.findings if f.kind == "review"]
    assert finding.status == FindingStatus.DISMISSED and "Arbiter ruled for the developer" in finding.resolution
    arbitration = final.task_by_origin(f"arbitrate:{finding.id}")
    assert arbitration.assignee == "Emi"
    decisions = [r for _, r in await _decisions(container, mission.mission_id)]
    assert any("Review findings from Haruto" in r for r in decisions)
    assert any("the Arbiter decides" in r for r in decisions)
    assert any("Arbiter dismissed" in r for r in decisions)
    await container.stop()


async def test_bounded_autonomy_escalates_instead_of_looping(make, sample_repo):
    container, _ = await make(everyone(Takeshi=dev_noop() * 3, Aiko=aiko() * 3,
                                       Haruto=approve("git_diff") * 3, Kaito=approve("secret_scan") * 3),
                              repair_max_attempts=2, escalate_after_failures=5)
    mission = await container.board.create_mission("Fix add() so the test suite passes", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)
    assert final.status == MissionStatus.AWAITING_HUMAN
    [issue] = final.open_issues
    assert issue.raised_by == "Atlas" and "after 2 repair round(s)" in issue.text
    assert [d.decision for d in final.decisions].count(DecisionType.ESCALATE) == 1
    # Work that finished in the same round is still recorded, but nothing new starts.
    assert not [t for t in final.tasks if t.status == TaskStatus.IN_PROGRESS]
    await container.stop()


async def test_model_outage_is_retried_then_escalated(make, sample_repo):
    down = ExternalServiceError("provider down")
    container, _ = await make(everyone(Kenji=[down, down, down]))
    mission = await container.board.create_mission("Fix add()", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)
    assert final.status == MissionStatus.AWAITING_HUMAN
    decisions = await _decisions(container, mission.mission_id)
    retries = [r for d, r in decisions if d == DecisionType.RETRY]
    assert len(retries) == 2 and all("llm_unavailable" in r for r in retries)
    assert not [d for d, _ in decisions if d == DecisionType.REASSIGN]   # an outage is nobody's fault
    assert "failed 3 time(s): the language model is unavailable" in final.open_issues[0].text
    assert final.task_by_origin("bootstrap:requirements").assignee == "Kenji"
    await container.stop()


async def test_missing_test_runner_hires_a_test_author(make, tmp_path):
    repo = tmp_path / "norunner"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    riku = [reply(call("write_file", path="pytest.ini", content="[pytest]\npythonpath = .\n")),
            reply(call("write_file", path="tests/test_calc.py",
                       content="from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n")),
            reply(call("run_tests")), finish(summary="Added tests")]
    container, _ = await make(everyone(Hana=[finish(summary="no tests", test_command=None, key_files=["calc.py"],
                                                    conventions=[], risks=["no tests"])],
                                       Riku=riku, Aiko=aiko() * 2, Haruto=approve("git_diff") * 2,
                                       Kaito=approve("secret_scan") * 2))
    mission = await container.board.create_mission("Fix add()", str(repo))
    final = await container.orchestrator.run(mission.mission_id)
    assert final.status == MissionStatus.COMPLETED, final.status_reason
    author = next(t for t in final.tasks if (t.origin or "").startswith("author-tests:"))
    assert author.assignee == "Riku" and author.title == "Write tests: The project has no runnable tests"
    await container.stop()


async def test_budget_exhaustion_fails_the_mission_with_a_report(make, sample_repo):
    container, _ = await make(everyone(), mission_max_tool_calls=3)
    mission = await container.board.create_mission("Fix add()", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)
    assert final.status == MissionStatus.FAILED and "budget" in final.status_reason
    assert final.report and "Criterion not verified" in final.report
    await container.stop()


# ── planning rules ──────────────────────────────────────────────────────────
def test_plan_parsing_and_gates():
    from app.domain.mission import Mission

    m = Mission(title="t", request="Add mul endpoint", repo_path="/r")
    m.set_requirements(["o"], ["c1", "c2"])
    ac = m.acceptance_criteria[0].id
    tasks = parse_plan({"tasks": [
        {"title": "Add /mul", "kind": "implementation", "criteria": [ac, "ac_bogus"], "skills": ["python"]},
        {"title": "Write tests", "kind": "test", "depends_on": ["Add /mul"]},
        {"title": "Review", "kind": "code_review", "depends_on": ["T1"]},
        {"title": "", "kind": "fix"}, {"title": "Dance", "kind": "party"},
    ]}, m)
    assert [(t.title, t.kind) for t in tasks] == [("Add /mul", "implement"), ("Write tests", "test"), ("Review", "review")]
    assert tasks[0].criteria_ids == [ac] and tasks[1].depends_on == [tasks[0].id] == tasks[2].depends_on
    gated = enforce_gates(tasks, m)
    added = [t for t in gated if t.origin]
    assert [t.kind for t in added] == [TaskKind.SECURITY, TaskKind.VERIFY]
    verify = added[-1]
    assert set(verify.depends_on) == {tasks[1].id, tasks[2].id, added[0].id}
    assert enforce_gates([], m)[0].kind == TaskKind.IMPLEMENT  # no work in the plan → fallback work task
    assert parse_plan({"tasks": "nonsense"}, m) == []


def test_report_separates_verified_from_claims(make):
    from app.domain.mission import Evidence, Mission
    from app.services.report import build_report

    m = Mission(title="t", request="r", repo_path="/r")
    m.set_requirements(["o"], ["tests pass", "docs updated"])
    m.set_criterion(m.acceptance_criteria[0].id, CriterionStatus.MET, by="Mei", evidence_ids=["evd_1"])
    evidence = [Evidence(evidence_id="evd_1", mission_id=m.mission_id, kind="test_run", agent="Aiko", tool="run_tests",
                         passed=True, summary="2 passed")]
    report = build_report(m, evidence, [])
    assert "| tests pass | ✅ met | `evd_1` 2 passed | Mei |" in report
    assert "Criterion not verified: docs updated (pending)" in report
    assert "Agent summaries are claims" in report


def test_run_and_auto_resume_over_http(settings, sample_repo):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings.hiring_llm_mandates = False
    ask = [reply(call("raise_issue", kind="clarification", needs_human=True, text="Floats too?")),
           finish(status="blocked", summary="Need an answer")]
    container = Container.build(settings, llm=gateway(RoleScripted(everyone(Kenji=ask + kenji()))))
    container.approvals = AllowAllApprover()
    container.runtime._approver = container.approvals
    app = create_app(settings, container=container)
    with TestClient(app) as client:
        created = client.post("/api/v1/missions", json={"request": "Fix add()", "repo_path": str(sample_repo),
                                                        "start": True}).json()
        mid = created["mission"]["mission_id"]
        assert created["running"] is True
        client.portal.call(container.runner.wait, mid)
        # Re-running while the question is unanswered is accepted but does not move the mission.
        assert client.post(f"/api/v1/missions/{mid}/run").status_code == 202
        client.portal.call(container.runner.wait, mid)

        mission = client.get(f"/api/v1/missions/{mid}").json()["mission"]
        assert mission["status"] == "awaiting_human"
        issue = next(i for i in mission["issues"] if i["status"] == "open")
        answered = client.post(f"/api/v1/missions/{mid}/issues/{issue['id']}/resolve", json={"resolution": "No"})
        assert answered.json()["resumed"] is True
        client.portal.call(container.runner.wait, mid)

        snap = client.get(f"/api/v1/realtime/snapshot/{mid}").json()
        assert snap["mission"]["status"] == "completed" and snap["complete"] is True
        assert client.post(f"/api/v1/missions/{mid}/run").status_code == 409   # nothing left to run
        assert "✅ met" in client.get(f"/api/v1/missions/{mid}").json()["mission"]["report"]


async def test_plan_with_its_own_verify_task_is_ordered_not_rejected(make, sample_repo):
    """Seen live: Ryo planned a verify task; Atlas then made it depend on a gate added after it."""
    plan = FIX_PLAN + [{"title": "Verify", "kind": "verify", "depends_on": ["Fix add()"]}]
    container, _ = await make(everyone(Ryo=ryo(plan)))
    mission = await container.board.create_mission("Fix add() so the test suite passes", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)
    assert final.status == MissionStatus.COMPLETED, final.status_reason
    verify = next(t for t in final.tasks if t.kind == TaskKind.VERIFY)
    assert verify.created_by == "Ryo" and len(verify.depends_on) == 4   # its own dependency plus the three checks
    await container.stop()


async def test_bad_agent_output_escalates_instead_of_crashing(make, sample_repo, monkeypatch):
    from app.core.errors import ValidationFailed
    from app.services.mission_board import MissionBoard

    async def broken(*args, **kwargs):
        raise ValidationFailed("task depends on unknown task(s): tsk_x")

    container, _ = await make(everyone())
    monkeypatch.setattr(MissionBoard, "set_requirements", broken)
    mission = await container.board.create_mission("Fix add()", str(sample_repo))
    final = await container.orchestrator.run(mission.mission_id)
    assert final.status == MissionStatus.AWAITING_HUMAN
    assert "could not be applied" in final.open_issues[0].text
    await container.stop()


def test_report_unresolved_lists_only_what_blocks():
    from app.domain.mission import Finding, Mission
    from app.services.report import build_report

    m = Mission(title="t", request="r", repo_path="/r")
    m.set_requirements(["o"], ["tests pass"])
    m.set_criterion(m.acceptance_criteria[0].id, CriterionStatus.MET, by="Mei", evidence_ids=["evd_1"])
    m.add_finding(Finding(agent="Kenji", kind="bug", severity="high", title="Bug in add"))
    m.add_finding(Finding(agent="Haruto", kind="review", severity="info", title="Only add changed"))
    report = build_report(m, [], [])
    assert "Nothing unresolved." in report
    assert "not blocking" in report and "Bug in add (high, Kenji)" in report and "Only add changed (info" not in report
