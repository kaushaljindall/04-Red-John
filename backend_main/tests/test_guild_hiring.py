import pytest

from app.agents.guild import WRITE_TOOLS, build_guild, clearance
from app.domain.events import EventKind
from app.domain.mission import Mission, Task, TaskKind, TaskStatus
from app.domain.skills import CANONICAL_SKILLS, infer_skills, normalize_skill
from app.infrastructure.bus.memory import InMemoryEventBus
from app.infrastructure.llm.gateway import LLMGateway
from app.infrastructure.storage.memory import InMemoryStorage
from app.security.sandbox import WorkspaceManager
from app.services.event_publisher import EventPublisher
from app.services.hiring import HiringEngine, HiringService
from app.services.mission_board import MissionBoard
from tests.fake_llm import ScriptedProvider, gateway, reply

GUILD = build_guild()


# ── roster ──────────────────────────────────────────────────────────────────
def test_guild_shape(registry):
    agents = GUILD.all()
    assert len(agents) == 16
    assert [d.name for d in agents if d.always_on] == ["Atlas", "Maya"]
    assert {d.team for d in agents} == {"executive", "planning", "forge", "sentinel", "shadow", "support"}
    assert len({d.badge for d in agents}) == 16
    for d in agents:
        assert all(registry.get(t) for t in d.tools), f"{d.name} is granted a tool that does not exist"
        assert set(d.skills) <= CANONICAL_SKILLS, f"{d.name} has a non-canonical skill"
        assert d.system_prompt.strip() and d.default_mandate


def test_separation_of_duties_is_in_the_tool_grants():
    for name in ("Mei", "Haruto", "Kaito", "Emi", "Aiko", "Scribe", "Kenji", "Hana", "Ryo"):
        assert not WRITE_TOOLS & {*GUILD.require(name).tools} - {"run_command"}, f"{name} can change code"
    assert "verify_criterion" in GUILD.require("Mei").tools
    assert all("verify_criterion" not in d.tools for d in GUILD.all() if d.name != "Mei")
    assert clearance(GUILD.require("Takeshi")) == "AMBER" and clearance(GUILD.require("Haruto")) == "GREEN"


def test_skill_taxonomy():
    assert normalize_skill("pytest") == "python" and normalize_skill("XSS") == "security"
    assert normalize_skill("unit tests") == "test-authoring" and normalize_skill("nonsense") is None
    assert infer_skills("Fix the SQL injection in the login API") == ["backend", "debugging", "security"]
    assert "frontend" in infer_skills("Add a dark-mode toggle button to the React settings page")


# ── engine ──────────────────────────────────────────────────────────────────
def _mission() -> Mission:
    m = Mission(title="t", request="r", repo_path="/r")
    m.set_requirements(["o"], ["c"])
    return m


@pytest.mark.parametrize("kind, title, expected", [
    (TaskKind.FIX, "Fix add() returning the difference in calc.py", "Takeshi"),
    (TaskKind.IMPLEMENT, "Add a dark-mode toggle to the React settings page", "Yuki"),
    (TaskKind.IMPLEMENT, "Add GET /mul endpoint to the Flask API", "Takeshi"),
    (TaskKind.TEST, "Run the full test suite", "Aiko"),
    (TaskKind.TEST, "Write regression tests for add()", "Riku"),
    (TaskKind.REVIEW, "Review the change", "Haruto"),
    (TaskKind.SECURITY, "Inspect the change for injection and secrets", "Kaito"),
    (TaskKind.VERIFY, "Verify the acceptance criteria", "Mei"),
    (TaskKind.REPORT, "Write the evidence report", "Scribe"),
    (TaskKind.IMPLEMENT, "Refactor duplicated parsing code for readability", "Sora"),
])
def test_engine_picks_the_specialist(kind, title, expected):
    engine = HiringEngine(GUILD)
    m = _mission()
    task = m.add_task(Task(title=title, kind=kind))
    best = engine.rank(engine.request_for_task(task, m))[0]
    assert best.eligible and best.agent == expected, engine.rank(engine.request_for_task(task, m))[:3]


def test_independent_review_excludes_the_author():
    engine = HiringEngine(GUILD)
    m = _mission()
    fix = m.add_task(Task(title="Fix the review-flagged naming", kind=TaskKind.FIX))
    m.assign(fix.id, "Sora")
    review = m.add_task(Task(title="Review the change", kind=TaskKind.REVIEW, depends_on=[fix.id]))
    request = engine.request_for_task(review, m)
    assert request.exclude == ["Sora"]
    ranked = engine.rank(request)
    assert ranked[0].agent == "Haruto"
    assert next(c for c in ranked if c.agent == "Sora").note == "excluded for this work"


def test_always_on_and_core_skills_are_hard_filters():
    engine = HiringEngine(GUILD)
    scores = {c.agent: c for c in engine.rank(engine.request_for_skills(["orchestration"], "x"))}
    assert not scores["Atlas"].eligible and scores["Atlas"].note.startswith("always on duty")
    security = {c.agent: c for c in engine.rank(engine.request_for_skills(["security", "python"], "x"))}
    assert not security["Takeshi"].eligible and "lacks core skill: security" in security["Takeshi"].note


# ── Maya ────────────────────────────────────────────────────────────────────
@pytest.fixture
def setup(settings, sample_repo):
    storage = InMemoryStorage()
    events = EventPublisher(storage.events, InMemoryEventBus())
    board = MissionBoard(storage, events, WorkspaceManager(settings.workspaces_dir))

    def maya(llm: LLMGateway | None = None) -> HiringService:
        return HiringService(GUILD, HiringEngine(GUILD), board, events, llm or LLMGateway({}))

    return storage, board, maya, sample_repo


async def test_initial_staffing_hires_the_council_and_early_specialists(setup):
    storage, board, maya, repo = setup
    mission = await board.create_mission("Fix the SQL injection in the login API", str(repo))
    hires = await maya().initial_staffing(mission.mission_id)
    assert [h.agent for h in hires] == ["Kenji", "Hana", "Ryo", "Kaito"]
    assert hires[0].mandate == GUILD.require("Kenji").default_mandate   # no LLM → fallback mandate

    events = await storage.events.list(mission.mission_id)
    hire_events = [e.data for e in events if e.kind == EventKind.HIRE]
    assert [h["agent"] for h in hire_events] == ["Kenji", "Hana", "Ryo", "Kaito"]
    assert {"team": "Shadow", "clearance": "GREEN", "badge": "DVP-014-KAI"}.items() <= hire_events[3].items()
    assert any("Maya → hire Kaito" in e.data.get("text", "") for e in events if e.kind == EventKind.LOG)

    plain = await board.create_mission("Add a mul() function", str(repo))
    assert [h.agent for h in await maya().initial_staffing(plain.mission_id)] == ["Kenji", "Hana", "Ryo"]


async def test_mandates_come_from_the_llm_when_available(setup):
    storage, board, maya, repo = setup
    provider = ScriptedProvider([reply(content='"Find why add() subtracts and fix it without touching mul()."')])
    mission = await board.create_mission("Fix add()", str(repo))
    [hire] = await maya(gateway(provider)).hire_for_skills(mission.mission_id, ["debugging", "python"], "add() is wrong")
    assert hire.agent == "Takeshi"
    assert hire.mandate == "Find why add() subtracts and fix it without touching mul()."
    assert provider.requests[0].tier == "fast" and "add() is wrong" in provider.requests[0].messages[1]["content"]


async def test_hiring_is_idempotent_and_release_works(setup):
    storage, board, maya, repo = setup
    mission = await board.create_mission("Fix add()", str(repo))
    service = maya()
    first = await service.hire_for_skills(mission.mission_id, ["code-review"], "review")
    again = await service.hire_for_skills(mission.mission_id, ["code-review"], "review again")
    assert first[0].hired_at == again[0].hired_at
    assert [e.kind for e in await storage.events.list(mission.mission_id)].count(EventKind.HIRE) == 1

    await service.release(mission.mission_id, "Haruto", "review done")
    assert "Haruto" not in (await board.get(mission.mission_id)).hired_agents
    rehired = await service.hire_for_skills(mission.mission_id, ["code-review"], "new diff to review")
    assert rehired[0].agent == "Haruto" and rehired[0].released_at is None
    hires = (await board.get(mission.mission_id)).hires
    assert [(h.agent, h.active) for h in hires] == [("Haruto", False), ("Haruto", True)]


async def test_repeated_failure_brings_in_a_different_specialist(setup):
    """The feedback loop the PS asks for: failing evidence changes who does the work next."""
    storage, board, maya, repo = setup
    mission = await board.create_mission("Fix add()", str(repo))
    mid = mission.mission_id
    [fix] = await board.add_tasks(mid, [Task(title="Fix add() in calc.py", kind=TaskKind.FIX)], by="Ryo")
    service = maya()

    first = await service.staff_task(mid, fix.id)
    assert first.name == "Takeshi"
    await board.assign_task(mid, fix.id, first.name, by="Atlas")
    await board.update_task(mid, fix.id, TaskStatus.IN_PROGRESS, by="Takeshi")
    await board.update_task(mid, fix.id, TaskStatus.FAILED, by="Takeshi", summary="tests still fail")
    await board.update_task(mid, fix.id, TaskStatus.PENDING, by="Atlas")
    assert (await service.staff_task(mid, fix.id)).name == "Takeshi"     # one failure: same owner retries

    await board.update_task(mid, fix.id, TaskStatus.IN_PROGRESS, by="Takeshi")
    await board.update_task(mid, fix.id, TaskStatus.FAILED, by="Takeshi", summary="still failing")
    await board.update_task(mid, fix.id, TaskStatus.PENDING, by="Atlas")
    escalated = await service.staff_task(mid, fix.id)
    assert escalated.name == "Daichi"                                    # two failures: the Debugger takes over
    logs = [e.data["text"] for e in await storage.events.list(mid) if e.kind == EventKind.LOG]
    assert any("failed 2 time(s) under Takeshi" in t for t in logs)


async def test_nobody_qualified_is_reported_not_faked(setup):
    storage, board, maya, repo = setup
    mission = await board.create_mission("x y z", str(repo))
    service = maya()
    request = service.engine.request_for_skills(["security"], "audit", exclude=["Kaito"])
    assert await service.hire(mission.mission_id, request) == []
    logs = [e.data["text"] for e in await storage.events.list(mission.mission_id) if e.kind == EventKind.LOG]
    assert any("could not staff 'audit'" in t for t in logs)
