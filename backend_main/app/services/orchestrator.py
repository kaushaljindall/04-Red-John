"""
Atlas — the mission orchestrator. Replaces Mycel's fixed 4-phase pipeline
with a decision loop:

    observe the board → decide → dispatch ready tasks (in parallel) → read results → decide …

Atlas follows explicit rules rather than an LLM. Every choice it makes —
continue, retry, reassign, re-plan, clarify, escalate, stop, complete — is
written to the board as a Decision with its reason, so the run log explains
itself. The thinking is done by the agents Atlas dispatches; Atlas makes sure
their results are acted on:

- Failing tests → a repair task for the Forge → tests re-run.            (C → B)
- Review or security findings → repair → the reviewer re-checks.          (D → B)
- A developer disputes a finding → the Arbiter rules, or a human does.    (D)
- Unmet / unverifiable criteria → repair or new tests → re-verification.  (C → B)
- A task that keeps failing → Maya brings in a different specialist, then a human.
- A missing test runner → a test author is hired.                         (unexpected conditions)
- Ambiguity → the mission pauses for the human, then resumes.

Bounded: round limit, per-task attempt limits, repair limits, mission budgets.
The mission completes only when every acceptance criterion is verified with
evidence and no blocking finding is open.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

from app.agents.runtime import AgentRuntime
from app.core.config import Settings
from app.core.errors import ValidationFailed
from app.core.logging import mission_id_var
from app.domain.agents import AgentDefinition, AgentRunResult, RunStatus
from app.domain.events import LogLevel
from app.domain.mission import (
    CriterionStatus,
    Decision,
    DecisionType,
    Finding,
    FindingKind,
    FindingStatus,
    IssueKind,
    Mission,
    MissionStatus,
    OpenIssue,
    Severity,
    Task,
    TaskKind,
    TaskStatus,
)
from app.security.sandbox import WorkspaceManager
from app.services.event_publisher import EventPublisher
from app.services.governance import Governor
from app.services.hiring import HiringService
from app.services.mission_board import MissionBoard
from app.services.planning import (
    WORK_KINDS,
    enforce_gates,
    fallback_plan,
    fallback_requirements,
    parse_plan,
    parse_requirements,
)
from app.services.report import build_report
from app.tools.registry import Approver, ToolContext

log = logging.getLogger(__name__)

ATLAS = "Atlas"
CHECK_KINDS = {TaskKind.TEST, TaskKind.REVIEW, TaskKind.SECURITY, TaskKind.VERIFY}
BLOCKING_SEVERITIES = {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}


def bootstrap_tasks() -> list[Task]:
    """Every mission opens the same way: analysis and recon in parallel, then planning."""
    analysis = Task(title="Analyse the request", kind=TaskKind.ANALYSIS, required_skills=["requirements"],
                    origin="bootstrap:requirements", created_by=ATLAS,
                    description="Turn the request into objectives and testable acceptance criteria.")
    recon = Task(title="Reconnoitre the codebase", kind=TaskKind.ANALYSIS, required_skills=["recon"],
                 origin="bootstrap:recon", created_by=ATLAS,
                 description="Map the files, conventions and test setup relevant to the request; record the "
                             "test baseline.")
    plan = Task(title="Plan the work", kind=TaskKind.PLAN, origin="bootstrap:plan", created_by=ATLAS,
                depends_on=[analysis.id, recon.id],
                description="Break the mission into tasks using the acceptance criteria and the recon findings.")
    return [analysis, recon, plan]


class Orchestrator:
    def __init__(
        self,
        *,
        board: MissionBoard,
        hiring: HiringService,
        runtime: AgentRuntime,
        events: EventPublisher,
        workspaces: WorkspaceManager,
        settings: Settings,
        approver: Approver,
        governor_factory: Callable[[str], Governor],
        audit_list: Callable,
    ) -> None:
        self._board = board
        self._hiring = hiring
        self._runtime = runtime
        self._events = events
        self._workspaces = workspaces
        self._settings = settings
        self._approver = approver
        self._governor_factory = governor_factory
        self._audit_list = audit_list

    # ── entry point ──────────────────────────────────────────────────────
    async def run(self, mission_id: str) -> Mission:
        """Run the mission until it completes, fails, is stopped, or needs a human. Safe to call again to resume."""
        token = mission_id_var.set(mission_id)
        try:
            return await self._run(mission_id)
        finally:
            mission_id_var.reset(token)

    async def _run(self, mid: str) -> Mission:
        mission = await self._board.get(mid)
        if mission.is_terminal:
            return mission
        governor = self._governor_factory(mid)
        await self._recover_interrupted(mission)
        if mission.status == MissionStatus.CREATED:
            await self._open(mission)
        elif mission.status == MissionStatus.AWAITING_HUMAN and not await self._resume(mid):
            return await self._board.get(mid)

        for _ in range(self._settings.orchestrator_max_rounds):
            mission = await self._board.get(mid)
            if mission.is_terminal or mission.status == MissionStatus.AWAITING_HUMAN:
                return mission
            exhausted = governor.exhausted
            if exhausted:
                await self._finish(mid, MissionStatus.FAILED, f"Mission budget exhausted: {exhausted}", governor)
                return await self._board.get(mid)
            if await self._try_complete(mission, governor):
                return await self._board.get(mid)
            batch = await self._staff_batch(mission)
            if batch is None:
                continue  # escalated; the next round sees AWAITING_HUMAN
            if not batch:
                await self._unstick(mission)
                continue
            results = await asyncio.gather(*(self._execute(mid, tid, d, governor) for tid, d in batch))
            for task_id, result in results:
                await self._handle(mid, task_id, result, governor)

        await self._finish(mid, MissionStatus.FAILED,
                           f"Stopped after {self._settings.orchestrator_max_rounds} orchestration rounds", governor)
        return await self._board.get(mid)

    # ── opening and resuming ─────────────────────────────────────────────
    async def _open(self, mission: Mission) -> None:
        mid = mission.mission_id
        await self._board.set_status(mid, MissionStatus.PLANNING, by=ATLAS, reason="mission opened")
        await self._decide(mid, DecisionType.CONTINUE, "Mission opened: Maya staffs the Shogun's Council; analysis "
                                                       "and reconnaissance run in parallel, then planning.")
        await self._hiring.initial_staffing(mid)
        await self._board.add_tasks(mid, bootstrap_tasks(), by=ATLAS)

    async def _recover_interrupted(self, mission: Mission) -> None:
        """Tasks left in progress belong to a run that died (worker crash, restart). Only one runner holds a
        mission at a time, so they can safely go back to pending."""
        stuck = [t for t in mission.tasks if t.status == TaskStatus.IN_PROGRESS]
        for task in stuck:
            await self._board.update_task(mission.mission_id, task.id, TaskStatus.PENDING, by=ATLAS,
                                          summary="interrupted: the previous run stopped before this task finished")
        if stuck:
            await self._decide(mission.mission_id, DecisionType.RETRY, "Resuming after an interrupted run: "
                               + ", ".join(f"'{t.title}'" for t in stuck) + " will run again.")

    async def _resume(self, mid: str) -> bool:
        mission = await self._board.get(mid)
        waiting = [i for i in mission.open_issues if i.needs_human]
        if waiting:
            return False
        answered = {i.task_id for i in mission.issues if i.needs_human and i.task_id}
        for task in mission.tasks:
            if task.id not in answered:
                continue
            if task.status == TaskStatus.BLOCKED:
                await self._board.update_task(mid, task.id, TaskStatus.PENDING, by=ATLAS)
            elif task.status == TaskStatus.FAILED:
                if task.attempts >= task.max_attempts:
                    await self._board.set_task_details(mid, task.id, by=ATLAS, max_attempts=task.attempts + 2)
                await self._board.update_task(mid, task.id, TaskStatus.PENDING, by=ATLAS)
        planned = mission.task_by_origin("bootstrap:plan")
        status = MissionStatus.RUNNING if planned and planned.status == TaskStatus.DONE else MissionStatus.PLANNING
        await self._board.set_status(mid, status, by=ATLAS, reason="human answered")
        answers = "; ".join(f"{i.text} → {i.resolution}" for i in mission.issues if i.needs_human and i.resolution)
        await self._decide(mid, DecisionType.CONTINUE, f"Resuming with the human's answers: {answers[:400]}")
        return True

    # ── dispatch ─────────────────────────────────────────────────────────
    async def _staff_batch(self, mission: Mission) -> list[tuple[str, AgentDefinition]] | None:
        batch: list[tuple[str, AgentDefinition]] = []
        busy: set[str] = set()
        for task in mission.ready_tasks():
            if len(batch) >= self._settings.orchestrator_max_parallel:
                break
            definition = await self._hiring.staff_task(mission.mission_id, task.id, busy=busy)
            if definition is None:
                await self._escalate(mission.mission_id, task.id, f"No agent in the guild can take '{task.title}'.")
                return None
            if definition.name in busy:
                continue  # one agent, one task at a time; it waits for the next round
            busy.add(definition.name)
            batch.append((task.id, definition))
        if len(batch) > 1:
            await self._events.log(mission.mission_id, "Atlas dispatching in parallel: " + ", ".join(
                f"{d.name} → {mission.task(t).title}" for t, d in batch), LogLevel.ACTION)
        return batch

    async def _execute(self, mid: str, task_id: str, definition: AgentDefinition,
                       governor: Governor) -> tuple[str, AgentRunResult]:
        try:
            await self._board.assign_task(mid, task_id, definition.name, by=ATLAS)
            await self._board.update_task(mid, task_id, TaskStatus.IN_PROGRESS, by=ATLAS)
        except ValidationFailed as exc:
            return task_id, AgentRunResult(agent=definition.name, task_id=task_id, status=RunStatus.FAILED,
                                           summary=exc.message, stop_reason="not_startable")
        mission = await self._board.get(mid)
        hire = mission.hire_of(definition.name)
        result = await self._runtime.run(
            definition, mid, governor, task_id=task_id, mandate=hire.mandate if hire else None,
            inbox_seq=mission.inbox_cursors.get(definition.name, 0))
        if result.inbox_seq:
            await self._board.set_inbox_cursor(mid, definition.name, result.inbox_seq)
        return task_id, result

    # ── results ──────────────────────────────────────────────────────────
    async def _handle(self, mid: str, task_id: str, result: AgentRunResult, governor: Governor) -> None:
        mission = await self._board.get(mid)
        if mission.is_terminal:
            return
        task = mission.task(task_id)
        try:
            if result.status == RunStatus.DONE:
                await self._on_done(mission, task, result)
            else:
                await self._on_unsuccessful(mission, task, result, governor)
                return
        except ValidationFailed as exc:
            # An agent's result broke a mission rule (e.g. a plan with impossible dependencies). Atlas never
            # crashes on bad agent output: it stops that line of work and asks a human.
            log.warning("result of %s on %s rejected: %s", result.agent, task_id, exc.message)
            await self._escalate(mid, task_id, f"{result.agent}'s result for '{task.title}' could not be applied: "
                                               f"{exc.message}")
            return
        # An agent can finish its part and still need a human (e.g. Kenji found a real ambiguity).
        mission = await self._board.get(mid)
        asks = [i for i in mission.open_issues if i.needs_human and i.task_id == task_id]
        if asks and not mission.is_terminal and mission.status != MissionStatus.AWAITING_HUMAN:
            await self._decide(mid, DecisionType.CLARIFY, f"{result.agent} needs a human: {asks[0].text}", task_id)
            await self._board.set_status(mid, MissionStatus.AWAITING_HUMAN, by=ATLAS, reason=asks[0].text[:200])

    async def _on_unsuccessful(self, mission: Mission, task: Task, result: AgentRunResult, governor: Governor) -> None:
        mid, reason = mission.mission_id, result.stop_reason
        if reason in ("mission_stopped", "mission_missing"):
            return
        if reason == "budget":
            await self._finish(mid, MissionStatus.FAILED, result.summary, governor)
            return
        if reason == "not_startable":
            await self._escalate(mid, task.id, f"'{task.title}' cannot start again: {result.summary}")
            return
        asks = [i for i in mission.open_issues if i.needs_human and i.task_id == task.id]
        if result.status == RunStatus.BLOCKED and asks:
            await self._board.update_task(mid, task.id, TaskStatus.BLOCKED, by=ATLAS, summary=result.summary)
            await self._decide(mid, DecisionType.CLARIFY, f"{result.agent} needs a human: {asks[0].text}", task.id)
            await self._board.set_status(mid, MissionStatus.AWAITING_HUMAN, by=ATLAS, reason=asks[0].text[:200])
            return
        if reason == "llm_unavailable":
            # An outage is not the agent's failure: retry the same agent after a pause, never reassign;
            # once the attempts are spent, a human decides.
            if task.attempts < task.max_attempts:
                await asyncio.sleep(self._settings.llm_retry_backoff_s)
                await self._decide(mid, DecisionType.RETRY, f"'{task.title}' attempt {task.attempts}/{task.max_attempts}: "
                                   f"the language model is unavailable (llm_unavailable). Retrying after a pause.", task.id)
                await self._board.update_task(mid, task.id, TaskStatus.PENDING, by=ATLAS, summary=result.summary)
            else:
                await self._board.update_task(mid, task.id, TaskStatus.BLOCKED, by=ATLAS, summary=result.summary)
                await self._escalate(mid, task.id, f"'{task.title}' failed {task.attempts} time(s): the language "
                                                   f"model is unavailable. {result.summary[:200]}")
            return
        await self._board.update_task(mid, task.id, TaskStatus.FAILED, by=ATLAS, summary=result.summary)
        if task.attempts < task.max_attempts:
            switch = task.failures + 1 >= self._settings.escalate_after_failures  # this failure included
            await self._decide(mid, DecisionType.REASSIGN if switch else DecisionType.RETRY,
                               f"'{task.title}' attempt {task.attempts}/{task.max_attempts} ended {result.status}"
                               f"{f' ({reason})' if reason else ''}: {result.summary[:200]}. "
                               + ("Maya will bring in a different specialist." if switch else "Retrying."), task.id)
            await self._board.update_task(mid, task.id, TaskStatus.PENDING, by=ATLAS)
        else:
            await self._escalate(mid, task.id, f"'{task.title}' failed {task.attempts} time(s). Last result: "
                                               f"{result.summary[:300]}")

    async def _on_done(self, mission: Mission, task: Task, result: AgentRunResult) -> None:
        mid, origin = mission.mission_id, task.origin or ""
        if origin == "bootstrap:requirements":
            await self._apply_requirements(mission, task, result)
            return
        if origin == "bootstrap:plan":
            await self._apply_plan(mission, task, result)
            return
        await self._board.update_task(mid, task.id, TaskStatus.DONE, by=ATLAS, summary=result.summary)
        mission = await self._board.get(mid)
        task = mission.task(task.id)
        if origin == "bootstrap:recon":
            await self._board.add_finding(mid, Finding(
                agent=result.agent, kind=FindingKind.RECON, severity=Severity.INFO, title="Codebase recon",
                detail=json.dumps(result.data, default=str)[:3000] if result.data else result.summary,
                task_id=task.id, evidence_ids=result.evidence_ids))
        elif origin.startswith("arbitrate:"):
            await self._after_ruling(mission, task, result)
        elif task.kind == TaskKind.TEST:
            await self._after_test(mission, task)
        elif task.kind in (TaskKind.REVIEW, TaskKind.SECURITY):
            await self._after_review(mission, task, result)
        elif task.kind == TaskKind.VERIFY:
            await self._after_verify(mission, task)
        elif task.kind in WORK_KINDS:
            await self._after_work(mission, task)

    # ── planning results ─────────────────────────────────────────────────
    async def _apply_requirements(self, mission: Mission, task: Task, result: AgentRunResult) -> None:
        mid = mission.mission_id
        parsed = parse_requirements(result.data)
        if parsed is None and task.attempts < task.max_attempts:
            await self._board.update_task(mid, task.id, TaskStatus.FAILED, by=ATLAS, summary="no acceptance criteria")
            await self._decide(mid, DecisionType.RETRY, "Kenji finished without acceptance criteria; asking again.", task.id)
            await self._board.update_task(mid, task.id, TaskStatus.PENDING, by=ATLAS)
            return
        if parsed is None:
            parsed = fallback_requirements(mission)
            await self._decide(mid, DecisionType.CONTINUE, "No usable criteria from analysis; using conservative "
                                                           "fallback criteria derived from the request.", task.id)
        await self._board.set_requirements(mid, *parsed, by=result.agent)
        await self._board.update_task(mid, task.id, TaskStatus.DONE, by=ATLAS, summary=result.summary)
        await self._decide(mid, DecisionType.CONTINUE, f"{len(parsed[1])} acceptance criteria set.", task.id)

    async def _apply_plan(self, mission: Mission, task: Task, result: AgentRunResult) -> None:
        mid = mission.mission_id
        planned = parse_plan(result.data, mission)
        if not planned and task.attempts < task.max_attempts:
            await self._board.update_task(mid, task.id, TaskStatus.FAILED, by=ATLAS, summary="no usable tasks")
            await self._decide(mid, DecisionType.RETRY, "Ryo's plan had no usable tasks; asking again.", task.id)
            await self._board.update_task(mid, task.id, TaskStatus.PENDING, by=ATLAS)
            return
        note = ""
        if not planned:
            planned = fallback_plan(mission)
            note = " Ryo's plan was unusable; Atlas used a minimal fallback plan."
        tasks = enforce_gates(planned, mission)
        for t in tasks:
            if t.kind in CHECK_KINDS:
                t.max_attempts = self._settings.repair_max_attempts + 2  # checks legitimately re-run after repairs
        added = [t for t in tasks if t.created_by == ATLAS and t.origin and t.origin.startswith("gate:")]
        await self._board.add_tasks(mid, tasks, by=result.agent)
        await self._board.update_task(mid, task.id, TaskStatus.DONE, by=ATLAS, summary=result.summary)
        await self._decide(mid, DecisionType.CONTINUE,
                           f"Plan accepted: {len(tasks)} task(s)"
                           + (f"; Atlas added the required checks: {', '.join(t.title for t in added)}" if added else "")
                           + f".{note}", task.id)
        await self._board.set_status(mid, MissionStatus.RUNNING, by=ATLAS, reason="plan accepted")

    # ── feedback loops ───────────────────────────────────────────────────
    async def _after_test(self, mission: Mission, task: Task) -> None:
        mid = mission.mission_id
        runs = await self._board.evidence(mid, kind="test_run", task_id=task.id)
        latest = runs[-1] if runs else None
        if latest is None:
            await self._retry_check(mission, task, "finished without running the tests — a claim is not evidence")
            return
        if latest.passed:
            await self._decide(mid, DecisionType.CONTINUE, f"Tests pass: {latest.summary} ({latest.evidence_id}).", task.id)
            return
        summary = latest.data.get("summary") or {}
        if summary.get("no_runner"):
            await self._repair(mission, task, kind=TaskKind.TEST, skills=["test-authoring"],
                               reason="The project has no runnable tests",
                               details="run_tests found no test runner. Add a minimal test setup in the project's "
                                       "language and tests that cover the acceptance criteria, then run them.",
                               evidence_ids=[latest.evidence_id])
            return
        failing = ", ".join(summary.get("failing_tests") or []) or "see output"
        await self._repair(mission, task, reason="Tests fail",
                           details=f"{latest.summary}\nFailing: {failing}\nOutput (tail):\n"
                                   f"{str(latest.data.get('output_tail', ''))[-1800:]}",
                           evidence_ids=[latest.evidence_id])

    async def _after_review(self, mission: Mission, task: Task, result: AgentRunResult) -> None:
        mid, reviewer = mission.mission_id, task.assignee
        mine = [f for f in mission.findings if f.agent == reviewer and f.kind in (FindingKind.REVIEW, FindingKind.SECURITY)]
        disputed = [f for f in mine if f.status == FindingStatus.DISPUTED]
        if disputed:
            await self._arbitrate(mission, task, disputed)
            return
        blocking = [f for f in mine if f.status == FindingStatus.OPEN and f.severity in BLOCKING_SEVERITIES]
        approved = bool(result.data.get("approved"))
        if not blocking and approved:
            await self._decide(mid, DecisionType.CONTINUE, f"{reviewer} approved the change.", task.id)
            return
        if approved and blocking:
            await self._events.log(mid, f"Atlas: {reviewer} said approved but left {len(blocking)} open finding(s); "
                                        "the findings win.", LogLevel.WARN)
        what = "Security findings" if task.kind == TaskKind.SECURITY else "Review findings"
        details = "\n".join(f"- {f.id} [{f.severity}] {f.title}" + (f" ({f.file}:{f.line})" if f.file else "")
                            + (f": {f.detail[:400]}" if f.detail else "") for f in blocking) or result.summary
        await self._repair(mission, task, reason=f"{what} from {reviewer}",
                           details=details + "\nAddress each finding, then mark it addressed with update_finding "
                                             "— or dispute it with evidence if you believe it is wrong.",
                           finding_ids=[f.id for f in blocking])

    async def _arbitrate(self, mission: Mission, review: Task, disputed: list[Finding]) -> None:
        mid = mission.mission_id
        new_ids: list[str] = []
        for finding in disputed:
            origin = f"arbitrate:{finding.id}"
            if mission.task_by_origin(origin):
                continue
            [ruling] = await self._board.add_tasks(mid, [Task(
                title=f"Settle dispute: {finding.title}"[:160], kind=TaskKind.REVIEW, required_skills=["arbitration"],
                origin=origin, created_by=ATLAS,
                description=f"{review.assignee} raised finding {finding.id} ({finding.severity}): {finding.title}\n"
                            f"{finding.detail[:600]}\nThe developer disputes it: {finding.resolution or '(no reason given)'}\n"
                            "Decide on evidence: ruling=developer (dismiss the finding) or reviewer (it must be fixed), "
                            "or escalate if evidence cannot settle it.")], by=ATLAS)
            new_ids.append(ruling.id)
        if new_ids:
            await self._board.set_dependencies(mid, review.id, review.depends_on + new_ids, by=ATLAS)
            await self._decide(mid, DecisionType.REASSIGN, f"Developer disputes {len(new_ids)} finding(s) from "
                                                           f"{review.assignee}; the Arbiter decides.", review.id)
        await self._board.update_task(mid, review.id, TaskStatus.PENDING, by=ATLAS)

    async def _after_ruling(self, mission: Mission, task: Task, result: AgentRunResult) -> None:
        mid = mission.mission_id
        finding = mission.finding(task.origin.split(":", 1)[1])
        ruling = str(result.data.get("ruling", "")).strip().lower()
        reason = str(result.data.get("reason") or result.summary)[:400]
        review = next((t for t in mission.tasks if task.id in t.depends_on), None)
        if ruling == "developer":
            if finding.status != FindingStatus.DISMISSED:
                await self._board.set_finding_status(mid, finding.id, FindingStatus.DISMISSED, by=result.agent,
                                                     resolution=f"Arbiter ruled for the developer: {reason}")
            await self._decide(mid, DecisionType.CONTINUE, f"Arbiter dismissed '{finding.title}': {reason}", task.id)
        elif ruling == "reviewer":
            await self._board.set_finding_status(mid, finding.id, FindingStatus.OPEN, by=result.agent,
                                                 resolution=f"Arbiter upheld the finding: {reason}")
            await self._decide(mid, DecisionType.CONTINUE, f"Arbiter upheld '{finding.title}': {reason}", task.id)
            if review is not None:
                await self._repair(await self._board.get(mid), review, reason="Arbiter upheld a finding",
                                   details=f"- {finding.id} [{finding.severity}] {finding.title}: {finding.detail[:400]}\n"
                                           f"Arbiter: {reason}", finding_ids=[finding.id])
        else:
            await self._escalate(mid, task.id, f"Developer and reviewer disagree on '{finding.title}' and evidence "
                                               f"does not settle it: {reason}")

    async def _after_verify(self, mission: Mission, task: Task) -> None:
        mid = mission.mission_id
        wanted = set(task.criteria_ids) or {c.id for c in mission.acceptance_criteria}
        criteria = [c for c in mission.acceptance_criteria if c.id in wanted]
        unchecked = [c for c in criteria if c.status == CriterionStatus.PENDING]
        unmet = [c for c in criteria if c.status == CriterionStatus.UNMET]
        unverifiable = [c for c in criteria if c.status == CriterionStatus.UNVERIFIABLE]
        if not (unchecked or unmet or unverifiable):
            await self._decide(mid, DecisionType.CONTINUE, f"All {len(criteria)} criteria verified with evidence.", task.id)
            return
        if unmet:
            await self._repair(mission, task, reason="Acceptance criteria not met", details="\n".join(
                f"- {c.id} {c.text}: {c.note or 'unmet'}" for c in unmet))
            return
        if unverifiable:
            await self._repair(mission, task, kind=TaskKind.TEST, skills=["test-authoring"],
                               reason="Criteria cannot be verified yet", details="Write tests that prove these:\n" +
                               "\n".join(f"- {c.id} {c.text}: {c.note or 'no evidence'}" for c in unverifiable))
            return
        await self._retry_check(mission, task, f"left {len(unchecked)} criteria unchecked")

    async def _after_work(self, mission: Mission, task: Task) -> None:
        """Code changed: every check that already ran is stale and runs again."""
        stale = [t for t in mission.tasks if t.kind in CHECK_KINDS and t.status == TaskStatus.DONE
                 and not (t.origin or "").startswith("arbitrate:")]
        for check in stale:
            await self._board.update_task(mission.mission_id, check.id, TaskStatus.PENDING, by=ATLAS)
        if stale:
            await self._decide(mission.mission_id, DecisionType.CONTINUE, f"'{task.title}' changed the code; re-running "
                               f"{len(stale)} check(s): {', '.join(t.title for t in stale)}.", task.id)

    async def _repair(
        self, mission: Mission, source: Task, *, reason: str, details: str, kind: TaskKind = TaskKind.FIX,
        skills: list[str] | None = None, evidence_ids: list[str] | None = None, finding_ids: list[str] | None = None,
    ) -> None:
        """Send work back: add (or reopen) a repair task, and re-run `source` after it."""
        mid = mission.mission_id
        origin = f"{'repair' if kind == TaskKind.FIX else 'author-tests'}:{source.id}"
        existing = mission.task_by_origin(origin)
        if existing and existing.attempts >= existing.max_attempts:
            await self._escalate(mid, source.id, f"{reason} after {existing.attempts} repair round(s). "
                                                 f"Latest: {details[:400]}")
            return
        refs = ""
        if evidence_ids:
            refs += f"\nEvidence: {', '.join(evidence_ids)}"
        if finding_ids:
            refs += f"\nFindings: {', '.join(finding_ids)}"
        description = f"{reason} (raised by '{source.title}').\n{details}{refs}"
        if existing:
            # The last repair round did not fix it: that counts as a failure (Maya escalates on these).
            await self._board.set_task_details(mid, existing.id, by=ATLAS, description=description, add_failure=True)
            if existing.status in (TaskStatus.DONE, TaskStatus.FAILED):
                await self._board.update_task(mid, existing.id, TaskStatus.PENDING, by=ATLAS)
            repair = existing
        else:
            title = ("Fix: " if kind == TaskKind.FIX else "Write tests: ") + reason
            [repair] = await self._board.add_tasks(mid, [Task(
                title=title[:160], kind=kind, description=description, required_skills=skills or [],
                criteria_ids=source.criteria_ids, origin=origin, max_attempts=self._settings.repair_max_attempts,
                created_by=ATLAS)], by=ATLAS)
        current = (await self._board.get(mid)).task(source.id)
        if repair.id not in current.depends_on:
            await self._board.set_dependencies(mid, source.id, current.depends_on + [repair.id], by=ATLAS)
        if current.status == TaskStatus.DONE:
            await self._board.update_task(mid, source.id, TaskStatus.PENDING, by=ATLAS)
        round_no = (existing.attempts + 1) if existing else 1
        await self._decide(mid, DecisionType.REPLAN,
                           f"{reason} → {'reopened' if existing else 'added'} '{repair.title}' (round {round_no}); "
                           f"'{source.title}' runs again after it."
                           + (f" Evidence: {', '.join(evidence_ids)}." if evidence_ids else ""), source.id)

    async def _retry_check(self, mission: Mission, task: Task, why: str) -> None:
        mid = mission.mission_id
        if task.attempts >= task.max_attempts:
            await self._escalate(mid, task.id, f"{task.assignee} {why}, {task.attempts} time(s).")
            return
        await self._decide(mid, DecisionType.RETRY, f"{task.assignee} {why}; running '{task.title}' again.", task.id)
        await self._board.update_task(mid, task.id, TaskStatus.PENDING, by=ATLAS)

    # ── ending ───────────────────────────────────────────────────────────
    async def _try_complete(self, mission: Mission, governor: Governor) -> bool:
        if mission.status != MissionStatus.RUNNING:
            return False
        if any(t.status not in (TaskStatus.DONE, TaskStatus.CANCELLED) for t in mission.tasks):
            return False
        if not mission.all_criteria_met or self._blocking_findings(mission):
            return False
        mid = mission.mission_id
        await self._record_diff(mid, governor)
        await self._decide(mid, DecisionType.COMPLETE, f"All {len(mission.acceptance_criteria)} acceptance criteria "
                                                       "verified with evidence; tests, review and security passed.")
        await self._board.set_status(mid, MissionStatus.COMPLETED, by=ATLAS, reason="all criteria verified")
        await self._wrap_up(mid, "mission complete")
        await self._events.log(mid, "MISSION COMPLETE — verified change ready for human review before it is applied.",
                               LogLevel.SUCCESS)
        return True

    async def _finish(self, mid: str, status: MissionStatus, reason: str, governor: Governor | None = None) -> None:
        mission = await self._board.get(mid)
        if mission.is_terminal:
            return
        if governor is not None:
            await self._record_diff(mid, governor, respect_budget=True)
        await self._decide(mid, DecisionType.STOP, reason)
        await self._board.set_status(mid, status, by=ATLAS, reason=reason[:300])
        await self._wrap_up(mid, reason)

    async def _wrap_up(self, mid: str, reason: str) -> None:
        mission = await self._board.get(mid)
        for agent in mission.hired_agents:
            await self._hiring.release(mid, agent, reason)
        mission = await self._board.get(mid)
        report = build_report(mission, await self._board.evidence(mid), await self._audit_list(mid))
        await self._board.set_report(mid, report)

    async def _record_diff(self, mid: str, governor: Governor, respect_budget: bool = False) -> None:
        if respect_budget and governor.exhausted:
            return
        ctx = ToolContext(mission_id=mid, agent=ATLAS, workspace=self._workspaces.get(mid), settings=self._settings,
                          approver=self._approver, events=self._events, board=self._board)
        await governor.execute("git_diff", {"stat_only": True}, ctx)

    async def _unstick(self, mission: Mission) -> None:
        """Nothing is ready and the mission is not complete: find out why, fix it or ask a human."""
        mid = mission.mission_id
        blockers = [t for t in mission.tasks if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)]
        pending = [t for t in mission.tasks if t.status == TaskStatus.PENDING]
        if pending and blockers:
            await self._escalate(mid, blockers[0].id, "Work is waiting on task(s) that cannot finish: "
                                 + ", ".join(f"'{t.title}' ({t.status})" for t in blockers))
            return
        if mission.status == MissionStatus.RUNNING and not pending:
            findings = self._blocking_findings(mission)
            if findings:
                await self._escalate(mid, None, "Checks are done but blocking findings remain open: "
                                     + ", ".join(f.title for f in findings[:5]))
                return
            verify = [t for t in mission.tasks if t.kind == TaskKind.VERIFY]
            if verify and not mission.all_criteria_met:
                await self._retry_check(mission, verify[-1], "did not verify every criterion")
                return
        await self._escalate(mid, None, f"The mission is stuck in {mission.status} with no runnable work.")

    @staticmethod
    def _blocking_findings(mission: Mission) -> list[Finding]:
        return [f for f in mission.findings if f.status == FindingStatus.OPEN
                and f.kind in (FindingKind.REVIEW, FindingKind.SECURITY) and f.severity in (Severity.HIGH, Severity.CRITICAL)]

    # ── helpers ──────────────────────────────────────────────────────────
    async def _decide(self, mid: str, decision: DecisionType, reason: str, task_id: str | None = None) -> None:
        await self._board.record_decision(mid, Decision(by=ATLAS, decision=decision, reason=reason, task_id=task_id))
        level = LogLevel.WARN if decision in (DecisionType.ESCALATE, DecisionType.STOP, DecisionType.CLARIFY) else LogLevel.ACTION
        await self._events.log(mid, f"Atlas [{decision}]: {reason}", level)

    async def _escalate(self, mid: str, task_id: str | None, text: str) -> None:
        mission = await self._board.get(mid)
        if mission.is_terminal:
            return
        await self._board.raise_issue(mid, OpenIssue(raised_by=ATLAS, kind=IssueKind.BLOCKER, text=text,
                                                     needs_human=True, task_id=task_id))
        await self._decide(mid, DecisionType.ESCALATE, text, task_id)
        if mission.status != MissionStatus.AWAITING_HUMAN:
            await self._board.set_status(mid, MissionStatus.AWAITING_HUMAN, by=ATLAS, reason=text[:200])
