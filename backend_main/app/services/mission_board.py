"""
The mission board: the one service agents use to read and change shared
mission state, record evidence, and talk to each other.

Every change is a load → apply → save with optimistic concurrency, retried on
conflict, so agents running in parallel (or on different workers) never
overwrite each other. Every change is also published as an event, which is
how the control room and the run log see the mission unfold.
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from app.core.errors import ConflictError, NotFoundError, ValidationFailed
from app.domain.events import EventKind, LogLevel
from app.domain.hiring import HireRecord
from app.domain.mission import (
    AcceptanceCriterion,
    AgentMessage,
    ApplyRecord,
    CriterionStatus,
    Decision,
    Evidence,
    Finding,
    FindingStatus,
    Mission,
    MissionError,
    MissionStatus,
    OpenIssue,
    Task,
    TaskStatus,
)
from app.infrastructure.storage.base import ConcurrencyConflict, Storage
from app.security.sandbox import WorkspaceManager
from app.services.event_publisher import EventPublisher

log = logging.getLogger(__name__)
T = TypeVar("T")

MAX_SAVE_ATTEMPTS = 6


def evidence_verdict(kind: str, data: dict[str, Any]) -> bool | None:
    """Did the check pass? None for records without a verdict (a diff just shows what changed)."""
    if "passed" in data and isinstance(data["passed"], bool):
        return data["passed"]
    if kind in ("secret_scan", "static_scan"):
        return int(data.get("count", 0)) == 0
    return None


def _summarize(kind: str, data: dict[str, Any]) -> str:
    if kind == "test_run":
        s = data.get("summary", {})
        if s.get("no_runner"):
            return "no test runner detected — the project has no runnable tests"
        if s.get("no_tests_collected"):
            return f"{data.get('command', 'tests')}: no tests were collected (exit {data.get('exit_code')})"
        return (f"{data.get('command', 'tests')}: {s.get('passed', 0)} passed, {s.get('failed', 0)} failed, "
                f"{s.get('errors', 0)} errors (exit {data.get('exit_code')})")
    if kind == "diff":
        return f"{data.get('files_changed', 0)} file(s) changed, +{data.get('insertions', 0)} −{data.get('deletions', 0)}"
    if kind in ("secret_scan", "static_scan"):
        return f"{kind}: {data.get('count', 0)} finding(s) in {data.get('scope', 'repository')}"
    return kind


def _dependency_order(tasks: list[Task]) -> list[Task]:
    batch = {t.id: t for t in tasks}
    placed: list[Task] = []
    seen: set[str] = set()
    remaining = list(tasks)
    while remaining:
        progressed = False
        for task in list(remaining):
            if all(d in seen or d not in batch for d in task.depends_on):
                placed.append(task)
                seen.add(task.id)
                remaining.remove(task)
                progressed = True
        if not progressed:  # a cycle inside the batch: keep the given order and let the mission reject it
            return placed + remaining
    return placed


def _title_from(request: str) -> str:
    first = request.strip().splitlines()[0] if request.strip() else "Untitled mission"
    return first if len(first) <= 80 else first[:77] + "…"


class MissionBoard:
    def __init__(self, storage: Storage, events: EventPublisher, workspaces: WorkspaceManager) -> None:
        self._storage = storage
        self._events = events
        self._workspaces = workspaces
        # Serialises writers inside one process; the version check covers other processes.
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    # ── lifecycle ────────────────────────────────────────────────────────
    async def create_mission(
        self, request: str, repo_path: str, *, title: str | None = None, created_by: str = "operator"
    ) -> Mission:
        if not request.strip():
            raise ValidationFailed("The request is empty")
        mission = Mission(title=title or _title_from(request), request=request.strip(),
                          repo_path=str(Path(repo_path).expanduser().resolve()), created_by=created_by)
        workspace = await self._workspaces.create(mission.mission_id, Path(repo_path))
        mission.workspace_path = str(workspace.root)
        mission.baseline_commit = workspace.baseline_commit
        try:
            stored = await self._storage.missions.create(mission)
        except Exception:
            await self._workspaces.destroy(mission.mission_id)
            raise
        await self._events.publish(stored.mission_id, EventKind.MISSION_CREATED, {
            "title": stored.title, "request": stored.request, "repo_path": stored.repo_path,
            "baseline_commit": stored.baseline_commit, "created_by": created_by,
        })
        log.info("mission %s created for %s", stored.mission_id, stored.repo_path)
        return stored

    async def get(self, mission_id: str) -> Mission:
        mission = await self._storage.missions.get(mission_id)
        if mission is None:
            raise NotFoundError(f"Mission {mission_id} not found")
        return mission

    async def list(self, *, status: MissionStatus | None = None, limit: int = 50) -> list[Mission]:
        return await self._storage.missions.list(status=status, limit=limit)

    async def mutate(self, mission_id: str, change: Callable[[Mission], T]) -> tuple[Mission, T]:
        """Apply `change` to the latest mission and save it, retrying on concurrent edits.
        `change` must only touch the mission (it may run more than once)."""
        async with self._locks[mission_id]:
            for attempt in range(MAX_SAVE_ATTEMPTS):
                mission = await self.get(mission_id)
                try:
                    result = change(mission)
                except MissionError as exc:
                    raise ValidationFailed(str(exc)) from None
                try:
                    return await self._storage.missions.save(mission), result
                except ConcurrencyConflict:
                    await asyncio.sleep(random.uniform(0.01, 0.05) * (attempt + 1))
        raise ConflictError(f"Mission {mission_id} is changing too fast; try again")

    # ── status ───────────────────────────────────────────────────────────
    async def set_status(self, mission_id: str, status: MissionStatus, *, by: str, reason: str | None = None) -> Mission:
        def change(m: Mission) -> bool:
            changed = m.status != status
            m.transition(status, reason)
            return changed

        mission, changed = await self.mutate(mission_id, change)
        if not changed:
            return mission  # idempotent: repeating a status change is not news
        await self._events.publish(mission_id, EventKind.MISSION_STATUS,
                                   {"status": status, "reason": reason, "by": by, "progress": mission.progress()})
        if status == MissionStatus.COMPLETED:
            await self._events.publish(mission_id, EventKind.COMPLETE, {"status": status})
        return mission

    # ── plan ─────────────────────────────────────────────────────────────
    async def set_requirements(self, mission_id: str, objectives: list[str], criteria: list[str], *, by: str) -> Mission:
        mission, _ = await self.mutate(mission_id, lambda m: m.set_requirements(objectives, criteria))
        await self._events.publish(mission_id, EventKind.PLAN_UPDATED, {
            "action": "requirements", "by": by,
            "objectives": [o.model_dump() for o in mission.objectives],
            "acceptance_criteria": [c.model_dump() for c in mission.acceptance_criteria],
        })
        return mission

    async def add_tasks(self, mission_id: str, tasks: list[Task], *, by: str) -> list[Task]:
        """Add a batch of tasks. Dependencies inside the batch may point either way; tasks are added in
        dependency order (a cycle or an unknown dependency is rejected by the mission)."""
        ordered = _dependency_order(tasks)

        def change(m: Mission) -> list[Task]:
            added = []
            for task in ordered:
                task = task.model_copy(deep=True, update={"created_by": task.created_by if task.created_by != "system" else by})
                added.append(m.add_task(task))
            return added

        _, added = await self.mutate(mission_id, change)
        await self._events.publish(mission_id, EventKind.PLAN_UPDATED, {
            "action": "tasks_added", "by": by, "tasks": [t.model_dump() for t in added],
        })
        return added

    async def set_dependencies(self, mission_id: str, task_id: str, depends_on: list[str], *, by: str) -> Task:
        def change(m: Mission) -> Task:
            m.set_dependencies(task_id, depends_on)
            return m.task(task_id)

        _, task = await self.mutate(mission_id, change)
        await self._events.publish(mission_id, EventKind.PLAN_UPDATED,
                                   {"action": "dependencies", "by": by, "task_id": task_id, "depends_on": task.depends_on})
        return task

    async def assign_task(self, mission_id: str, task_id: str, agent: str, *, by: str) -> Task:
        _, task = await self.mutate(mission_id, lambda m: m.assign(task_id, agent))
        await self._publish_task(mission_id, task, by=by, action="assigned")
        return task

    async def update_task(
        self, mission_id: str, task_id: str, status: TaskStatus, *, by: str, summary: str | None = None
    ) -> Task:
        _, task = await self.mutate(mission_id, lambda m: m.update_task(task_id, status, summary=summary))
        await self._publish_task(mission_id, task, by=by, action="status")
        return task

    async def set_task_details(
        self, mission_id: str, task_id: str, *, by: str, description: str | None = None,
        max_attempts: int | None = None, add_failure: bool = False,
    ) -> Task:
        _, task = await self.mutate(mission_id, lambda m: m.set_task_details(
            task_id, description=description, max_attempts=max_attempts, add_failure=add_failure))
        await self._publish_task(mission_id, task, by=by, action="details")
        return task

    async def set_inbox_cursor(self, mission_id: str, agent: str, seq: int) -> None:
        def change(m: Mission) -> None:
            if seq > m.inbox_cursors.get(agent, 0):
                m.inbox_cursors[agent] = seq
                m.touch()

        await self.mutate(mission_id, change)

    async def record_application(self, mission_id: str, record: ApplyRecord) -> Mission:
        mission, _ = await self.mutate(mission_id, lambda m: m.record_application(record))
        await self._events.publish(mission_id, EventKind.APPLY, record.model_dump(mode="json"))
        return mission

    async def set_report(self, mission_id: str, report: str) -> None:
        def change(m: Mission) -> None:
            m.report = report
            m.touch()

        await self.mutate(mission_id, change)

    async def _publish_task(self, mission_id: str, task: Task, *, by: str, action: str) -> None:
        await self._events.publish(mission_id, EventKind.TASK_UPDATED, {
            "action": action, "by": by, "task_id": task.id, "title": task.title, "status": task.status,
            "assignee": task.assignee, "attempts": task.attempts, "summary": task.result_summary,
        })

    async def record_hire(self, mission_id: str, record: HireRecord) -> tuple[HireRecord, bool]:
        """Returns the hire and whether it is new (False if the agent was already on the task force)."""
        def change(m: Mission) -> tuple[HireRecord, bool]:
            existing = m.hire_of(record.agent)
            return (existing, False) if existing else (m.record_hire(record.model_copy(deep=True)), True)

        _, (hire, new) = await self.mutate(mission_id, change)
        return hire, new

    async def release(self, mission_id: str, agent: str, reason: str) -> HireRecord:
        _, hire = await self.mutate(mission_id, lambda m: m.release(agent, reason))
        return hire

    # ── findings, decisions, issues ──────────────────────────────────────
    async def add_finding(self, mission_id: str, finding: Finding) -> Finding:
        _, stored = await self.mutate(mission_id, lambda m: m.add_finding(finding.model_copy(deep=True)))
        await self._events.publish(mission_id, EventKind.FINDING, {"action": "added", **stored.model_dump()})
        return stored

    async def set_finding_status(
        self, mission_id: str, finding_id: str, status: FindingStatus, *, by: str, resolution: str | None = None
    ) -> Finding:
        _, finding = await self.mutate(
            mission_id, lambda m: m.set_finding_status(finding_id, status, by=by, resolution=resolution)
        )
        await self._events.publish(mission_id, EventKind.FINDING, {"action": "status", **finding.model_dump()})
        return finding

    async def record_decision(self, mission_id: str, decision: Decision) -> Decision:
        _, stored = await self.mutate(mission_id, lambda m: m.record_decision(decision.model_copy(deep=True)))
        await self._events.publish(mission_id, EventKind.DECISION, stored.model_dump())
        return stored

    async def raise_issue(self, mission_id: str, issue: OpenIssue) -> OpenIssue:
        _, stored = await self.mutate(mission_id, lambda m: m.raise_issue(issue.model_copy(deep=True)))
        await self._events.publish(mission_id, EventKind.ISSUE, {"action": "raised", **stored.model_dump()})
        if stored.needs_human:
            await self._events.log(mission_id, f"{stored.raised_by} needs a human: {stored.text}", LogLevel.WARN)
        return stored

    async def resolve_issue(self, mission_id: str, issue_id: str, *, by: str, resolution: str) -> OpenIssue:
        _, issue = await self.mutate(mission_id, lambda m: m.resolve_issue(issue_id, by=by, resolution=resolution))
        await self._events.publish(mission_id, EventKind.ISSUE, {"action": "resolved", **issue.model_dump()})
        return issue

    # ── evidence & verification ──────────────────────────────────────────
    async def record_evidence(
        self, mission_id: str, *, agent: str, tool: str, data: dict[str, Any], task_id: str | None = None
    ) -> Evidence:
        kind = str(data.get("kind") or tool)
        evidence = Evidence(
            mission_id=mission_id, kind=kind, agent=agent, tool=tool, task_id=task_id,
            passed=evidence_verdict(kind, data), summary=_summarize(kind, data),
            data={k: v for k, v in data.items() if k != "kind"},
        )
        await self._storage.evidence.add(evidence)
        await self._events.publish(mission_id, EventKind.EVIDENCE, {
            "evidence_id": evidence.evidence_id, "kind": kind, "agent": agent, "tool": tool, "task_id": task_id,
            "passed": evidence.passed, "summary": evidence.summary,
        })
        return evidence

    async def evidence(self, mission_id: str, *, kind: str | None = None, task_id: str | None = None) -> list[Evidence]:
        return await self._storage.evidence.list(mission_id, kind=kind, task_id=task_id)

    async def verify_criterion(
        self,
        mission_id: str,
        criterion_id: str,
        status: CriterionStatus,
        *,
        by: str,
        evidence_ids: list[str],
        note: str | None = None,
    ) -> AcceptanceCriterion:
        """Evidence-based completion: a criterion is only MET when it cites real, passing evidence."""
        cited = await self._storage.evidence.get_many(mission_id, evidence_ids) if evidence_ids else []
        missing = set(evidence_ids) - {e.evidence_id for e in cited}
        if missing:
            raise ValidationFailed(f"Unknown evidence for this mission: {', '.join(sorted(missing))}")
        if status == CriterionStatus.MET:
            failing = [e for e in cited if e.passed is False]
            if failing:
                raise ValidationFailed(
                    "Cannot mark the criterion met: it cites failing evidence "
                    + ", ".join(f"{e.evidence_id} ({e.summary})" for e in failing)
                )
        _, criterion = await self.mutate(
            mission_id, lambda m: m.set_criterion(criterion_id, status, by=by, evidence_ids=evidence_ids, note=note)
        )
        await self._events.publish(mission_id, EventKind.CRITERION, criterion.model_dump())
        return criterion

    # ── messages ─────────────────────────────────────────────────────────
    async def post_message(self, message: AgentMessage) -> AgentMessage:
        stored = await self._storage.messages.post(message)
        await self._events.publish(message.mission_id, EventKind.MESSAGE, {
            "message_id": stored.message_id, "seq": stored.seq, "from": stored.sender, "to": stored.to,
            "kind": stored.kind, "subject": stored.subject, "body": stored.body[:500], "task_id": stored.task_id,
            "finding_ids": stored.finding_ids, "evidence_ids": stored.evidence_ids, "in_reply_to": stored.in_reply_to,
        })
        return stored

    async def inbox(
        self, mission_id: str, agent: str, *, teams: set[str] | None = None, after_seq: int = 0, limit: int = 100
    ) -> list[AgentMessage]:
        """Messages for `agent` (direct, its teams, or broadcast) after `after_seq`, excluding its own."""
        messages = await self._storage.messages.list(mission_id, after_seq=after_seq, limit=2000)
        mine = [m for m in messages if m.sender != agent and m.addressed_to(agent, teams)]
        return mine[:limit]

    async def messages(self, mission_id: str, *, after_seq: int = 0, limit: int = 500) -> list[AgentMessage]:
        return await self._storage.messages.list(mission_id, after_seq=after_seq, limit=limit)
