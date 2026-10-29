"""
The Mission aggregate — DevPilot's shared mission state (the "blackboard").

Holds what every agent needs to coordinate: objectives, acceptance criteria,
the task DAG, findings, decisions and open issues. Bulky, append-only data
(evidence, agent messages) lives in its own repositories and is referenced
by id, so this document stays small enough to load and save on every change.

Invariants enforced here, not in prompts:
- The task graph stays acyclic and only references tasks that exist.
- Status changes follow the allowed transitions.
- An acceptance criterion can only be MET when it cites evidence.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.core.ids import new_id, now_ms
from app.domain.hiring import HireRecord


class MissionError(ValueError):
    """A change would break a mission invariant."""


# ── enums ───────────────────────────────────────────────────────────────────
class MissionStatus(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RUNNING = "running"
    AWAITING_HUMAN = "awaiting_human"   # clarification, dispute or final approval
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


TERMINAL_STATUSES = {MissionStatus.COMPLETED, MissionStatus.FAILED, MissionStatus.STOPPED}

_MISSION_TRANSITIONS: dict[MissionStatus, set[MissionStatus]] = {
    MissionStatus.CREATED: {MissionStatus.PLANNING, MissionStatus.STOPPED, MissionStatus.FAILED},
    MissionStatus.PLANNING: {MissionStatus.RUNNING, MissionStatus.AWAITING_HUMAN, MissionStatus.STOPPED, MissionStatus.FAILED},
    MissionStatus.RUNNING: {MissionStatus.PLANNING, MissionStatus.AWAITING_HUMAN, MissionStatus.COMPLETED,
                            MissionStatus.STOPPED, MissionStatus.FAILED},
    MissionStatus.AWAITING_HUMAN: {MissionStatus.PLANNING, MissionStatus.RUNNING, MissionStatus.COMPLETED,
                                   MissionStatus.STOPPED, MissionStatus.FAILED},
}


class TaskKind(StrEnum):
    ANALYSIS = "analysis"       # requirements, recon
    PLAN = "plan"
    IMPLEMENT = "implement"
    FIX = "fix"
    TEST = "test"
    REVIEW = "review"
    SECURITY = "security"
    VERIFY = "verify"
    REPORT = "report"


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TASK_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
    TaskStatus.IN_PROGRESS: {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.BLOCKED, TaskStatus.PENDING, TaskStatus.CANCELLED},
    TaskStatus.BLOCKED: {TaskStatus.PENDING, TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
    # A failed task can be retried; a done task can be reopened when new evidence shows it is not done.
    TaskStatus.FAILED: {TaskStatus.PENDING, TaskStatus.CANCELLED},
    TaskStatus.DONE: {TaskStatus.PENDING},
    TaskStatus.CANCELLED: set(),
}


class CriterionStatus(StrEnum):
    PENDING = "pending"
    MET = "met"
    UNMET = "unmet"
    UNVERIFIABLE = "unverifiable"


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingKind(StrEnum):
    RECON = "recon"             # facts about the codebase (conventions, test command, layout)
    RISK = "risk"
    BUG = "bug"
    TEST_FAILURE = "test_failure"
    REVIEW = "review"
    SECURITY = "security"
    NOTE = "note"


class FindingStatus(StrEnum):
    OPEN = "open"
    ADDRESSED = "addressed"
    DISPUTED = "disputed"
    DISMISSED = "dismissed"


class DecisionType(StrEnum):
    CONTINUE = "continue"
    RETRY = "retry"
    REASSIGN = "reassign"
    REPLAN = "replan"
    HIRE = "hire"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    STOP = "stop"
    COMPLETE = "complete"


class IssueKind(StrEnum):
    QUESTION = "question"
    CLARIFICATION = "clarification"   # needs the human who made the request
    BLOCKER = "blocker"
    DISPUTE = "dispute"               # developer vs reviewer


class IssueStatus(StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


# ── parts ───────────────────────────────────────────────────────────────────
class Objective(BaseModel):
    id: str = Field(default_factory=lambda: new_id("obj"))
    text: str


class AcceptanceCriterion(BaseModel):
    id: str = Field(default_factory=lambda: new_id("ac"))
    text: str
    status: CriterionStatus = CriterionStatus.PENDING
    evidence_ids: list[str] = Field(default_factory=list)
    verified_by: str | None = None
    note: str | None = None
    updated_at: int | None = None


class Task(BaseModel):
    id: str = Field(default_factory=lambda: new_id("tsk"))
    title: str
    description: str = ""
    kind: TaskKind = TaskKind.IMPLEMENT
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    criteria_ids: list[str] = Field(default_factory=list)
    assignee: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    result_summary: str | None = None
    # Why Atlas created this task: "bootstrap:requirements", "gate:review", "repair:<task id>",
    # "arbitrate:<finding id>"… None for tasks from the planner.
    origin: str | None = None
    created_by: str = "system"
    created_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)


class Finding(BaseModel):
    id: str = Field(default_factory=lambda: new_id("fnd"))
    agent: str
    kind: FindingKind
    severity: Severity = Severity.INFO
    title: str
    detail: str = ""
    file: str | None = None
    line: int | None = None
    task_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    status: FindingStatus = FindingStatus.OPEN
    resolution: str | None = None
    resolved_by: str | None = None
    at: int = Field(default_factory=now_ms)


class Decision(BaseModel):
    id: str = Field(default_factory=lambda: new_id("dec"))
    by: str
    decision: DecisionType
    reason: str
    task_id: str | None = None
    at: int = Field(default_factory=now_ms)


class OpenIssue(BaseModel):
    id: str = Field(default_factory=lambda: new_id("iss"))
    raised_by: str
    kind: IssueKind
    text: str
    needs_human: bool = False
    task_id: str | None = None
    status: IssueStatus = IssueStatus.OPEN
    resolution: str | None = None
    resolved_by: str | None = None
    at: int = Field(default_factory=now_ms)


# ── aggregate ───────────────────────────────────────────────────────────────
class Mission(BaseModel):
    mission_id: str = Field(default_factory=lambda: new_id("msn"))
    title: str
    request: str
    repo_path: str
    workspace_path: str | None = None
    baseline_commit: str | None = None
    status: MissionStatus = MissionStatus.CREATED
    status_reason: str | None = None
    objectives: list[Objective] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    issues: list[OpenIssue] = Field(default_factory=list)
    hires: list[HireRecord] = Field(default_factory=list)
    # Last message each agent has seen, so a resumed run does not replay old messages.
    inbox_cursors: dict[str, int] = Field(default_factory=dict)
    # The final evidence report (Markdown), written when the mission ends.
    report: str | None = None
    created_by: str = "operator"
    created_at: int = Field(default_factory=now_ms)
    updated_at: int = Field(default_factory=now_ms)
    # Optimistic-concurrency token: bumped by the repository on every save.
    version: int = 0

    # ── lookups ──────────────────────────────────────────────
    def task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise MissionError(f"unknown task {task_id}")

    def criterion(self, criterion_id: str) -> AcceptanceCriterion:
        for criterion in self.acceptance_criteria:
            if criterion.id == criterion_id:
                return criterion
        raise MissionError(f"unknown acceptance criterion {criterion_id}")

    def finding(self, finding_id: str) -> Finding:
        for finding in self.findings:
            if finding.id == finding_id:
                return finding
        raise MissionError(f"unknown finding {finding_id}")

    def issue(self, issue_id: str) -> OpenIssue:
        for issue in self.issues:
            if issue.id == issue_id:
                return issue
        raise MissionError(f"unknown issue {issue_id}")

    @property
    def hired_agents(self) -> list[str]:
        """Agents currently on the task force."""
        return [h.agent for h in self.hires if h.active]

    def hire_of(self, agent: str) -> HireRecord | None:
        for hire in reversed(self.hires):
            if hire.agent.lower() == agent.lower() and hire.active:
                return hire
        return None

    def record_hire(self, record: HireRecord) -> HireRecord:
        self._ensure_active()
        existing = self.hire_of(record.agent)
        if existing is not None:
            return existing  # already on the task force; hiring is idempotent
        self.hires.append(record)
        self.touch()
        return record

    def release(self, agent: str, reason: str) -> HireRecord:
        hire = self.hire_of(agent)
        if hire is None:
            raise MissionError(f"{agent} is not on this mission's task force")
        hire.released_at = now_ms()
        hire.release_reason = reason
        self.touch()
        return hire

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    # ── status ───────────────────────────────────────────────
    def transition(self, status: MissionStatus, reason: str | None = None) -> None:
        if status == self.status:
            return
        if status not in _MISSION_TRANSITIONS.get(self.status, set()):
            raise MissionError(f"mission cannot go from {self.status} to {status}")
        if status == MissionStatus.COMPLETED and not self.all_criteria_met:
            raise MissionError("mission cannot complete: not every acceptance criterion is met with evidence")
        self.status = status
        self.status_reason = reason
        self.touch()

    def touch(self) -> None:
        self.updated_at = now_ms()

    def _ensure_active(self) -> None:
        if self.is_terminal:
            raise MissionError(f"mission is {self.status}; it can no longer change")

    # ── plan ─────────────────────────────────────────────────
    def set_requirements(self, objectives: list[str], criteria: list[str]) -> None:
        """Replace objectives and criteria (Kenji's analysis). Keeps verified criteria whose text is unchanged."""
        self._ensure_active()
        if not criteria:
            raise MissionError("at least one acceptance criterion is required")
        existing = {c.text.strip().lower(): c for c in self.acceptance_criteria}
        self.objectives = [Objective(text=t.strip()) for t in objectives if t.strip()]
        self.acceptance_criteria = [
            existing.get(t.strip().lower()) or AcceptanceCriterion(text=t.strip()) for t in criteria if t.strip()
        ]
        self.touch()

    def add_task(self, task: Task) -> Task:
        self._ensure_active()
        ids = {t.id for t in self.tasks}
        if task.id in ids:
            raise MissionError(f"task {task.id} already exists")
        missing = [d for d in task.depends_on if d not in ids]
        if missing:
            raise MissionError(f"task depends on unknown task(s): {', '.join(missing)}")
        unknown_ac = [c for c in task.criteria_ids if c not in {a.id for a in self.acceptance_criteria}]
        if unknown_ac:
            raise MissionError(f"task references unknown criteria: {', '.join(unknown_ac)}")
        # New tasks can only depend on existing ones, so adding one cannot create a cycle.
        self.tasks.append(task)
        self.touch()
        return task

    def set_dependencies(self, task_id: str, depends_on: list[str]) -> None:
        self._ensure_active()
        task = self.task(task_id)
        ids = {t.id for t in self.tasks}
        missing = [d for d in depends_on if d not in ids]
        if missing:
            raise MissionError(f"unknown task(s): {', '.join(missing)}")
        previous = task.depends_on
        task.depends_on = list(dict.fromkeys(depends_on))
        if self._has_cycle():
            task.depends_on = previous
            raise MissionError(f"dependencies of {task_id} would create a cycle")
        self.touch()

    def _has_cycle(self) -> bool:
        graph = {t.id: t.depends_on for t in self.tasks}
        state: dict[str, int] = {}  # 1 = visiting, 2 = done

        def visit(node: str) -> bool:
            if state.get(node) == 1:
                return True
            if state.get(node) == 2:
                return False
            state[node] = 1
            if any(visit(dep) for dep in graph.get(node, [])):
                return True
            state[node] = 2
            return False

        return any(visit(node) for node in graph)

    def ready_tasks(self) -> list[Task]:
        """Pending tasks whose dependencies are all done — what can start now (possibly in parallel)."""
        done = {t.id for t in self.tasks if t.status == TaskStatus.DONE}
        return [t for t in self.tasks if t.status == TaskStatus.PENDING and all(d in done for d in t.depends_on)]

    def assign(self, task_id: str, agent: str) -> Task:
        self._ensure_active()
        task = self.task(task_id)
        task.assignee = agent
        task.updated_at = now_ms()
        self.touch()
        return task

    def set_task_details(
        self, task_id: str, *, description: str | None = None, max_attempts: int | None = None
    ) -> Task:
        self._ensure_active()
        task = self.task(task_id)
        if description is not None:
            task.description = description
        if max_attempts is not None:
            task.max_attempts = max_attempts
        task.updated_at = now_ms()
        self.touch()
        return task

    def task_by_origin(self, origin: str) -> Task | None:
        return next((t for t in self.tasks if t.origin == origin), None)

    def update_task(self, task_id: str, status: TaskStatus, *, summary: str | None = None) -> Task:
        self._ensure_active()
        task = self.task(task_id)
        if status == task.status:
            return task
        if status not in _TASK_TRANSITIONS[task.status]:
            raise MissionError(f"task {task_id} cannot go from {task.status} to {status}")
        if status == TaskStatus.IN_PROGRESS:
            done = {t.id for t in self.tasks if t.status == TaskStatus.DONE}
            waiting = [d for d in task.depends_on if d not in done]
            if waiting:
                raise MissionError(f"task {task_id} is waiting on: {', '.join(waiting)}")
            if task.attempts >= task.max_attempts:
                raise MissionError(f"task {task_id} has used all {task.max_attempts} attempts; re-plan or escalate")
            task.attempts += 1
        # Reopening a done task is how new evidence sends work back; its done dependants must rerun too.
        if task.status == TaskStatus.DONE and status == TaskStatus.PENDING:
            for dependant in self.tasks:
                if task_id in dependant.depends_on and dependant.status == TaskStatus.DONE:
                    dependant.status = TaskStatus.PENDING
                    dependant.updated_at = now_ms()
        task.status = status
        if summary is not None:
            task.result_summary = summary
        task.updated_at = now_ms()
        self.touch()
        return task

    # ── findings, decisions, issues ──────────────────────────
    def add_finding(self, finding: Finding) -> Finding:
        self._ensure_active()
        if finding.task_id:
            self.task(finding.task_id)
        self.findings.append(finding)
        self.touch()
        return finding

    def set_finding_status(self, finding_id: str, status: FindingStatus, *, by: str, resolution: str | None = None) -> Finding:
        self._ensure_active()
        finding = self.finding(finding_id)
        finding.status = status
        finding.resolution = resolution
        finding.resolved_by = by
        self.touch()
        return finding

    def record_decision(self, decision: Decision) -> Decision:
        if decision.task_id:
            self.task(decision.task_id)
        self.decisions.append(decision)
        self.touch()
        return decision

    def raise_issue(self, issue: OpenIssue) -> OpenIssue:
        self._ensure_active()
        self.issues.append(issue)
        self.touch()
        return issue

    def resolve_issue(self, issue_id: str, *, by: str, resolution: str) -> OpenIssue:
        issue = self.issue(issue_id)
        if issue.status == IssueStatus.RESOLVED:
            raise MissionError(f"issue {issue_id} is already resolved")
        issue.status = IssueStatus.RESOLVED
        issue.resolution = resolution
        issue.resolved_by = by
        self.touch()
        return issue

    @property
    def open_issues(self) -> list[OpenIssue]:
        return [i for i in self.issues if i.status == IssueStatus.OPEN]

    @property
    def open_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.status in (FindingStatus.OPEN, FindingStatus.DISPUTED)]

    # ── verification ─────────────────────────────────────────
    def set_criterion(
        self, criterion_id: str, status: CriterionStatus, *, by: str, evidence_ids: list[str], note: str | None = None
    ) -> AcceptanceCriterion:
        self._ensure_active()
        criterion = self.criterion(criterion_id)
        if status == CriterionStatus.MET and not evidence_ids:
            raise MissionError("a criterion can only be marked met with evidence (a test run, diff or scan id)")
        criterion.status = status
        criterion.evidence_ids = list(dict.fromkeys(evidence_ids))
        criterion.verified_by = by
        criterion.note = note
        criterion.updated_at = now_ms()
        self.touch()
        return criterion

    @property
    def all_criteria_met(self) -> bool:
        return bool(self.acceptance_criteria) and all(c.status == CriterionStatus.MET for c in self.acceptance_criteria)

    def progress(self) -> dict[str, Any]:
        tasks_by_status: dict[str, int] = {}
        for task in self.tasks:
            tasks_by_status[task.status] = tasks_by_status.get(task.status, 0) + 1
        criteria_by_status: dict[str, int] = {}
        for criterion in self.acceptance_criteria:
            criteria_by_status[criterion.status] = criteria_by_status.get(criterion.status, 0) + 1
        return {
            "tasks": tasks_by_status,
            "criteria": criteria_by_status,
            "open_findings": len(self.open_findings),
            "open_issues": len(self.open_issues),
        }


# ── evidence & messages (stored separately) ─────────────────────────────────
class EvidenceKind(StrEnum):
    TEST_RUN = "test_run"
    DIFF = "diff"
    SECRET_SCAN = "secret_scan"
    STATIC_SCAN = "static_scan"
    COMMAND = "command"
    REVIEW = "review"


class Evidence(BaseModel):
    """Something that actually happened, captured from a real tool run."""

    evidence_id: str = Field(default_factory=lambda: new_id("evd"))
    mission_id: str
    kind: str
    agent: str
    tool: str
    task_id: str | None = None
    # True / False for checks with a verdict (tests, scans); None for neutral records (diffs).
    passed: bool | None = None
    summary: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    at: int = Field(default_factory=now_ms)


class MessageKind(StrEnum):
    FINDING = "finding"       # "I found X" — evidence for others to use
    HANDOFF = "handoff"       # "my part is done; here is what you need"
    CRITIQUE = "critique"     # reviewer → developer
    QUESTION = "question"
    ANSWER = "answer"
    REQUEST = "request"       # ask another agent to do something
    INFO = "info"


BROADCAST = "*"


class AgentMessage(BaseModel):
    """Agent-to-agent communication. `to` is an agent name, a team ("@forge"), or "*" for everyone."""

    message_id: str = Field(default_factory=lambda: new_id("msg"))
    mission_id: str
    seq: int | None = None
    sender: str
    to: str = BROADCAST
    kind: MessageKind = MessageKind.INFO
    subject: str
    body: str = ""
    task_id: str | None = None
    finding_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    in_reply_to: str | None = None
    at: int = Field(default_factory=now_ms)

    def addressed_to(self, agent: str, teams: set[str] | None = None) -> bool:
        if self.to == BROADCAST or self.to.lower() == agent.lower():
            return True
        return self.to.startswith("@") and self.to[1:].lower() in {t.lower() for t in teams or set()}
