"""
Turning agent output into mission state, and the guarantees Atlas adds on top.

- Kenji's and Ryo's results are parsed leniently (models vary in shape) but
  validated strictly before they touch the board.
- If the planners fail, deterministic fallbacks keep the mission moving.
- `enforce_gates` makes the PS-001 workflow non-negotiable: whatever the plan
  says, every change is tested, reviewed, security-inspected and
  independently verified before the mission can complete.
"""
from __future__ import annotations

import re
from typing import Any

from app.domain.mission import Mission, Task, TaskKind

WORK_KINDS = {TaskKind.IMPLEMENT, TaskKind.FIX}
_KIND_ALIASES = {
    "implementation": TaskKind.IMPLEMENT, "feature": TaskKind.IMPLEMENT, "build": TaskKind.IMPLEMENT,
    "bugfix": TaskKind.FIX, "bug": TaskKind.FIX, "debug": TaskKind.FIX, "repair": TaskKind.FIX,
    "testing": TaskKind.TEST, "tests": TaskKind.TEST, "qa": TaskKind.TEST,
    "code_review": TaskKind.REVIEW, "code-review": TaskKind.REVIEW,
    "security_review": TaskKind.SECURITY, "audit": TaskKind.SECURITY,
    "verification": TaskKind.VERIFY, "validate": TaskKind.VERIFY,
    "documentation": TaskKind.REPORT, "docs": TaskKind.REPORT,
}


def _text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for key in ("text", "criterion", "description", "objective", "title"):
            if isinstance(item.get(key), str):
                return item[key].strip()
    return str(item).strip() if item is not None else ""


def parse_requirements(data: dict[str, Any]) -> tuple[list[str], list[str]] | None:
    objectives = [t for t in (_text(i) for i in data.get("objectives") or []) if t]
    criteria = [t for t in (_text(i) for i in data.get("acceptance_criteria") or []) if t]
    if not criteria:
        return None
    return objectives or criteria[:1], criteria[:12]


def fallback_requirements(mission: Mission) -> tuple[list[str], list[str]]:
    return ([mission.title], [f"The request is implemented as asked: {mission.request[:300]}",
                              "The project's existing tests still pass"])


def _kind(value: Any) -> TaskKind | None:
    key = str(value or "").strip().lower()
    try:
        return TaskKind(key)
    except ValueError:
        return _KIND_ALIASES.get(key)


def parse_plan(data: dict[str, Any], mission: Mission) -> list[Task]:
    """Ryo's `tasks` → Task objects with dependencies resolved by title, index or 'T<n>'."""
    raw = data.get("tasks") or []
    if not isinstance(raw, list):
        return []
    criteria = {c.id for c in mission.acceptance_criteria}
    tasks: list[Task] = []
    by_title: dict[str, str] = {}
    for item in raw[:12]:
        if not isinstance(item, dict):
            continue
        title = _text(item.get("title") or item.get("name"))
        kind = _kind(item.get("kind") or item.get("type"))
        if not title or kind is None or kind in (TaskKind.ANALYSIS, TaskKind.PLAN):
            continue
        task = Task(
            title=title[:160], kind=kind, description=_text(item.get("description"))[:2000],
            required_skills=[s for s in (item.get("required_skills") or item.get("skills") or []) if isinstance(s, str)][:6],
            criteria_ids=[c for c in (item.get("criteria") or item.get("criteria_ids") or []) if c in criteria],
            created_by="Ryo",
        )
        deps = item.get("depends_on") or []
        resolved: list[str] = []
        for dep in deps if isinstance(deps, list) else [deps]:
            ref = str(dep).strip()
            index = re.fullmatch(r"[Tt]?(\d+)", ref)
            if index and 0 < int(index.group(1)) <= len(tasks):
                resolved.append(tasks[int(index.group(1)) - 1].id)
            elif ref.lower() in by_title:
                resolved.append(by_title[ref.lower()])
        task.depends_on = list(dict.fromkeys(resolved))
        tasks.append(task)
        by_title[title.lower()] = task.id
    return tasks


def fallback_plan(mission: Mission) -> list[Task]:
    request = mission.request.lower()
    kind = TaskKind.FIX if re.search(r"\b(fix|bug|fail|broken|error|wrong|vulnerab)", request) else TaskKind.IMPLEMENT
    return [Task(title=("Fix: " if kind == TaskKind.FIX else "Implement: ") + mission.title, kind=kind,
                 description=mission.request, criteria_ids=[c.id for c in mission.acceptance_criteria],
                 created_by="Atlas")]


def enforce_gates(tasks: list[Task], mission: Mission) -> list[Task]:
    """Add the checks PS-001 requires that the plan left out, and order verification after them."""
    tasks = list(tasks)
    all_criteria = [c.id for c in mission.acceptance_criteria]
    work = [t for t in tasks if t.kind in WORK_KINDS]
    if not work:
        work = fallback_plan(mission)
        tasks = work + tasks
    work_ids = [t.id for t in work]

    def ensure(kind: TaskKind, title: str, description: str, origin: str) -> list[Task]:
        existing = [t for t in tasks if t.kind == kind]
        if existing:
            return existing
        gate = Task(title=title, kind=kind, description=description, depends_on=work_ids,
                    criteria_ids=all_criteria, origin=origin, created_by="Atlas")
        tasks.append(gate)
        return [gate]

    tests = ensure(TaskKind.TEST, "Run the test suite",
                   "Run the project's tests (and lint/build if present) against the change and report what the "
                   "run shows.", "gate:test")
    reviews = ensure(TaskKind.REVIEW, "Review the change",
                     "Review the diff for correctness, readability, consistency with the codebase and regressions.",
                     "gate:review")
    security = ensure(TaskKind.SECURITY, "Security inspection",
                      "Check the change for exposed secrets, unsafe operations and injection risks.", "gate:security")
    checks = [t.id for t in tests + reviews + security]
    verify = [t for t in tasks if t.kind == TaskKind.VERIFY]
    if not verify:
        verify = [Task(title="Verify the acceptance criteria", kind=TaskKind.VERIFY,
                       description="Independently check every acceptance criterion against evidence.",
                       depends_on=checks, criteria_ids=all_criteria, origin="gate:verify", created_by="Atlas")]
        tasks.append(verify[0])
    else:
        for task in verify:
            task.depends_on = list(dict.fromkeys(task.depends_on + checks))
            task.criteria_ids = all_criteria  # independent verification always covers every criterion
    for task in tasks:
        if task.kind == TaskKind.REPORT:
            task.depends_on = list(dict.fromkeys(task.depends_on + [v.id for v in verify]))
    return tasks
