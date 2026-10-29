"""
Snapshot reducer: folds a mission's event stream (plus the current mission
document) into the state the control room renders on load. Ported from
Mycel's `/realtime/snapshot`: the Mycel keys (clock, logs, hires, agents,
complete, pendingApprovals) keep their shape so the existing UI works, and
DevPilot's mission state rides alongside.
"""
from __future__ import annotations

from typing import Any

from app.core.ids import now_ms
from app.domain.events import EventEnvelope, EventKind
from app.domain.governance import Approval
from app.domain.mission import Evidence, Mission, MissionStatus

_ALWAYS_ON = ("Maya", "Atlas")


def build_snapshot(
    mission: Mission,
    events: list[EventEnvelope],
    pending_approvals: list[Approval],
    evidence: list[Evidence],
) -> dict[str, Any]:
    start = mission.created_at
    logs: list[dict[str, Any]] = []
    hires: list[dict[str, Any]] = []
    agents: dict[str, dict[str, Any]] = {
        name: {"name": name, "phase": "working", "task": "On duty", "startedAt": 0, "finishedAt": None}
        for name in _ALWAYS_ON
    }
    tool_calls = 0
    last_seq = 0

    for event in events:
        at = event.at - start
        data = event.data
        last_seq = max(last_seq, event.seq or 0)
        if event.kind == EventKind.LOG:
            logs.append({"id": len(logs), "at": at, "level": data.get("level", "info"), "text": data.get("text", "")})
        elif event.kind == EventKind.HIRE:
            hires.append({
                "id": len(hires), "at": at, "agent": data.get("agent"), "team": data.get("team"),
                "role": data.get("role"), "badge": data.get("badge", ""),
                "clearance": data.get("clearance", "GREEN"), "mandate": data.get("mandate", ""),
            })
            agents[data["agent"]] = {"name": data["agent"], "phase": "hired", "task": "Awaiting assignment",
                                     "startedAt": None, "finishedAt": None}
        elif event.kind == EventKind.RELEASE and data.get("agent") in agents:
            agents[data["agent"]].update({"phase": "done", "task": "Released", "finishedAt": at})
        elif event.kind == EventKind.START:
            agent = agents.setdefault(data["agent"], {"name": data["agent"]})
            agent.update({"phase": "working", "task": data.get("task", ""), "startedAt": at, "finishedAt": None})
        elif event.kind == EventKind.FINISH and data.get("agent") in agents:
            agents[data["agent"]].update({"phase": "done", "finishedAt": at})
        elif event.kind == EventKind.TOOL_CALL:
            tool_calls += 1

    complete = mission.status == MissionStatus.COMPLETED
    if mission.is_terminal:
        for name in _ALWAYS_ON:
            agents[name].update({"phase": "done", "finishedAt": mission.updated_at - start})

    return {
        # ── Mycel control-room shape ──
        "clock": (mission.updated_at if mission.is_terminal else now_ms()) - start,
        "logs": logs,
        "hires": hires,
        "agents": agents,
        "complete": complete,
        "pendingApprovals": [a.request_payload() for a in pending_approvals],
        # ── DevPilot ──
        "lastSeq": last_seq,
        "mission": {
            "mission_id": mission.mission_id,
            "title": mission.title,
            "request": mission.request,
            "status": mission.status,
            "status_reason": mission.status_reason,
            "baseline_commit": mission.baseline_commit,
            "created_at": mission.created_at,
            "updated_at": mission.updated_at,
            "hired_agents": mission.hired_agents,
        },
        "progress": {**mission.progress(), "tool_calls": tool_calls},
        "objectives": [o.model_dump() for o in mission.objectives],
        "acceptanceCriteria": [c.model_dump() for c in mission.acceptance_criteria],
        "tasks": [t.model_dump() for t in mission.tasks],
        "readyTasks": [t.id for t in mission.ready_tasks()],
        "findings": [f.model_dump() for f in mission.findings],
        "decisions": [d.model_dump() for d in mission.decisions[-50:]],
        "issues": [i.model_dump() for i in mission.issues],
        "evidence": [
            {"evidence_id": e.evidence_id, "kind": e.kind, "agent": e.agent, "passed": e.passed,
             "summary": e.summary, "task_id": e.task_id, "at": e.at - start}
            for e in evidence
        ],
    }
