"""
Board tools: how agents read the shared mission state and talk to each other.

They are ordinary registry tools, so every message, finding and verification
goes through the Governor — granted per agent, budgeted, and in the audit
trail, the same as file edits and test runs. That is what makes the agents'
coordination inspectable rather than hidden inside prompts.
"""
from __future__ import annotations

from app.core.errors import DevPilotError
from app.domain.mission import (
    AgentMessage,
    CriterionStatus,
    Finding,
    FindingKind,
    FindingStatus,
    IssueKind,
    MessageKind,
    OpenIssue,
    Severity,
)
from app.tools.registry import ToolContext, ToolRegistry, ToolResult

BOARD_TOOLS = ("mission_state", "read_inbox", "send_message", "record_finding", "update_finding",
               "raise_issue", "list_evidence", "verify_criterion")


def _board(ctx: ToolContext):
    if ctx.board is None:
        raise RuntimeError("board tools need a mission board in the tool context")
    return ctx.board


def _err(exc: DevPilotError) -> ToolResult:
    return ToolResult(False, error=exc.message)


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "mission_state",
        "Read the current shared mission state: criteria, plan, findings, open issues and recent decisions.",
        {"type": "object", "properties": {}},
        category="board")
    async def mission_state(ctx: ToolContext) -> ToolResult:
        mission = await _board(ctx).get(ctx.mission_id)
        return ToolResult(True, display="read mission state", content={
            "status": mission.status,
            "acceptance_criteria": [{"id": c.id, "text": c.text, "status": c.status, "evidence_ids": c.evidence_ids}
                                    for c in mission.acceptance_criteria],
            "tasks": [{"id": t.id, "title": t.title, "kind": t.kind, "status": t.status, "assignee": t.assignee,
                       "depends_on": t.depends_on, "result": t.result_summary} for t in mission.tasks],
            "open_findings": [{"id": f.id, "agent": f.agent, "kind": f.kind, "severity": f.severity, "title": f.title,
                               "detail": f.detail[:600], "file": f.file, "line": f.line, "status": f.status}
                              for f in mission.open_findings],
            "open_issues": [{"id": i.id, "kind": i.kind, "text": i.text, "by": i.raised_by} for i in mission.open_issues],
            "recent_decisions": [{"by": d.by, "decision": d.decision, "reason": d.reason} for d in mission.decisions[-8:]],
        })

    @registry.tool(
        "read_inbox",
        "Read new messages sent to you, your team, or everyone since you last checked.",
        {"type": "object", "properties": {}},
        category="board")
    async def read_inbox(ctx: ToolContext) -> ToolResult:
        messages = await _board(ctx).inbox(ctx.mission_id, ctx.agent, teams=ctx.teams, after_seq=ctx.inbox_seq)
        if messages:
            ctx.inbox_seq = max(m.seq or 0 for m in messages)
        return ToolResult(True, display=f"{len(messages)} new message(s)", content={"messages": [
            {"seq": m.seq, "from": m.sender, "to": m.to, "kind": m.kind, "subject": m.subject, "body": m.body,
             "task_id": m.task_id, "evidence_ids": m.evidence_ids, "finding_ids": m.finding_ids,
             "message_id": m.message_id} for m in messages]})

    @registry.tool(
        "send_message",
        "Send a message to another agent (by name), a team ('@planning', '@forge', '@sentinel', '@shadow'), "
        "or everyone ('*'). Use it to hand off work, report a finding, critique a change, or ask a question. "
        "Cite evidence_ids and finding_ids so the receiver can check your claims.",
        {"type": "object", "properties": {
            "to": {"type": "string", "description": "Agent name, '@team', or '*'"},
            "kind": {"type": "string", "enum": [k.value for k in MessageKind]},
            "subject": {"type": "string"},
            "body": {"type": "string"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "finding_ids": {"type": "array", "items": {"type": "string"}},
            "in_reply_to": {"type": "string", "description": "message_id you are answering"}},
         "required": ["to", "kind", "subject", "body"]},
        category="board")
    async def send_message(ctx: ToolContext, to: str, kind: str, subject: str, body: str,
                           evidence_ids: list[str] | None = None, finding_ids: list[str] | None = None,
                           in_reply_to: str | None = None) -> ToolResult:
        if to.strip().lower() == ctx.agent.lower():
            return ToolResult(False, error="You cannot send a message to yourself.")
        message = await _board(ctx).post_message(AgentMessage(
            mission_id=ctx.mission_id, sender=ctx.agent, to=to.strip(), kind=MessageKind(kind), subject=subject,
            body=body, task_id=ctx.task_id, evidence_ids=evidence_ids or [], finding_ids=finding_ids or [],
            in_reply_to=in_reply_to))
        return ToolResult(True, display=f"message to {message.to}: {subject[:60]}",
                          content={"message_id": message.message_id, "seq": message.seq})

    @registry.tool(
        "record_finding",
        "Record a finding on the shared board: a fact about the codebase (recon), a risk, a bug, a test failure, "
        "a review or security issue. Others will see it and may act on it.",
        {"type": "object", "properties": {
            "kind": {"type": "string", "enum": [k.value for k in FindingKind]},
            "severity": {"type": "string", "enum": [s.value for s in Severity]},
            "title": {"type": "string"},
            "detail": {"type": "string"},
            "file": {"type": "string"},
            "line": {"type": "integer"},
            "evidence_ids": {"type": "array", "items": {"type": "string"}}},
         "required": ["kind", "severity", "title"]},
        category="board")
    async def record_finding(ctx: ToolContext, kind: str, severity: str, title: str, detail: str = "",
                             file: str | None = None, line: int | None = None,
                             evidence_ids: list[str] | None = None) -> ToolResult:
        try:
            finding = await _board(ctx).add_finding(ctx.mission_id, Finding(
                agent=ctx.agent, kind=FindingKind(kind), severity=Severity(severity), title=title, detail=detail,
                file=file, line=line, task_id=ctx.task_id, evidence_ids=evidence_ids or []))
        except DevPilotError as exc:
            return _err(exc)
        return ToolResult(True, display=f"finding [{severity}] {title[:60]}", content={"finding_id": finding.id})

    @registry.tool(
        "update_finding",
        "Mark a finding as addressed (you fixed it), disputed (you disagree — explain why with evidence), "
        "or dismissed (it was wrong).",
        {"type": "object", "properties": {
            "finding_id": {"type": "string"},
            "status": {"type": "string", "enum": [FindingStatus.ADDRESSED.value, FindingStatus.DISPUTED.value,
                                                  FindingStatus.DISMISSED.value]},
            "resolution": {"type": "string", "description": "What you did, or why you disagree"}},
         "required": ["finding_id", "status", "resolution"]},
        category="board")
    async def update_finding(ctx: ToolContext, finding_id: str, status: str, resolution: str) -> ToolResult:
        try:
            finding = await _board(ctx).set_finding_status(ctx.mission_id, finding_id, FindingStatus(status),
                                                          by=ctx.agent, resolution=resolution)
        except DevPilotError as exc:
            return _err(exc)
        return ToolResult(True, display=f"finding {finding.id} → {status}", content={"finding_id": finding.id,
                                                                                    "status": finding.status})

    @registry.tool(
        "raise_issue",
        "Raise an issue you cannot resolve yourself: a question for another agent, a blocker, a dispute, "
        "or a clarification only the human who made the request can answer (needs_human=true).",
        {"type": "object", "properties": {
            "kind": {"type": "string", "enum": [k.value for k in IssueKind]},
            "text": {"type": "string"},
            "needs_human": {"type": "boolean"}},
         "required": ["kind", "text"]},
        category="board")
    async def raise_issue(ctx: ToolContext, kind: str, text: str, needs_human: bool = False) -> ToolResult:
        issue_kind = IssueKind(kind)
        try:
            issue = await _board(ctx).raise_issue(ctx.mission_id, OpenIssue(
                raised_by=ctx.agent, kind=issue_kind, text=text, task_id=ctx.task_id,
                needs_human=needs_human or issue_kind == IssueKind.CLARIFICATION))
        except DevPilotError as exc:
            return _err(exc)
        return ToolResult(True, display=f"issue [{kind}] {text[:60]}", content={"issue_id": issue.id})

    @registry.tool(
        "list_evidence",
        "List evidence recorded in this mission (test runs, diffs, scans) with their verdicts and ids.",
        {"type": "object", "properties": {
            "kind": {"type": "string", "description": "test_run, diff, secret_scan, static_scan"}}},
        category="board")
    async def list_evidence(ctx: ToolContext, kind: str | None = None) -> ToolResult:
        items = await _board(ctx).evidence(ctx.mission_id, kind=kind or None)
        return ToolResult(True, display=f"{len(items)} evidence item(s)", content={"evidence": [
            {"evidence_id": e.evidence_id, "kind": e.kind, "agent": e.agent, "passed": e.passed,
             "summary": e.summary, "task_id": e.task_id} for e in items[-40:]]})

    @registry.tool(
        "verify_criterion",
        "Record your independent verdict on an acceptance criterion. 'met' requires evidence_ids of real, "
        "passing evidence (e.g. a test run that covers it); failing or unknown evidence is rejected.",
        {"type": "object", "properties": {
            "criterion_id": {"type": "string"},
            "status": {"type": "string", "enum": [CriterionStatus.MET.value, CriterionStatus.UNMET.value,
                                                  CriterionStatus.UNVERIFIABLE.value]},
            "evidence_ids": {"type": "array", "items": {"type": "string"}},
            "note": {"type": "string", "description": "How the evidence shows the criterion is (not) met"}},
         "required": ["criterion_id", "status", "evidence_ids", "note"]},
        category="board")
    async def verify_criterion(ctx: ToolContext, criterion_id: str, status: str, evidence_ids: list[str],
                               note: str) -> ToolResult:
        try:
            criterion = await _board(ctx).verify_criterion(ctx.mission_id, criterion_id, CriterionStatus(status),
                                                           by=ctx.agent, evidence_ids=evidence_ids, note=note)
        except DevPilotError as exc:
            return _err(exc)
        return ToolResult(True, display=f"criterion {criterion_id} → {status}",
                          content={"criterion_id": criterion.id, "status": criterion.status})
