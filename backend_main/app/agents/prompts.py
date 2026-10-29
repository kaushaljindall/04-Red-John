"""
Shared prompt text and the briefing every agent gets at the start of a run.
Role-specific instructions live in each agent's definition (Phase 6).
"""
from __future__ import annotations

from app.domain.agents import AgentDefinition
from app.domain.mission import AgentMessage, Finding, Mission, Task

GUILD_RULES = """\
You are {name}, {role} in DevPilot — an AI engineering guild that turns a request into a working, tested,
reviewed change to a small repository. You work alongside other agents; Atlas orchestrates, Maya hires.

How you work:
- You act only through your tools. You cannot see or change anything any other way.
- Paths are relative to the repository root. You work on an isolated copy; nothing reaches the real repository
  until a human approves it.
- Follow the project's existing style, structure and conventions. Read before you write. Prefer small,
  focused edits (edit_file) over rewriting whole files.
- Never claim something works without evidence. Tests passing means a run_tests result says so; cite its
  evidence_id. If you could not verify something, say so plainly.
- If a tool fails, read the error and change your approach. Do not repeat a call that already failed.
- Share what others need: send_message to hand off work, ask questions or critique; record_finding for facts,
  risks and problems; raise_issue when you are blocked or need a human.
- Blocked or refused actions are final for this run: work around them or report them.
- Finish by calling `finish` exactly once with status done / failed / blocked and a short, factual summary
  of what you did and what you verified. Put structured results in finish's `data`.
- Only the tools listed for you exist. There is no `json`, `commentary`, `python`, `browser` or
  `repo_browser` tool: never call them, and never prefix a tool name.

Your role:
{role_prompt}
"""


def system_prompt(definition: AgentDefinition, mandate: str | None = None) -> str:
    prompt = GUILD_RULES.format(name=definition.name, role=definition.role, role_prompt=definition.system_prompt.strip())
    if mandate:
        prompt += f"\nYour mandate for this mission (from Maya): {mandate.strip()}\n"
    if definition.output_schema:
        required = ", ".join(definition.output_schema.get("required", []))
        prompt += f"\nWhen you finish, `data` must include: {required}.\n"
        for name, meaning in (definition.output_schema.get("properties") or {}).items():
            prompt += f"  - {name}: {meaning}\n"
    return prompt


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _task_line(task: Task) -> str:
    who = f" @{task.assignee}" if task.assignee else ""
    deps = f" after {', '.join(task.depends_on)}" if task.depends_on else ""
    return f"- {task.id} [{task.status}] ({task.kind}) {task.title}{who}{deps}"


def _finding_line(finding: Finding) -> str:
    where = f" {finding.file}:{finding.line}" if finding.file else ""
    return f"- {finding.id} [{finding.severity}/{finding.kind}] {finding.title}{where} — by {finding.agent}"


def _message_block(message: AgentMessage) -> str:
    refs = []
    if message.evidence_ids:
        refs.append("evidence " + ", ".join(message.evidence_ids))
    if message.finding_ids:
        refs.append("findings " + ", ".join(message.finding_ids))
    ref = f" ({'; '.join(refs)})" if refs else ""
    return f"- #{message.seq} {message.kind} from {message.sender} to {message.to}: {message.subject}{ref}\n  {_clip(message.body, 1200)}"


def briefing(
    mission: Mission,
    task: Task | None,
    inbox: list[AgentMessage],
    instructions: str | None = None,
) -> str:
    parts = [f"# Mission {mission.mission_id}: {mission.title}", f"Request:\n{mission.request}"]
    if mission.objectives:
        parts.append("Objectives:\n" + "\n".join(f"- {o.text}" for o in mission.objectives))
    if mission.acceptance_criteria:
        parts.append("Acceptance criteria:\n" + "\n".join(
            f"- {c.id} [{c.status}] {c.text}" for c in mission.acceptance_criteria))
    if task:
        lines = [f"# Your task: {task.id} — {task.title} ({task.kind})"]
        if task.description:
            lines.append(task.description)
        if task.criteria_ids:
            lines.append("Covers criteria: " + ", ".join(task.criteria_ids))
        if task.attempts > 1:
            lines.append(f"This is attempt {task.attempts} of {task.max_attempts}.")
            if task.result_summary:
                lines.append(f"Previous attempt ended with: {_clip(task.result_summary, 800)}")
        parts.append("\n".join(lines))
    if instructions:
        parts.append(f"# Instructions from Atlas\n{instructions.strip()}")
    if mission.tasks:
        parts.append("# Plan\n" + "\n".join(_task_line(t) for t in mission.tasks[-30:]))
    findings = mission.open_findings[-12:]
    if findings:
        parts.append("# Open findings\n" + "\n".join(_finding_line(f) for f in findings))
    if mission.open_issues:
        parts.append("# Open issues\n" + "\n".join(
            f"- {i.id} [{i.kind}] {i.text} — by {i.raised_by}" for i in mission.open_issues[-8:]))
    resolved = [i for i in mission.issues if i.resolution][-5:]
    if resolved:
        parts.append("# Answers to earlier questions\n" + "\n".join(
            f"- {i.text} → {i.resolution} ({i.resolved_by})" for i in resolved))
    if inbox:
        parts.append("# Messages for you\n" + "\n".join(_message_block(m) for m in inbox[-15:]))
    parts.append("Start by deciding what you need to know, then use your tools. Call `finish` when you are done.")
    return "\n\n".join(parts)
