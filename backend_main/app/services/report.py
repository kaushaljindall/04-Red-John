"""
The mission's evidence report, built from recorded facts — not from any
agent's summary. PS-001: "record what changed, which checks passed or
failed, and what remains unresolved … clearly separate verified outcomes
from assumptions and model claims."
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from app.domain.governance import AuditEntry
from app.domain.mission import CriterionStatus, Evidence, FindingStatus, Mission, TaskStatus


def _when(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _cell(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_report(mission: Mission, evidence: list[Evidence], audit: list[AuditEntry]) -> str:
    by_id = {e.evidence_id: e for e in evidence}
    met = [c for c in mission.acceptance_criteria if c.status == CriterionStatus.MET]
    lines = [
        f"# DevPilot mission report — {mission.title}",
        "",
        f"- **Mission:** `{mission.mission_id}` · **Status:** {mission.status}"
        + (f" ({mission.status_reason})" if mission.status_reason else ""),
        f"- **Repository:** `{mission.repo_path}` · baseline `{(mission.baseline_commit or '')[:10]}`",
        f"- **Started:** {_when(mission.created_at)} · **Last change:** {_when(mission.updated_at)}",
        f"- **Acceptance criteria verified:** {len(met)} of {len(mission.acceptance_criteria)}",
        "",
        "## Request",
        "",
        mission.request,
        "",
        "## Acceptance criteria",
        "",
        "| Criterion | Status | Evidence | Verified by |",
        "|---|---|---|---|",
    ]
    for criterion in mission.acceptance_criteria:
        cited = [by_id[i] for i in criterion.evidence_ids if i in by_id]
        proof = "; ".join(f"`{e.evidence_id}` {_cell(e.summary, 90)}" for e in cited) or "—"
        status = "✅ met" if criterion.status == CriterionStatus.MET else f"❌ {criterion.status}"
        lines.append(f"| {_cell(criterion.text)} | {status} | {proof} | {criterion.verified_by or '—'} |")

    diffs = [e for e in evidence if e.kind == "diff"]
    lines += ["", "## What changed", ""]
    if diffs:
        latest = diffs[-1]
        lines += [f"From `{latest.evidence_id}` ({latest.summary}):", "", "```", str(latest.data.get("stat", "")).strip(), "```"]
    else:
        lines.append("No diff was recorded.")

    checks = [e for e in evidence if e.passed is not None]
    lines += ["", "## Checks that ran", "", "| Evidence | Check | Result | By |", "|---|---|---|---|"]
    for e in checks:
        lines.append(f"| `{e.evidence_id}` | {_cell(e.summary, 120)} | {'pass' if e.passed else 'FAIL'} | {e.agent} |")
    if not checks:
        lines.append("| — | no checks recorded | — | — |")

    review = [f for f in mission.findings if f.kind in ("review", "security", "test_failure", "bug")]
    if review:
        lines += ["", "## Findings", "", "| Finding | Severity | By | Status | Resolution |", "|---|---|---|---|---|"]
        for f in review:
            where = f" ({f.file}:{f.line})" if f.file else ""
            lines.append(f"| {_cell(f.title + where, 120)} | {f.severity} | {f.agent} | {f.status} | "
                         f"{_cell(f.resolution or '—', 100)} |")

    unresolved: list[str] = []
    unresolved += [f"Criterion not verified: {c.text} ({c.status})" for c in mission.acceptance_criteria
                   if c.status != CriterionStatus.MET]
    still_open = [f for f in mission.findings if f.status in (FindingStatus.OPEN, FindingStatus.DISPUTED)
                  and f.kind in ("review", "security", "bug", "test_failure")]
    blocking = [f for f in still_open if f.kind in ("review", "security") and f.severity in ("medium", "high", "critical")]
    unresolved += [f"Open {f.severity} {f.kind} finding: {f.title}" for f in blocking]
    unresolved += [f"Open issue ({i.kind}): {i.text}" for i in mission.open_issues]
    unresolved += [f"Task {t.status}: {t.title}" for t in mission.tasks if t.status in (TaskStatus.FAILED, TaskStatus.BLOCKED)]
    lines += ["", "## Unresolved", ""]
    lines += [f"- {u}" for u in unresolved] or ["- Nothing unresolved."]
    informational = [f for f in still_open if f not in blocking and f.severity != "info"]
    if informational:
        lines += ["", "Findings still marked open but not blocking (not re-checked after the change; the criteria "
                      "above are the verified outcome): " + "; ".join(f"{f.title} ({f.severity}, {f.agent})"
                                                                   for f in informational) + "."]

    lines += ["", "## Decisions", ""]
    lines += [f"- `{d.decision}` by {d.by}: {_cell(d.reason, 220)}" for d in mission.decisions[-40:]] or ["- None recorded."]

    lines += ["", "## Task force", "", "| Agent | Role | Hired for | Mandate |", "|---|---|---|---|"]
    for hire in mission.hires:
        lines.append(f"| {hire.agent} | {hire.role} | {_cell(hire.reason, 80)} | {_cell(hire.mandate, 120)} |")

    outcomes = Counter(str(a.outcome) for a in audit)
    agents = Counter(a.agent for a in audit)
    lines += ["", "## Tool use", "",
              f"{len(audit)} governed tool call(s): " + ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items())) + ".",
              "By agent: " + ", ".join(f"{k} {v}" for k, v in agents.most_common()) + "." if agents else ""]

    lines += ["", "---", "",
              "_Verified_ lines above cite evidence captured from real tool runs in the mission workspace "
              "(test output, diffs, scans). Agent summaries are claims and are not used as evidence."]
    return "\n".join(lines).rstrip() + "\n"
