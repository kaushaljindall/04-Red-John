"""
Bounded autonomy for one mission run: budgets, loop detection and the audit
trail, all enforced in one place — the `Governor` every agent's tool calls
go through.

Loop detection is ported from Mycel `autonomy/loop_detector.py`, moved down
to the tool-call level: an agent repeating the exact call that already failed
is told to change approach instead of being allowed to spin forever.

One mission runs on one worker at a time (Phase 8 enforces this with a lease),
so the counters live in memory; every refusal and snapshot is also published
as an event and written to the audit log, so nothing is lost if the worker dies.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.domain.events import EventKind, LogLevel
from app.domain.governance import AuditEntry, AuditOutcome, MissionLimits
from app.infrastructure.storage.base import AuditLog
from app.services.event_publisher import EventPublisher
from app.services.mission_board import MissionBoard
from app.tools.registry import ToolContext, ToolRegistry, ToolResult, preview_args

log = logging.getLogger(__name__)


# ── budget ──────────────────────────────────────────────────────────────────
@dataclass
class MissionBudget:
    limits: MissionLimits
    started_at: float = field(default_factory=time.monotonic)
    tool_calls: int = 0
    llm_tokens: int = 0
    tool_calls_by_agent: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_by_agent: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def exceeded(self) -> str | None:
        """The first limit that is used up, or None."""
        if self.tool_calls >= self.limits.max_tool_calls:
            return f"tool-call limit reached ({self.tool_calls}/{self.limits.max_tool_calls})"
        if self.llm_tokens >= self.limits.max_llm_tokens:
            return f"LLM token limit reached ({self.llm_tokens:,}/{self.limits.max_llm_tokens:,})"
        if self.elapsed_s >= self.limits.max_wall_clock_s:
            return f"time limit reached ({self.elapsed_s:.0f}s/{self.limits.max_wall_clock_s}s)"
        return None

    def record_tool_call(self, agent: str) -> None:
        self.tool_calls += 1
        self.tool_calls_by_agent[agent] += 1

    def record_tokens(self, agent: str, tokens: int) -> None:
        self.llm_tokens += tokens
        self.tokens_by_agent[agent] += tokens

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.limits.max_tool_calls,
            "llm_tokens": self.llm_tokens,
            "max_llm_tokens": self.limits.max_llm_tokens,
            "elapsed_s": round(self.elapsed_s, 1),
            "max_wall_clock_s": self.limits.max_wall_clock_s,
            "tool_calls_by_agent": dict(self.tool_calls_by_agent),
            "tokens_by_agent": dict(self.tokens_by_agent),
        }


# ── loop detection ──────────────────────────────────────────────────────────
def call_signature(agent: str, tool: str, args: Any) -> str:
    try:
        canonical = json.dumps(args, sort_keys=True, default=str)
    except TypeError:
        canonical = repr(args)
    return hashlib.sha256(f"{agent}\x00{tool}\x00{canonical}".encode()).hexdigest()[:16]


@dataclass
class _FailureRecord:
    count: int = 0
    last_error: str = ""


class LoopDetector:
    def __init__(self, max_identical_failures: int, max_consecutive_failures: int) -> None:
        self._max_identical = max_identical_failures
        self._max_consecutive = max_consecutive_failures
        self._failures: dict[str, _FailureRecord] = defaultdict(_FailureRecord)
        self._consecutive: dict[str, int] = defaultdict(int)

    def check(self, agent: str, tool: str, args: Any) -> str | None:
        record = self._failures.get(call_signature(agent, tool, args))
        if record and record.count >= self._max_identical:
            return (
                f"This exact {tool} call has already failed {record.count} times "
                f"(last error: {record.last_error[:300]}). Repeating it will not help: change your approach, "
                f"gather more information, or report the blocker."
            )
        if self._consecutive[agent] >= self._max_consecutive:
            return (
                f"{agent} has had {self._consecutive[agent]} tool failures in a row. Stop and report what is blocking "
                f"you so the orchestrator can re-plan or escalate."
            )
        return None

    def record(self, agent: str, tool: str, args: Any, result: ToolResult) -> None:
        signature = call_signature(agent, tool, args)
        if result.ok:
            self._consecutive[agent] = 0
            self._failures.pop(signature, None)
            return
        if result.meta.get("outcome") in (AuditOutcome.REFUSED_LOOP, AuditOutcome.REFUSED_BUDGET):
            return  # refusals are not fresh evidence of a loop
        self._consecutive[agent] += 1
        record = self._failures[signature]
        record.count += 1
        record.last_error = result.error

    def failure_count(self, agent: str, tool: str, args: Any) -> int:
        record = self._failures.get(call_signature(agent, tool, args))
        return record.count if record else 0


# ── governor ────────────────────────────────────────────────────────────────
_OUTCOMES = {o.value for o in AuditOutcome}


class Governor:
    """The only path from an agent to a tool. One instance per mission run."""

    def __init__(
        self,
        mission_id: str,
        registry: ToolRegistry,
        audit: AuditLog,
        events: EventPublisher,
        limits: MissionLimits | None = None,
        board: MissionBoard | None = None,
    ) -> None:
        self.mission_id = mission_id
        self._board = board
        self.limits = limits or MissionLimits()
        self.budget = MissionBudget(self.limits)
        self.loops = LoopDetector(self.limits.max_identical_failures, self.limits.max_consecutive_failures)
        self._registry = registry
        self._audit = audit
        self._events = events
        self._budget_announced = False

    @property
    def exhausted(self) -> str | None:
        return self.budget.exceeded()

    async def execute(
        self, tool: str, args: Any, ctx: ToolContext, *, allowed: Iterable[str] | None = None
    ) -> ToolResult:
        if ctx.mission_id != self.mission_id:
            raise ValueError(f"governor for {self.mission_id} got a call for {ctx.mission_id}")

        exceeded = self.budget.exceeded()
        if exceeded:
            result = ToolResult(False, error=f"Mission budget exhausted: {exceeded}. Stop and report your status.",
                                meta={"outcome": AuditOutcome.REFUSED_BUDGET})
            await self._announce_budget(exceeded)
            await self._record(ctx, tool, args, result)
            return result

        loop = self.loops.check(ctx.agent, tool, args)
        if loop:
            result = ToolResult(False, error=loop, meta={"outcome": AuditOutcome.REFUSED_LOOP})
            await self._events.log(self.mission_id, f"Loop guard stopped {ctx.agent}: repeated failing {tool}", LogLevel.WARN)
            await self._record(ctx, tool, args, result)
            return result

        self.budget.record_tool_call(ctx.agent)
        result = await self._registry.execute(tool, args, ctx, allowed=allowed)
        if result.evidence is not None:
            await self._capture_evidence(ctx, tool, result)
        self.loops.record(ctx.agent, tool, args, result)
        await self._record(ctx, tool, args, result)
        return result

    async def _capture_evidence(self, ctx: ToolContext, tool: str, result: ToolResult) -> None:
        if self._board is None:
            await self._events.publish(self.mission_id, EventKind.EVIDENCE, {"agent": ctx.agent, "tool": tool, **result.evidence})
            return
        try:
            evidence = await self._board.record_evidence(
                self.mission_id, agent=ctx.agent, tool=tool, data=result.evidence, task_id=ctx.task_id
            )
            result.meta["evidence_id"] = evidence.evidence_id
        except Exception:
            log.exception("could not store evidence from %s", tool)

    def record_llm_usage(self, agent: str, tokens: int) -> None:
        self.budget.record_tokens(agent, tokens)

    async def _announce_budget(self, reason: str) -> None:
        if self._budget_announced:
            return
        self._budget_announced = True
        await self._events.publish(self.mission_id, EventKind.DECISION, {
            "decision": "stop", "by": "governor", "reason": f"Mission budget exhausted: {reason}",
            "budget": self.budget.snapshot(),
        })

    async def _record(self, ctx: ToolContext, tool: str, args: Any, result: ToolResult) -> None:
        outcome = str(result.meta.get("outcome") or ("ok" if result.ok else "failed"))
        entry = AuditEntry(
            mission_id=self.mission_id,
            agent=ctx.agent,
            tool=tool,
            args=result.meta["args"] if "args" in result.meta else (
                preview_args(args) if isinstance(args, dict) else {"raw": str(args)[:300]}),
            action_class=result.meta.get("action_class"),
            outcome=AuditOutcome(outcome) if outcome in _OUTCOMES else AuditOutcome.FAILED,
            approved_by_human=bool(result.meta.get("approved_by_human")),
            error=result.error[:1000],
            display=result.display,
            duration_s=result.duration_s,
        )
        try:
            await self._audit.append(entry)
        except Exception:
            # Never lose the action because the audit write failed, but make it loud.
            log.exception("audit write failed for %s/%s", self.mission_id, tool)
