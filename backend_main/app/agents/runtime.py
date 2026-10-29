"""
Agent runtime: one loop that runs any guild member on a task.

    briefing → [LLM → tool calls through the Governor → results back] × N → finish

Ported from Mycel's `ArchitectureBaseAgent.run_task`, with what DevPilot
needs on top: tool grants per agent, every call governed and audited,
evidence ids handed back to the model, an explicit `finish` contract instead
of parsing free text, inbox/outbox on the mission board, context trimming
that never splits a tool call from its result, and START / FINISH / log
events that drive the control room's pixel office.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.agents.prompts import briefing, system_prompt
from app.core.config import Settings
from app.core.errors import ExternalServiceError, NotFoundError
from app.domain.agents import AgentDefinition, AgentRunResult, RunStatus
from app.domain.events import EventKind, LogLevel
from app.domain.mission import Task
from app.infrastructure.llm.gateway import LLMGateway
from app.infrastructure.llm.types import ChatRequest, ModelTier, ToolCallRejected
from app.security.sandbox import WorkspaceManager
from app.services.event_publisher import EventPublisher
from app.services.governance import Governor
from app.services.mission_board import MissionBoard
from app.tools.registry import Approver, ToolContext, ToolRegistry, nullable_optionals

log = logging.getLogger(__name__)

FINISH_TOOL = "finish"
MAX_NUDGES = 2
MAX_REJECTED_CALLS = 3
HISTORY_CHAR_BUDGET = 90_000

FINISH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": FINISH_TOOL,
        "description": "End your turn. Call exactly once, when your task is done, has failed, or you are blocked.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["done", "failed", "blocked"]},
                "summary": {"type": "string", "description": "What you did and what you verified (cite evidence ids)"},
                "data": {"type": "object", "description": "Structured result, if your role defines one"},
            },
            "required": ["status", "summary"],
        },
    },
}


def _size(message: dict[str, Any]) -> int:
    return len(json.dumps(message, default=str))


def trim_history(messages: list[dict[str, Any]], budget: int = HISTORY_CHAR_BUDGET, keep_groups: int = 3) -> list[dict[str, Any]]:
    """Drop the oldest steps until the history fits. A step is an assistant message plus the tool
    results that answer it, so a tool result is never left without the call it belongs to."""
    head, rest = messages[:2], messages[2:]
    if rest and rest[0].get("role") == "user" and str(rest[0].get("content", "")).startswith("[context note]"):
        rest = rest[1:]
    groups: list[list[dict[str, Any]]] = []
    for message in rest:
        if message.get("role") == "tool" and groups:
            groups[-1].append(message)
        else:
            groups.append([message])
    dropped = 0
    while len(groups) > keep_groups and sum(_size(m) for m in head) + sum(_size(m) for g in groups for m in g) > budget:
        groups.pop(0)
        dropped += 1
    if not dropped:
        return messages
    note = {"role": "user", "content": f"[context note] {dropped} earlier step(s) were removed to save space. "
                                       "Use mission_state or re-read files if you need them again."}
    return head + [note] + [m for g in groups for m in g]


@dataclass
class _RunState:
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    nudges: int = 0
    rejected: int = 0


class AgentRuntime:
    def __init__(
        self,
        *,
        llm: LLMGateway,
        tools: ToolRegistry,
        board: MissionBoard,
        events: EventPublisher,
        workspaces: WorkspaceManager,
        settings: Settings,
        approver: Approver,
    ) -> None:
        self._llm = llm
        self._tools = tools
        self._board = board
        self._events = events
        self._workspaces = workspaces
        self._settings = settings
        self._approver = approver

    def granted_tools(self, definition: AgentDefinition) -> list[str]:
        granted = [name for name in definition.tools if self._tools.get(name)]
        unknown = set(definition.tools) - set(granted)
        if unknown:
            log.warning("%s is granted unknown tools: %s", definition.name, sorted(unknown))
        return granted

    async def run(
        self,
        definition: AgentDefinition,
        mission_id: str,
        governor: Governor,
        *,
        task_id: str | None = None,
        instructions: str | None = None,
        mandate: str | None = None,
        inbox_seq: int = 0,
    ) -> AgentRunResult:
        mission = await self._board.get(mission_id)
        task: Task | None = mission.task(task_id) if task_id else None
        inbox = await self._board.inbox(mission_id, definition.name, teams=definition.teams, after_seq=inbox_seq)
        ctx = ToolContext(
            mission_id=mission_id, agent=definition.name, workspace=self._workspaces.get(mission_id),
            settings=self._settings, approver=self._approver, events=self._events, task_id=task_id,
            board=self._board, teams=definition.teams,
            inbox_seq=max([inbox_seq, *(m.seq or 0 for m in inbox)]),
        )
        granted = self.granted_tools(definition)
        finish = {**FINISH_SCHEMA, "function": {**FINISH_SCHEMA["function"],
                                                 "parameters": nullable_optionals(FINISH_SCHEMA["function"]["parameters"])}}
        schemas = self._tools.schemas(granted) + [finish]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt(definition, mandate)},
            {"role": "user", "content": briefing(mission, task, inbox, instructions)},
        ]

        what = task.title if task else (instructions or "Working")[:80]
        await self._events.publish(mission_id, EventKind.START, {"agent": definition.name, "task": what,
                                                                 "team": definition.team, "task_id": task_id})
        state = _RunState()
        evidence_ids: list[str] = []
        result: AgentRunResult | None = None
        try:
            result = await self._loop(definition, mission_id, governor, ctx, granted, schemas, messages, state, evidence_ids)
        except Exception as exc:  # the runtime must always hand a result back to the orchestrator
            log.exception("agent %s crashed", definition.name)
            result = self._result(definition, ctx, state, evidence_ids, RunStatus.FAILED,
                                  f"Runtime error: {type(exc).__name__}: {exc}", stop_reason="runtime_error")
        finally:
            final = result or self._result(definition, ctx, state, evidence_ids, RunStatus.INCOMPLETE,
                                           "Run was cancelled", stop_reason="cancelled")
            await self._events.publish(mission_id, EventKind.FINISH, {
                "agent": definition.name, "status": final.status, "summary": final.summary[:500],
                "task_id": task_id, "iterations": final.iterations, "tool_calls": final.tool_calls,
            })
            level = LogLevel.SUCCESS if final.status == RunStatus.DONE else LogLevel.WARN
            await self._events.log(mission_id, f"{definition.name} finished ({final.status}): {final.summary[:240]}", level)
        return result

    async def _loop(
        self,
        definition: AgentDefinition,
        mission_id: str,
        governor: Governor,
        ctx: ToolContext,
        granted: list[str],
        schemas: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        state: _RunState,
        evidence_ids: list[str],
    ) -> AgentRunResult:
        def done(status: RunStatus, summary: str, **extra: Any) -> AgentRunResult:
            return self._result(definition, ctx, state, evidence_ids, status, summary, **extra)

        while state.iterations < definition.max_iterations:
            state.iterations += 1
            try:
                mission = await self._board.get(mission_id)
            except NotFoundError:
                return done(RunStatus.BLOCKED, "The mission no longer exists.", stop_reason="mission_missing")
            if mission.is_terminal:
                return done(RunStatus.BLOCKED, f"The mission is {mission.status}.", stop_reason="mission_stopped")
            exhausted = governor.exhausted
            if exhausted:
                return done(RunStatus.BLOCKED, f"Mission budget exhausted: {exhausted}", stop_reason="budget")

            messages[:] = trim_history(messages)
            try:
                response = await self._llm.complete(ChatRequest(
                    messages=messages, tools=schemas, tier=ModelTier(definition.tier), caller=definition.name))
            except ToolCallRejected as exc:
                # The provider refused the model's own malformed call. Show it the error and let it try again.
                state.rejected += 1
                if state.rejected > MAX_REJECTED_CALLS:
                    return done(RunStatus.FAILED, f"Repeated malformed tool calls: {exc.message}",
                                stop_reason="tool_call_rejected")
                messages.append({"role": "user", "content": (
                    f"Your last tool call was rejected before it ran: {exc.message}\n"
                    "Use the exact tool name, send arguments that match its schema, and leave out optional "
                    "arguments you do not need.")})
                continue
            except ExternalServiceError as exc:
                return done(RunStatus.FAILED, f"The language model was unavailable: {exc.message}",
                            stop_reason="llm_unavailable")
            state.tokens += response.usage.total
            governor.record_llm_usage(definition.name, response.usage.total)
            messages.append(response.assistant_message())

            if response.content and response.tool_calls:
                await self._events.log(mission_id, f"{definition.name}: {response.content[:300]}", LogLevel.INFO)

            if not response.tool_calls:
                # Some models (gpt-oss) write their finish call as JSON text. Accept it if it passes
                # the same checks as a real finish call.
                verdict = self._finish_from_text(definition, response.content)
                if verdict is not None:
                    status, summary, data = verdict
                    return done(status, summary, data=data)
                if state.nudges >= MAX_NUDGES:
                    return done(RunStatus.INCOMPLETE, response.content or "Stopped without calling finish.",
                                stop_reason="no_finish")
                state.nudges += 1
                messages.append({"role": "user", "content": (
                    "You must act through your tools. If your work is complete (or you cannot continue), "
                    "call `finish` with a status and a factual summary.")})
                continue

            for call in response.tool_calls:
                if call.name == FINISH_TOOL:
                    verdict = self._accept_finish(definition, call.arguments)
                    if isinstance(verdict, str):
                        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                         "content": verdict})
                        continue
                    status, summary, data = verdict
                    return done(status, summary, data=data)
                if call.raw_arguments is not None:
                    content = "Error: the arguments were not valid JSON. Send a JSON object matching the tool's schema."
                else:
                    state.tool_calls += 1
                    result = await governor.execute(call.name, call.arguments, ctx, allowed=granted)
                    if result.meta.get("evidence_id"):
                        evidence_ids.append(result.meta["evidence_id"])
                    content = result.to_model_text()
                    await self._events.log(
                        mission_id,
                        f"{definition.name} → {call.name}: {(result.display or result.error or 'done')[:200]}",
                        LogLevel.ACTION if result.ok else LogLevel.WARN,
                    )
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content})

        return done(RunStatus.INCOMPLETE, f"Stopped after {definition.max_iterations} iterations without finishing.",
                    stop_reason="iteration_limit")

    @classmethod
    def _finish_from_text(cls, definition: AgentDefinition, content: str) -> tuple[RunStatus, str, dict[str, Any]] | None:
        start, end = content.find("{"), content.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(content[start : end + 1])
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("name") == FINISH_TOOL and isinstance(payload.get("arguments"), dict):
            payload = payload["arguments"]
        if "status" not in payload or "summary" not in payload:
            return None
        verdict = cls._accept_finish(definition, payload)
        return None if isinstance(verdict, str) else verdict

    @staticmethod
    def _accept_finish(definition: AgentDefinition, args: dict[str, Any]) -> tuple[RunStatus, str, dict[str, Any]] | str:
        """Returns (status, summary, data), or an error message for the model to fix and call again."""
        status = str(args.get("status", "")).lower()
        if status not in ("done", "failed", "blocked"):
            return "Error: status must be one of done, failed, blocked."
        summary = str(args.get("summary", "")).strip()
        if not summary:
            return "Error: summary is required — say what you did and what you verified."
        data = args.get("data") or {}
        if not isinstance(data, dict):
            return "Error: data must be an object."
        if status == "done" and definition.output_schema:
            missing = [k for k in definition.output_schema.get("required", []) if k not in data]
            if missing:
                return f"Error: data is missing required field(s): {', '.join(missing)}. Call finish again with them."
        return RunStatus(status), summary, data

    @staticmethod
    def _result(
        definition: AgentDefinition, ctx: ToolContext, state: _RunState, evidence_ids: list[str],
        status: RunStatus, summary: str, *, data: dict[str, Any] | None = None, stop_reason: str | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            agent=definition.name, task_id=ctx.task_id, status=status, summary=summary, data=data or {},
            stop_reason=stop_reason, iterations=state.iterations, tool_calls=state.tool_calls, tokens=state.tokens,
            evidence_ids=list(dict.fromkeys(evidence_ids)), inbox_seq=ctx.inbox_seq,
        )
