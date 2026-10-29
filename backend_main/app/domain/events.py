"""
Mission event envelope — the single message shape that flows from agents to
the event store, the bus and every WebSocket client.

`kind` values the Mycel control room already understands (log, hire, start,
finish, complete, approval_request, approval_response) are kept as-is so the
frontend can be reused; DevPilot adds its own on top.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.core.ids import new_id, now_ms


class EventKind(StrEnum):
    # Mycel-compatible
    LOG = "log"
    HIRE = "hire"
    START = "start"
    FINISH = "finish"
    COMPLETE = "complete"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    # DevPilot
    MISSION_CREATED = "mission_created"
    MISSION_STATUS = "mission_status"
    PLAN_UPDATED = "plan_updated"
    TASK_UPDATED = "task_updated"
    MESSAGE = "message"            # agent → agent (finding, handoff, critique, question)
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    EVIDENCE = "evidence"
    FINDING = "finding"
    ISSUE = "issue"                # question / clarification / blocker / dispute raised or resolved
    CRITERION = "criterion"        # acceptance criterion verified (or not)
    DECISION = "decision"          # Atlas: continue / retry / replan / escalate / stop
    RELEASE = "release"            # Maya releases an agent


class LogLevel(StrEnum):
    INFO = "info"
    ACTION = "action"
    SUCCESS = "success"
    WARN = "warn"
    ERROR = "error"
    ARMOR = "armor"


class EventEnvelope(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    mission_id: str
    kind: EventKind
    at: int = Field(default_factory=now_ms)
    # Monotonic per mission; assigned by the event store on append.
    seq: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


def mission_topic(mission_id: str) -> str:
    return f"mission.{mission_id}"
