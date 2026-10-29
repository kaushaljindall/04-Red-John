"""
Agent definitions and run results.

A definition is data, not code: who the agent is, what it is good at (Maya
matches on `skills`), which tools it may call (its grant), and how long it
may work. Every guild member runs on the same runtime.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AgentDefinition(BaseModel):
    agent_id: str                       # stable key, e.g. "forge_takeshi"
    name: str                           # display name, e.g. "Takeshi"
    team: str                           # e.g. "forge"; messages to "@forge" reach every member
    role: str
    # First skill = the agent's specialism (breaks ties when Maya ranks equally qualified agents).
    skills: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    system_prompt: str
    tier: str = "smart"                 # "smart" or "fast" (see LLM gateway routes)
    max_iterations: int = 25
    # JSON-schema-ish contract for the `data` this agent must hand back when it finishes
    # (e.g. Kenji returns objectives and criteria). Only `required` keys are enforced.
    output_schema: dict[str, Any] | None = None
    # Shown in the control room.
    badge: str = ""
    char_idx: int = 0
    always_on: bool = False             # Atlas and Maya are never hired or released
    # Escalation specialists are preferred only when earlier attempts failed (Daichi, the Debugger).
    escalation: bool = False
    # Fallback mandate when Maya cannot write a mission-specific one.
    default_mandate: str = ""

    @property
    def teams(self) -> set[str]:
        return {self.team}


class RunStatus(StrEnum):
    DONE = "done"               # the agent says its job is finished
    FAILED = "failed"           # the agent tried and could not do it
    BLOCKED = "blocked"         # needs something it cannot get (a human, another agent, a budget)
    INCOMPLETE = "incomplete"   # stopped without a verdict (iteration limit, no finish call)


class AgentRunResult(BaseModel):
    agent: str
    task_id: str | None = None
    status: RunStatus
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    # Why the runtime (not the agent) ended the run, if it did.
    stop_reason: str | None = None
    iterations: int = 0
    tool_calls: int = 0
    tokens: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    inbox_seq: int = 0          # last message seen; pass back in on the next run
