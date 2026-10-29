"""Hiring models: what Maya is asked for, how she scores candidates, and what she decides."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.ids import now_ms


class SkillNeed(BaseModel):
    skill: str
    weight: float = 1.0


class HiringRequest(BaseModel):
    """'Find me someone who can do this.' Built from a task, a request, or an explicit need."""

    needs: list[SkillNeed]
    reason: str
    task_id: str | None = None
    # Agents that may not take this work (e.g. the implementer may not verify their own change).
    exclude: list[str] = Field(default_factory=list)
    # Earlier attempts failed: prefer escalation specialists.
    escalation: bool = False


class CandidateScore(BaseModel):
    agent: str
    score: float                    # 0..1, share of weighted needs covered
    matched: list[str]
    missing: list[str]
    eligible: bool
    note: str = ""


class HireRecord(BaseModel):
    agent: str
    agent_id: str
    team: str
    role: str
    badge: str = ""
    clearance: str = "GREEN"        # AMBER = holds tools that can change code or run commands
    mandate: str
    reason: str
    score: float
    matched_skills: list[str] = Field(default_factory=list)
    task_id: str | None = None
    hired_by: str = "Maya"
    hired_at: int = Field(default_factory=now_ms)
    released_at: int | None = None
    release_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.released_at is None
