"""
Maya — DevPilot's hiring engine. Ported from Mycel's Maya + Smart Hiring System.

Two parts, kept apart on purpose:

- `HiringEngine` decides *who*: deterministic skill scoring with hard filters,
  so every hire is explainable ("Daichi: debugging, python — 1.00") and the
  same situation always gives the same answer. (Mycel's doc set the same
  rule: the LLM never bypasses the scoring.)
- `HiringService` (Maya) acts on it: hires at mission start and mid-mission,
  escalates a task that keeps failing to a different specialist, keeps hiring
  idempotent, releases agents, and asks the LLM only to *write* each hire's
  one-line mandate — with a fallback when no model is available.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable

from app.agents.guild import AgentRegistry, clearance
from app.domain.agents import AgentDefinition
from app.domain.events import EventKind, LogLevel
from app.domain.hiring import CandidateScore, HireRecord, HiringRequest, SkillNeed
from app.domain.mission import Mission, Task, TaskKind
from app.domain.skills import AREA_SKILLS, KIND_SKILLS, infer_skills, normalize_skills
from app.infrastructure.llm.gateway import LLMGateway
from app.infrastructure.llm.types import ChatRequest, ModelTier
from app.services.event_publisher import EventPublisher
from app.services.mission_board import MissionBoard

log = logging.getLogger(__name__)

CORE_WEIGHT = 2.0       # the kind of work (debugging, review…) — must be covered
EXPLICIT_WEIGHT = 1.5   # skills the planner asked for
INFERRED_WEIGHT = 1.0   # skills read from the task text

# Inferred from text only when they describe the area or a specialism; generic verbs are too noisy.
_INFERABLE = AREA_SKILLS | {"security", "refactoring", "test-authoring", "debugging"}
# Work that must be done by someone other than whoever wrote the change.
_INDEPENDENT_KINDS = {TaskKind.VERIFY, TaskKind.REVIEW, TaskKind.SECURITY}
_AUTHOR_KINDS = {TaskKind.IMPLEMENT, TaskKind.FIX}
# Hired at the start of every mission: the Shogun's Council.
PLANNING_CORE: tuple[tuple[str, str], ...] = (
    ("requirements", "Turn the request into objectives and acceptance criteria"),
    ("recon", "Map the codebase, its conventions and its test baseline"),
    ("planning", "Break the mission into a task graph"),
)
# Specialists worth hiring up front when the request itself calls for them.
_EARLY_SPECIALISTS = {"security": "The request concerns security; inspection must start with the planning"}


class HiringEngine:
    def __init__(self, registry: AgentRegistry, threshold: float = 0.5) -> None:
        self._registry = registry
        self._threshold = threshold

    # ── requests ─────────────────────────────────────────────────────────
    @staticmethod
    def _needs(*groups: tuple[Iterable[str], float]) -> list[SkillNeed]:
        weights: dict[str, float] = {}
        for skills, weight in groups:
            for skill in skills:
                weights[skill] = max(weights.get(skill, 0.0), weight)
        return [SkillNeed(skill=s, weight=w) for s, w in weights.items()]

    def request_for_task(
        self, task: Task, mission: Mission, *, exclude: Iterable[str] = (), escalation: bool = False
    ) -> HiringRequest:
        text = f"{task.title}\n{task.description}"
        explicit = normalize_skills(task.required_skills)
        kind_skills = KIND_SKILLS.get(task.kind, ())
        if not kind_skills:  # no default for this kind: the first explicit skill is the core need
            kind_skills, explicit = tuple(explicit[:1]), explicit[1:]
        needs = self._needs(
            (kind_skills, CORE_WEIGHT),
            (explicit, EXPLICIT_WEIGHT),
            (infer_skills(text, only=frozenset(_INFERABLE)), INFERRED_WEIGHT),
        )
        excluded = set(exclude)
        if task.kind in _INDEPENDENT_KINDS:
            excluded |= {t.assignee for t in mission.tasks if t.kind in _AUTHOR_KINDS and t.assignee}
        return HiringRequest(needs=needs, reason=f"{task.kind} task: {task.title}", task_id=task.id,
                             exclude=sorted(excluded), escalation=escalation)

    def request_for_skills(self, skills: Iterable[str], reason: str, *, exclude: Iterable[str] = ()) -> HiringRequest:
        normalized = normalize_skills(list(skills))
        if not normalized:
            raise ValueError(f"none of {list(skills)} is a known skill")
        # The first skill named is the one that matters; the rest refine the choice.
        needs = self._needs((normalized[:1], CORE_WEIGHT), (normalized[1:], EXPLICIT_WEIGHT))
        return HiringRequest(needs=needs, reason=reason, exclude=sorted(set(exclude)))

    # ── scoring ──────────────────────────────────────────────────────────
    def score(self, definition: AgentDefinition, request: HiringRequest) -> CandidateScore:
        have = set(normalize_skills(definition.skills))
        total = sum(n.weight for n in request.needs) or 1.0
        matched = [n.skill for n in request.needs if n.skill in have]
        missing = [n.skill for n in request.needs if n.skill not in have]
        score = round(sum(n.weight for n in request.needs if n.skill in have) / total, 3)
        core_missing = [n.skill for n in request.needs if n.weight >= CORE_WEIGHT and n.skill not in have]
        if definition.always_on:
            eligible, note = False, "always on duty; not hireable"
        elif definition.name in request.exclude:
            eligible, note = False, "excluded for this work"
        elif core_missing:
            eligible, note = False, f"lacks core skill: {', '.join(core_missing)}"
        elif score < self._threshold:
            eligible, note = False, f"score {score:.2f} below {self._threshold:.2f}"
        else:
            eligible, note = True, ""
        return CandidateScore(agent=definition.name, score=score, matched=matched, missing=missing,
                              eligible=eligible, note=note)

    def rank(
        self, request: HiringRequest, *, hired: set[str] = frozenset(), busy: set[str] = frozenset()
    ) -> list[CandidateScore]:
        order = {d.name: i for i, d in enumerate(self._registry.all())}
        core = {n.skill for n in request.needs if n.weight >= CORE_WEIGHT}
        scored = [(self.score(d, request), d) for d in self._registry.all()]

        def key(item: tuple[CandidateScore, AgentDefinition]) -> tuple:
            candidate, definition = item
            adjusted = candidate.score + (0.05 if candidate.agent in hired else 0.0) - (0.15 if candidate.agent in busy else 0.0)
            # Among equals prefer the specialist: an agent's first skill is their specialism, then the share
            # of their skills this work uses.
            primary = bool(definition.skills) and definition.skills[0] in core
            focus = len(candidate.matched) / max(1, len(definition.skills))
            wrong_line = definition.escalation != request.escalation
            return (not candidate.eligible, -round(adjusted, 3), wrong_line, not primary, -round(focus, 3),
                    order[candidate.agent])

        return [candidate for candidate, _ in sorted(scored, key=key)]


class HiringService:
    """Maya."""

    def __init__(
        self,
        registry: AgentRegistry,
        engine: HiringEngine,
        board: MissionBoard,
        events: EventPublisher,
        llm: LLMGateway,
        *,
        llm_mandates: bool = True,
        escalate_after_failures: int = 2,
    ) -> None:
        self.registry = registry
        self.engine = engine
        self._board = board
        self._events = events
        self._llm = llm
        self._llm_mandates = llm_mandates
        self._escalate_after = escalate_after_failures

    # ── public ───────────────────────────────────────────────────────────
    async def initial_staffing(self, mission_id: str) -> list[HireRecord]:
        """Staff the Shogun's Council, plus any specialist the request itself calls for."""
        mission = await self._board.get(mission_id)
        await self._events.log(mission_id, "Maya reviewing the request. Only load-bearing specialists will be hired.",
                               LogLevel.ACTION)
        hires: list[HireRecord] = []
        for skill, reason in PLANNING_CORE:
            hires += await self.hire(mission_id, self.engine.request_for_skills([skill], reason))
        requested = set(infer_skills(mission.request))
        for skill, reason in _EARLY_SPECIALISTS.items():
            if skill in requested:
                hires += await self.hire(mission_id, self.engine.request_for_skills([skill], reason))
        names = ", ".join(h.agent for h in hires)
        await self._events.log(mission_id, f"Maya: opening task force is {names}. More will be hired as the plan "
                                           "shows what the work needs.", LogLevel.SUCCESS)
        return hires

    async def hire(
        self, mission_id: str, request: HiringRequest, *, count: int = 1, busy: set[str] = frozenset()
    ) -> list[HireRecord]:
        mission = await self._board.get(mission_id)
        ranked = self.engine.rank(request, hired=set(mission.hired_agents), busy=set(busy))
        chosen = [c for c in ranked if c.eligible][:count]
        if not chosen:
            needs = ", ".join(n.skill for n in request.needs)
            best = ranked[0] if ranked else None
            await self._events.log(mission_id, f"Maya could not staff '{request.reason}': nobody covers {needs}"
                                   + (f" (closest: {best.agent}, {best.note})" if best else ""), LogLevel.WARN)
            return []
        records = []
        for candidate in chosen:
            records.append(await self._hire_one(mission, candidate, request))
        return records

    async def hire_for_skills(self, mission_id: str, skills: list[str], reason: str, *,
                              exclude: Iterable[str] = ()) -> list[HireRecord]:
        return await self.hire(mission_id, self.engine.request_for_skills(skills, reason, exclude=exclude))

    async def staff_task(self, mission_id: str, task_id: str, *, busy: set[str] = frozenset()) -> AgentDefinition | None:
        """Who should do this task now? Keeps the current assignee unless the task keeps failing."""
        mission = await self._board.get(mission_id)
        task = mission.task(task_id)
        exclude: list[str] = []
        if task.assignee:
            failed_enough = task.attempts >= self._escalate_after
            still_hired = task.assignee in mission.hired_agents
            if not failed_enough and still_hired:
                return self.registry.require(task.assignee)
            if failed_enough:
                # New evidence (repeated failure) changes who does the work.
                exclude.append(task.assignee)
                await self._events.log(mission_id, f"Maya: '{task.title}' failed {task.attempts} time(s) under "
                                                   f"{task.assignee}; bringing in a different specialist.", LogLevel.ACTION)
        request = self.engine.request_for_task(task, mission, exclude=exclude, escalation=bool(exclude))
        records = await self.hire(mission_id, request, busy=busy)
        if not records:
            return None
        return self.registry.require(records[0].agent)

    async def release(self, mission_id: str, agent: str, reason: str) -> HireRecord:
        record = await self._board.release(mission_id, agent, reason)
        await self._events.publish(mission_id, EventKind.RELEASE, {"agent": agent, "reason": reason})
        await self._events.log(mission_id, f"Maya released {agent}: {reason}", LogLevel.INFO)
        return record

    # ── internals ────────────────────────────────────────────────────────
    async def _hire_one(self, mission: Mission, candidate: CandidateScore, request: HiringRequest) -> HireRecord:
        definition = self.registry.require(candidate.agent)
        existing = mission.hire_of(definition.name)
        if existing is not None:
            return existing
        task = mission.task(request.task_id) if request.task_id else None
        record = HireRecord(
            agent=definition.name, agent_id=definition.agent_id, team=definition.team, role=definition.role,
            badge=definition.badge, clearance=clearance(definition),
            mandate=await self._mandate(definition, mission, request, task),
            reason=request.reason, score=candidate.score, matched_skills=candidate.matched, task_id=request.task_id,
        )
        stored, new = await self._board.record_hire(mission.mission_id, record)
        if new:
            await self._events.publish(mission.mission_id, EventKind.HIRE, {
                # Mycel hire-card shape
                "agent": stored.agent, "team": stored.team.capitalize(), "role": stored.role, "badge": stored.badge,
                "clearance": stored.clearance, "mandate": stored.mandate,
                # DevPilot
                "reason": stored.reason, "score": stored.score, "matched_skills": stored.matched_skills,
                "task_id": stored.task_id,
            })
            await self._events.log(
                mission.mission_id,
                f"Maya → hire {stored.agent} ({stored.role}; {', '.join(stored.matched_skills)}; "
                f"score {stored.score:.2f}) for: {stored.reason}",
                LogLevel.ACTION,
            )
        return stored

    async def _mandate(self, definition: AgentDefinition, mission: Mission, request: HiringRequest,
                       task: Task | None) -> str:
        fallback = definition.default_mandate or f"{definition.role}: {request.reason}"
        if not (self._llm_mandates and self._llm.available):
            return fallback
        remit = definition.system_prompt.strip().splitlines()[0]
        can_change_code = clearance(definition) == "AMBER"
        prompt = (
            f"Mission request: {mission.request[:700]}\n"
            f"Hire: {definition.name}, {definition.role}. Their remit: {remit}\n"
            + ("They can change code.\n" if can_change_code
               else "They cannot change code or write tests; other agents act on what they find.\n")
            + f"Why hired: {request.reason}\n"
            + (f"Task: {task.title}\n" if task else "")
            + f'Example of the tone: "{definition.default_mandate}"\n'
            "Write this hire's mandate: ONE short imperative sentence, at most 18 words, specific to this mission "
            "and strictly within their remit — never assign work that belongs to another role. "
            "Reply with the sentence only."
        )
        try:
            response = await self._llm.complete(ChatRequest(
                messages=[{"role": "system", "content": "You are Maya, head of hiring for DevPilot, an AI engineering guild."},
                          {"role": "user", "content": prompt}],
                tier=ModelTier.FAST, temperature=0.4, max_tokens=600, caller="Maya"))
        except Exception as exc:  # a missing mandate must never block a hire
            log.warning("mandate generation failed for %s: %s", definition.name, exc)
            return fallback
        text = (response.content or "").strip().splitlines()[0:1]
        mandate = text[0].strip().strip('"').strip() if text else ""
        return mandate[:200] if len(mandate) >= 8 else fallback
