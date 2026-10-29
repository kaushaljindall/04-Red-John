"""
Composition root. Builds every long-lived dependency once, from settings,
and owns their start/stop order. Nothing else constructs infrastructure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agents.guild import AgentRegistry, build_guild
from app.agents.runtime import AgentRuntime
from app.core.config import BusBackend, Settings, StorageBackend
from app.domain.governance import MissionLimits
from app.infrastructure.bus.base import EventBus
from app.infrastructure.bus.memory import InMemoryEventBus
from app.infrastructure.llm.gateway import LLMGateway, Route
from app.infrastructure.llm.openai_compat import OpenAICompatibleProvider
from app.infrastructure.llm.types import ModelTier
from app.infrastructure.storage.base import Storage
from app.infrastructure.storage.memory import InMemoryStorage
from app.security.sandbox import WorkspaceManager
from app.services.approval_gate import ApprovalGate
from app.services.event_publisher import EventPublisher
from app.services.governance import Governor
from app.services.hiring import HiringEngine, HiringService
from app.services.mission_board import MissionBoard
from app.tools.builtin import build_registry
from app.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


def build_llm_gateway(settings: Settings) -> LLMGateway:
    groq = OpenAICompatibleProvider(
        "groq", settings.groq_base_url, settings.groq_api_keys,
        timeout_s=settings.llm_request_timeout_s, max_attempts=settings.llm_max_attempts,
    )
    gemini = OpenAICompatibleProvider(
        "gemini", settings.gemini_base_url, settings.gemini_api_keys,
        timeout_s=settings.llm_request_timeout_s, max_attempts=settings.llm_max_attempts,
    )
    fallback = Route(gemini, settings.llm_fallback_model)
    return LLMGateway(
        routes={
            ModelTier.SMART: [Route(groq, settings.llm_model_smart), fallback],
            ModelTier.FAST: [Route(groq, settings.llm_model_fast), fallback],
        },
        max_concurrency=settings.llm_max_concurrency,
    )


def build_storage(settings: Settings) -> Storage:
    if settings.storage_backend == StorageBackend.MONGO:
        from app.infrastructure.storage.mongo import MongoStorage

        return MongoStorage(settings.mongodb_url, settings.mongodb_database)
    return InMemoryStorage()


def build_event_bus(settings: Settings) -> EventBus:
    if settings.bus_backend == BusBackend.REDIS:
        from app.infrastructure.bus.redis import RedisEventBus

        return RedisEventBus(settings.redis_url)
    return InMemoryEventBus()


def default_limits(settings: Settings) -> MissionLimits:
    return MissionLimits(
        max_tool_calls=settings.mission_max_tool_calls,
        max_llm_tokens=settings.mission_max_llm_tokens,
        max_wall_clock_s=settings.mission_max_wall_clock_s,
        max_identical_failures=settings.mission_max_identical_failures,
        max_consecutive_failures=settings.mission_max_consecutive_failures,
    )


@dataclass
class Container:
    settings: Settings
    storage: Storage
    event_bus: EventBus
    events: EventPublisher
    approvals: ApprovalGate
    llm: LLMGateway
    tools: ToolRegistry
    workspaces: WorkspaceManager
    board: MissionBoard
    runtime: AgentRuntime
    guild: AgentRegistry
    hiring: HiringService

    @classmethod
    def build(cls, settings: Settings) -> Container:
        storage = build_storage(settings)
        bus = build_event_bus(settings)
        events = EventPublisher(storage.events, bus)
        workspaces = WorkspaceManager(settings.workspaces_dir, settings.allowed_repo_roots)
        approvals = ApprovalGate(
            storage.approvals, bus, events,
            timeout_s=settings.approval_timeout_s,
            auto_approve_categories=set(settings.auto_approve_categories),
        )
        llm = build_llm_gateway(settings)
        tools = build_registry()
        board = MissionBoard(storage, events, workspaces)
        guild = build_guild()
        return cls(
            settings=settings,
            storage=storage,
            event_bus=bus,
            events=events,
            approvals=approvals,
            llm=llm,
            tools=tools,
            workspaces=workspaces,
            board=board,
            runtime=AgentRuntime(llm=llm, tools=tools, board=board, events=events, workspaces=workspaces,
                                 settings=settings, approver=approvals),
            guild=guild,
            hiring=HiringService(
                guild, HiringEngine(guild, settings.hiring_threshold), board, events, llm,
                llm_mandates=settings.hiring_llm_mandates,
                escalate_after_failures=settings.escalate_after_failures,
            ),
        )

    def governor(self, mission_id: str, limits: MissionLimits | None = None) -> Governor:
        """A fresh governor for one mission run."""
        return Governor(mission_id, self.tools, self.storage.audit, self.events,
                        limits or default_limits(self.settings), board=self.board)

    async def start(self) -> None:
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        await self.storage.start()
        await self.event_bus.start()
        if not self.llm.available:
            log.warning("no LLM API keys configured; agents will not be able to run")
        if self.settings.auto_approve_categories:
            log.warning("auto-approving categories without a human: %s", self.settings.auto_approve_categories)
        log.info(
            "container started (storage=%s, bus=%s, tools=%d, llm=%s)",
            self.settings.storage_backend, self.settings.bus_backend, len(self.tools.names()), self.llm.describe(),
        )

    async def stop(self) -> None:
        await self.llm.aclose()
        await self.event_bus.stop()
        await self.storage.stop()
