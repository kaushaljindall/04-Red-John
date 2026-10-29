"""
Publishes mission events: store first (durable, sequenced), then bus (live).
Ported from Mycel `core/events.py`, minus the direct WebSocket call — the
realtime layer subscribes to the bus instead, so producers and sockets can
live in different processes.
"""
from __future__ import annotations

import logging
from typing import Any

from app.domain.events import EventEnvelope, EventKind, LogLevel, mission_topic
from app.infrastructure.bus.base import EventBus
from app.infrastructure.storage.base import EventStore

log = logging.getLogger(__name__)


class EventPublisher:
    def __init__(self, store: EventStore, bus: EventBus) -> None:
        self._store = store
        self._bus = bus

    async def publish(
        self, mission_id: str, kind: EventKind, data: dict[str, Any] | None = None
    ) -> EventEnvelope:
        event = await self._store.append(
            EventEnvelope(mission_id=mission_id, kind=kind, data=data or {})
        )
        try:
            await self._bus.publish(mission_topic(mission_id), event.model_dump(mode="json"))
        except Exception:
            # Stored is what matters; live clients will catch up from the store.
            log.exception("bus publish failed for mission %s", mission_id)
        return event

    async def log(self, mission_id: str, text: str, level: LogLevel = LogLevel.INFO) -> EventEnvelope:
        return await self.publish(mission_id, EventKind.LOG, {"level": level, "text": text})
