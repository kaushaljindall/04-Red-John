"""In-process event bus for single-process deployments and tests."""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Self

log = logging.getLogger(__name__)


class _MemorySubscription:
    def __init__(self, bus: InMemoryEventBus, topic: str) -> None:
        self._bus = bus
        self._topic = topic
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(bus.queue_size)

    async def __aenter__(self) -> Self:
        self._bus.subscribers[self._topic].add(self._queue)
        return self

    async def __aexit__(self, *exc: object) -> None:
        subscribers = self._bus.subscribers.get(self._topic)
        if subscribers is not None:
            subscribers.discard(self._queue)
            if not subscribers:
                self._bus.subscribers.pop(self._topic, None)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> dict[str, Any]:
        return await self._queue.get()


class InMemoryEventBus:
    def __init__(self, queue_size: int = 1000) -> None:
        self.queue_size = queue_size
        self.subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = defaultdict(set)

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        self.subscribers.clear()

    async def publish(self, topic: str, message: dict[str, Any]) -> None:
        for queue in list(self.subscribers.get(topic, ())):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # A stalled consumer must never block producers; it can replay from the store.
                log.warning("bus subscriber on %s is full; dropping message", topic)

    def subscribe(self, topic: str) -> _MemorySubscription:
        return _MemorySubscription(self, topic)

    async def ping(self) -> bool:
        return True
