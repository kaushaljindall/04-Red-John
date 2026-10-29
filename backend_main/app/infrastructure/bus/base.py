"""
Event bus interface: fire-and-forget fan-out of JSON messages by topic.

The bus is for live delivery only. Durability lives in the event store, so a
client that connects late replays from the store and then tails the bus.

Subscriptions are async context managers: the subscription is registered
when `__aenter__` returns, so a caller can subscribe, *then* replay from the
store, and know nothing published in between was missed.
"""
from __future__ import annotations

from typing import Any, Protocol, Self


class Subscription(Protocol):
    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, *exc: object) -> None: ...

    def __aiter__(self) -> Self: ...

    async def __anext__(self) -> dict[str, Any]: ...


class EventBus(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def publish(self, topic: str, message: dict[str, Any]) -> None: ...

    def subscribe(self, topic: str) -> Subscription: ...

    async def ping(self) -> bool: ...
