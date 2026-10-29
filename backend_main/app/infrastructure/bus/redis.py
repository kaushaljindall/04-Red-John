"""
Redis pub/sub event bus. Lets any number of API replicas and workers share
one live event stream: a worker publishes, every replica's WebSockets receive.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Self

from redis.asyncio import Redis
from redis.asyncio.client import PubSub

log = logging.getLogger(__name__)


class _RedisSubscription:
    def __init__(self, redis: Redis, channel: str) -> None:
        self._redis = redis
        self._channel = channel
        self._pubsub: PubSub | None = None

    async def __aenter__(self) -> Self:
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        await self._pubsub.subscribe(self._channel)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._pubsub is not None:
            await self._pubsub.unsubscribe(self._channel)
            await self._pubsub.aclose()
            self._pubsub = None

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> dict[str, Any]:
        assert self._pubsub is not None, "use the subscription inside 'async with'"
        while True:
            raw = await self._pubsub.get_message(timeout=None)
            if raw is None or raw.get("type") != "message":
                continue
            try:
                return json.loads(raw["data"])
            except (TypeError, ValueError):
                log.warning("dropping malformed bus message on %s", self._channel)


class RedisEventBus:
    def __init__(self, url: str, channel_prefix: str = "devpilot:") -> None:
        self._url = url
        self._prefix = channel_prefix
        self._redis: Redis | None = None

    @property
    def redis(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("RedisEventBus used before start()")
        return self._redis

    async def start(self) -> None:
        self._redis = Redis.from_url(self._url, decode_responses=True)
        await self._redis.ping()
        log.info("redis event bus connected")

    async def stop(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def publish(self, topic: str, message: dict[str, Any]) -> None:
        await self.redis.publish(self._prefix + topic, json.dumps(message, default=str))

    def subscribe(self, topic: str) -> _RedisSubscription:
        return _RedisSubscription(self.redis, self._prefix + topic)

    async def ping(self) -> bool:
        try:
            return bool(await self.redis.ping())
        except Exception:
            return False
