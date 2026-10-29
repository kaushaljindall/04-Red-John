"""
Redis-backed job queue (Streams + consumer group) and mission leases.

Streams give at-least-once delivery: a job stays in the group's pending list
until a worker acknowledges it, and XAUTOCLAIM hands jobs held by a dead
worker to a live one. Leases are SET NX PX keys, renewed and released with
compare-and-set scripts so a worker can never extend or drop someone else's.
"""
from __future__ import annotations

import logging

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from app.infrastructure.jobs.base import Job

log = logging.getLogger(__name__)

_RENEW = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""
_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


class RedisJobQueue:
    def __init__(self, url: str, *, stream: str = "devpilot:jobs", group: str = "workers",
                 claim_after_s: float = 120.0, maxlen: int = 10_000) -> None:
        self._url = url
        self._stream = stream
        self._group = group
        self._claim_after_ms = int(claim_after_s * 1000)
        self._maxlen = maxlen
        self._redis: Redis | None = None

    @property
    def redis(self) -> Redis:
        if self._redis is None:
            raise RuntimeError("RedisJobQueue used before start()")
        return self._redis

    async def start(self) -> None:
        self._redis = Redis.from_url(self._url, decode_responses=True)
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def stop(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def enqueue(self, mission_id: str) -> Job:
        job = Job(mission_id=mission_id)
        job.receipt = await self.redis.xadd(
            self._stream, {"job_id": job.job_id, "mission_id": mission_id, "enqueued_at": str(job.enqueued_at)},
            maxlen=self._maxlen, approximate=True)
        return job

    @staticmethod
    def _job(entry_id: str, fields: dict, deliveries: int = 1) -> Job:
        return Job(job_id=fields.get("job_id", entry_id), mission_id=fields["mission_id"],
                   enqueued_at=int(fields.get("enqueued_at", 0)), deliveries=deliveries, receipt=entry_id)

    async def dequeue(self, consumer: str, timeout_s: float) -> Job | None:
        # First, jobs whose worker went quiet for too long.
        claimed = await self.redis.xautoclaim(self._stream, self._group, consumer,
                                              min_idle_time=self._claim_after_ms, start_id="0-0", count=1)
        entries = claimed[1] if claimed else []
        for entry_id, fields in entries:
            if fields:  # entries trimmed from the stream come back empty
                pending = await self.redis.xpending_range(self._stream, self._group, min=entry_id, max=entry_id, count=1)
                deliveries = pending[0]["times_delivered"] if pending else 2
                log.warning("reclaimed job %s for mission %s from a silent worker", entry_id, fields.get("mission_id"))
                return self._job(entry_id, fields, deliveries)
            await self.redis.xack(self._stream, self._group, entry_id)
        response = await self.redis.xreadgroup(self._group, consumer, {self._stream: ">"}, count=1,
                                               block=max(1, int(timeout_s * 1000)))
        if not response:
            return None
        _, messages = response[0]
        entry_id, fields = messages[0]
        return self._job(entry_id, fields)

    async def ack(self, job: Job) -> None:
        await self.redis.xack(self._stream, self._group, job.receipt)
        await self.redis.xdel(self._stream, job.receipt)

    async def release(self, job: Job) -> None:
        # Re-publish and drop ours, so another worker can take it without waiting for the claim timeout.
        await self.redis.xadd(self._stream, {"job_id": job.job_id, "mission_id": job.mission_id,
                                             "enqueued_at": str(job.enqueued_at)}, maxlen=self._maxlen, approximate=True)
        await self.ack(job)


class RedisMissionLease:
    def __init__(self, url: str, prefix: str = "devpilot:lease:") -> None:
        self._url = url
        self._prefix = prefix
        self._redis: Redis | None = None

    async def start(self) -> None:
        self._redis = Redis.from_url(self._url, decode_responses=True)
        self._renew = self._redis.register_script(_RENEW)
        self._release = self._redis.register_script(_RELEASE)

    async def stop(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def acquire(self, mission_id: str, owner: str, ttl_s: float) -> bool:
        key = self._prefix + mission_id
        if await self._redis.set(key, owner, nx=True, px=int(ttl_s * 1000)):
            return True
        return await self.renew(mission_id, owner, ttl_s)  # re-entrant for the current holder

    async def renew(self, mission_id: str, owner: str, ttl_s: float) -> bool:
        return bool(await self._renew(keys=[self._prefix + mission_id], args=[owner, int(ttl_s * 1000)]))

    async def release(self, mission_id: str, owner: str) -> None:
        await self._release(keys=[self._prefix + mission_id], args=[owner])

    async def holder(self, mission_id: str) -> str | None:
        return await self._redis.get(self._prefix + mission_id)
