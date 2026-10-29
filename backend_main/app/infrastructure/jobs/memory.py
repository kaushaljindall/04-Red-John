"""In-process job queue and leases, for a single process (dev, tests, inline mode)."""
from __future__ import annotations

import asyncio
import time

from app.infrastructure.jobs.base import Job


class InMemoryJobQueue:
    def __init__(self, claim_after_s: float = 120.0) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._in_flight: dict[str, tuple[Job, str, float]] = {}  # job_id -> (job, consumer, delivered_at)
        self._claim_after_s = claim_after_s

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def enqueue(self, mission_id: str) -> Job:
        job = Job(mission_id=mission_id)
        await self._queue.put(job)
        return job

    def _reclaim(self, consumer: str) -> Job | None:
        now = time.monotonic()
        for job_id, (job, owner, delivered) in list(self._in_flight.items()):
            if owner != consumer and now - delivered > self._claim_after_s:
                job = job.model_copy(update={"deliveries": job.deliveries + 1})
                self._in_flight[job_id] = (job, consumer, now)
                return job
        return None

    async def dequeue(self, consumer: str, timeout_s: float) -> Job | None:
        reclaimed = self._reclaim(consumer)
        if reclaimed:
            return reclaimed
        try:
            job = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
        except TimeoutError:
            return None
        self._in_flight[job.job_id] = (job, consumer, time.monotonic())
        return job

    async def ack(self, job: Job) -> None:
        self._in_flight.pop(job.job_id, None)

    async def release(self, job: Job) -> None:
        if self._in_flight.pop(job.job_id, None) is not None:
            await self._queue.put(job.model_copy(update={"deliveries": job.deliveries + 1}))


class InMemoryMissionLease:
    def __init__(self) -> None:
        self._leases: dict[str, tuple[str, float]] = {}

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    def _live(self, mission_id: str) -> tuple[str, float] | None:
        lease = self._leases.get(mission_id)
        if lease and lease[1] > time.monotonic():
            return lease
        self._leases.pop(mission_id, None)
        return None

    async def acquire(self, mission_id: str, owner: str, ttl_s: float) -> bool:
        lease = self._live(mission_id)
        if lease and lease[0] != owner:
            return False
        self._leases[mission_id] = (owner, time.monotonic() + ttl_s)
        return True

    async def renew(self, mission_id: str, owner: str, ttl_s: float) -> bool:
        lease = self._live(mission_id)
        if not lease or lease[0] != owner:
            return False
        self._leases[mission_id] = (owner, time.monotonic() + ttl_s)
        return True

    async def release(self, mission_id: str, owner: str) -> None:
        lease = self._live(mission_id)
        if lease and lease[0] == owner:
            self._leases.pop(mission_id, None)

    async def holder(self, mission_id: str) -> str | None:
        lease = self._live(mission_id)
        return lease[0] if lease else None
