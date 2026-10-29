"""
Job queue and mission leases: how missions run on workers at scale.

- The queue carries "run this mission" jobs from the API to any number of
  workers. A job is acknowledged only when its run ends; if a worker dies,
  the job is handed to another worker after `claim_after_s`.
- A lease makes sure exactly one worker runs a mission at a time, even if
  the same mission is queued twice. The holder renews it while it works; a
  dead holder's lease simply expires.
"""
from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field

from app.core.ids import new_id, now_ms


class Job(BaseModel):
    job_id: str = Field(default_factory=lambda: new_id("job"))
    mission_id: str
    enqueued_at: int = Field(default_factory=now_ms)
    deliveries: int = 1           # how many times a worker has received this job
    receipt: str = ""             # backend handle used to acknowledge it


class JobQueue(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def enqueue(self, mission_id: str) -> Job: ...

    async def dequeue(self, consumer: str, timeout_s: float) -> Job | None:
        """Next job for this consumer, including jobs abandoned by dead consumers. None on timeout."""
        ...

    async def ack(self, job: Job) -> None:
        """The run ended (in any state); the job is done."""
        ...

    async def release(self, job: Job) -> None:
        """Give the job back right away (e.g. this worker is shutting down)."""
        ...


class MissionLease(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def acquire(self, mission_id: str, owner: str, ttl_s: float) -> bool: ...

    async def renew(self, mission_id: str, owner: str, ttl_s: float) -> bool:
        """Extend the lease if `owner` still holds it. False means it was lost."""
        ...

    async def release(self, mission_id: str, owner: str) -> None: ...

    async def holder(self, mission_id: str) -> str | None: ...
