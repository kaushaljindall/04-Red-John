"""
Running missions: who runs a mission, where, and making sure it is only one.

- `MissionExecutor` runs one mission under a lease, renewing it while Atlas
  works. If the lease is lost (this process stalled and another took over),
  the run is cancelled so a mission never has two orchestrators.
- `InlineRunner` runs missions inside the API process (development, a single
  box). `QueueRunner` only enqueues; separate worker processes
  (`python -m app.worker`) pick the jobs up. Same interface, chosen by
  `EXECUTION_MODE`.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
from typing import Protocol

from app.core.errors import ConflictError
from app.core.ids import new_id
from app.infrastructure.jobs.base import JobQueue, MissionLease
from app.services.orchestrator import Orchestrator

log = logging.getLogger(__name__)


def process_identity(kind: str) -> str:
    return f"{kind}:{socket.gethostname()}:{os.getpid()}:{new_id('r')[2:8]}"


class MissionExecutor:
    def __init__(self, orchestrator: Orchestrator, lease: MissionLease, owner: str, *, ttl_s: float = 60.0) -> None:
        self._orchestrator = orchestrator
        self._lease = lease
        self.owner = owner
        self._ttl_s = ttl_s

    async def execute(self, mission_id: str) -> bool:
        """Run the mission if nobody else is. Returns False when another runner holds it."""
        if not await self._lease.acquire(mission_id, self.owner, self._ttl_s):
            return False
        run = asyncio.create_task(self._orchestrator.run(mission_id), name=f"orchestrate:{mission_id}")
        lost = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(mission_id, run, lost))
        try:
            mission = await run
            log.info("mission %s run ended: %s", mission_id, mission.status)
        except asyncio.CancelledError:
            if not lost.is_set():
                raise  # we were cancelled (shutdown): let the caller hand the job back
            log.error("mission %s: lease lost; this run stopped so only one runner remains", mission_id)
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            if not lost.is_set():
                await self._lease.release(mission_id, self.owner)
        return True

    async def _heartbeat(self, mission_id: str, run: asyncio.Task, lost: asyncio.Event) -> None:
        while not run.done():
            await asyncio.sleep(self._ttl_s / 3)
            if not await self._lease.renew(mission_id, self.owner, self._ttl_s):
                lost.set()
                run.cancel()
                return


class MissionRunner(Protocol):
    async def start(self, mission_id: str) -> None: ...

    async def is_running(self, mission_id: str) -> bool: ...

    async def wait(self, mission_id: str) -> None: ...

    async def shutdown(self) -> None: ...


class InlineRunner:
    """Runs missions as background tasks of this process."""

    def __init__(self, executor: MissionExecutor, lease: MissionLease) -> None:
        self._executor = executor
        self._lease = lease
        self._running: dict[str, asyncio.Task] = {}

    async def is_running(self, mission_id: str) -> bool:
        task = self._running.get(mission_id)
        if task is not None and not task.done():
            return True
        return await self._lease.holder(mission_id) is not None

    async def start(self, mission_id: str) -> None:
        if await self.is_running(mission_id):
            raise ConflictError(f"Mission {mission_id} is already running")
        self._running[mission_id] = asyncio.create_task(self._run(mission_id), name=f"mission:{mission_id}")

    async def _run(self, mission_id: str) -> None:
        try:
            if not await self._executor.execute(mission_id):
                log.info("mission %s is already running elsewhere", mission_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("mission %s run crashed", mission_id)
        finally:
            self._running.pop(mission_id, None)

    async def wait(self, mission_id: str) -> None:
        task = self._running.get(mission_id)
        if task is not None:
            await asyncio.shield(task)

    async def shutdown(self) -> None:
        tasks = list(self._running.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


class QueueRunner:
    """Hands missions to worker processes through the job queue."""

    def __init__(self, queue: JobQueue, lease: MissionLease) -> None:
        self._queue = queue
        self._lease = lease

    async def is_running(self, mission_id: str) -> bool:
        return await self._lease.holder(mission_id) is not None

    async def start(self, mission_id: str) -> None:
        if await self.is_running(mission_id):
            raise ConflictError(f"Mission {mission_id} is already running")
        job = await self._queue.enqueue(mission_id)
        log.info("mission %s queued as %s", mission_id, job.job_id)

    async def wait(self, mission_id: str) -> None:
        raise NotImplementedError("queued missions run on workers; follow the event stream instead")

    async def shutdown(self) -> None:
        return None
