"""
DevPilot mission worker — runs queued missions.

    python -m app.worker                 (from backend_main/, with EXECUTION_MODE=queue on the API)

Start as many as you like: the job queue spreads missions across them and
leases make sure each mission runs on exactly one. A worker that dies leaves
its jobs unacknowledged; another worker reclaims them after JOB_CLAIM_AFTER_S
and Atlas resumes the mission from its stored state.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from app.container import Container
from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError
from app.core.logging import configure_logging
from app.domain.mission import MissionStatus
from app.infrastructure.jobs.base import Job
from app.services.runner import MissionExecutor, process_identity

log = logging.getLogger("devpilot.worker")


class MissionWorker:
    def __init__(self, container: Container, *, concurrency: int, worker_id: str | None = None) -> None:
        self.container = container
        self.worker_id = worker_id or process_identity("worker")
        self.executor = MissionExecutor(container.orchestrator, container.lease, self.worker_id,
                                        ttl_s=container.settings.lease_ttl_s)
        self._slots = asyncio.Semaphore(concurrency)
        self._tasks: set[asyncio.Task] = set()
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> None:
        log.info("worker %s started", self.worker_id)
        queue = self.container.job_queue
        while not self._stopping.is_set():
            await self._slots.acquire()
            try:
                job = await queue.dequeue(self.worker_id, timeout_s=2.0)
            except Exception:
                self._slots.release()
                log.exception("dequeue failed; retrying")
                await asyncio.sleep(2)
                continue
            if job is None:
                self._slots.release()
                continue
            task = asyncio.create_task(self._handle(job), name=f"job:{job.job_id}")
            self._tasks.add(task)
            task.add_done_callback(self._done)
        await self._drain()
        log.info("worker %s stopped", self.worker_id)

    def _done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        self._slots.release()

    async def _handle(self, job: Job) -> None:
        queue, board = self.container.job_queue, self.container.board
        try:
            try:
                mission = await board.get(job.mission_id)
            except NotFoundError:
                log.warning("job %s: mission %s does not exist", job.job_id, job.mission_id)
                await queue.ack(job)
                return
            if mission.is_terminal:
                await queue.ack(job)
                return
            if job.deliveries > self.container.settings.worker_max_deliveries:
                # A mission that keeps killing its worker must not take the fleet down with it.
                await board.set_status(job.mission_id, MissionStatus.FAILED, by="worker",
                                       reason=f"run crashed or was interrupted {job.deliveries - 1} time(s)")
                await queue.ack(job)
                return
            ran = await self.executor.execute(job.mission_id)
            if not ran:
                log.info("job %s: mission %s is running on another worker", job.job_id, job.mission_id)
            await queue.ack(job)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await queue.release(job)  # shutting down: another worker resumes it
            raise
        except Exception:
            log.exception("job %s for mission %s crashed; it will be retried", job.job_id, job.mission_id)
            with contextlib.suppress(Exception):
                await queue.release(job)

    async def _drain(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def main(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    container = Container.build(settings)
    await container.start()
    worker = MissionWorker(container, concurrency=settings.worker_concurrency)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # not supported on Windows; Ctrl+C still stops it
            loop.add_signal_handler(sig, worker.stop)
    try:
        await worker.run()
    finally:
        await container.stop()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
