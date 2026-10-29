"""
Mission event stream over WebSocket. Ported from Mycel's realtime router.

On connect the client gets every stored event after `after_seq` (replay),
then the live tail from the bus. Because events come from the store and the
bus — not from an in-process connection list — any API replica can serve
any mission, whichever worker is running it.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.api.deps import ContainerDep, WsContainerDep
from app.domain.events import mission_topic

router = APIRouter(prefix="/realtime", tags=["realtime"])
log = logging.getLogger(__name__)


@router.get("/missions/{mission_id}/events")
async def list_events(
    mission_id: str, container: ContainerDep, after_seq: int = 0, limit: int = Query(1000, le=5000)
) -> dict[str, Any]:
    events = await container.storage.events.list(mission_id, after_seq=after_seq, limit=limit)
    return {"mission_id": mission_id, "events": [e.model_dump(mode="json") for e in events]}


async def _watch_disconnect(websocket: WebSocket) -> None:
    """Clients do not send anything; this only returns when they leave."""
    with contextlib.suppress(WebSocketDisconnect):
        while True:
            await websocket.receive_text()


@router.websocket("/missions/{mission_id}")
@router.websocket("/sessions/{mission_id}")  # Mycel control-room path
async def mission_stream(websocket: WebSocket, mission_id: str, container: WsContainerDep, after_seq: int = 0) -> None:
    await websocket.accept()
    watcher = asyncio.create_task(_watch_disconnect(websocket))
    try:
        # Subscribe first, then replay: anything published in between is in the
        # subscription and is skipped below by seq if the replay already sent it.
        async with container.event_bus.subscribe(mission_topic(mission_id)) as live:
            last_seq = after_seq
            for event in await container.storage.events.list(mission_id, after_seq=after_seq):
                await websocket.send_json(event.model_dump(mode="json"))
                last_seq = event.seq or last_seq

            while not watcher.done():
                getter = asyncio.ensure_future(anext(live))
                done, _ = await asyncio.wait({getter, watcher}, return_when=asyncio.FIRST_COMPLETED)
                if getter not in done:
                    getter.cancel()
                    break
                message = getter.result()
                seq = message.get("seq") or 0
                if seq and seq <= last_seq:
                    continue
                last_seq = max(last_seq, seq)
                await websocket.send_json(message)
    except (WebSocketDisconnect, RuntimeError):
        pass  # client went away mid-send
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
