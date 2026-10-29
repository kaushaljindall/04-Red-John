"""FastAPI dependencies. Routes take what they need from the container here."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request, WebSocket

from app.container import Container


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_ws_container(websocket: WebSocket) -> Container:
    return websocket.app.state.container


ContainerDep = Annotated[Container, Depends(get_container)]
WsContainerDep = Annotated[Container, Depends(get_ws_container)]
