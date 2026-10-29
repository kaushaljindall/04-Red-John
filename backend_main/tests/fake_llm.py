"""A scripted LLM provider: replays prepared responses and records every request it gets."""
from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from app.core.errors import ExternalServiceError
from app.infrastructure.llm.gateway import LLMGateway, Route
from app.infrastructure.llm.types import ChatRequest, ChatResponse, ModelTier, ToolCall, Usage

_ids = itertools.count(1)

Step = ChatResponse | Callable[[ChatRequest], ChatResponse] | Exception


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id=f"call_{next(_ids)}", name=name, arguments=arguments)


def reply(*calls: ToolCall, content: str = "", tokens: int = 100) -> ChatResponse:
    return ChatResponse(content=content, tool_calls=list(calls), usage=Usage(prompt_tokens=tokens, completion_tokens=0),
                        provider="fake", model="fake")


def finish(status: str = "done", summary: str = "done", **data: Any) -> ChatResponse:
    return reply(call("finish", status=status, summary=summary, data=data))


class ScriptedProvider:
    name = "fake"

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.requests: list[ChatRequest] = []

    @property
    def available(self) -> bool:
        return True

    async def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        self.requests.append(request.model_copy(deep=True))
        if not self.steps:
            raise ExternalServiceError("script exhausted")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(request) if callable(step) else step

    async def aclose(self) -> None:
        return None

    def last_tool_results(self) -> list[str]:
        """Tool messages the agent received just before the latest request."""
        messages = self.requests[-1].messages
        out = []
        for message in reversed(messages):
            if message.get("role") != "tool":
                break
            out.append(message["content"])
        return out[::-1]


def gateway(provider: ScriptedProvider) -> LLMGateway:
    return LLMGateway({tier: [Route(provider, "fake")] for tier in ModelTier})
