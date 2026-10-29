"""Provider-neutral chat types. Agents only ever see these, never an SDK object."""
from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class ModelTier(StrEnum):
    SMART = "smart"   # planning, coding, review
    FAST = "fast"     # classification, short summaries


class ToolCallRejected(Exception):
    """The provider refused the model's own tool call (e.g. Groq's server-side schema check).

    The request was fine; the generation was not. The agent should see the error and try again,
    not fail over to another provider.
    """

    def __init__(self, message: str, failed_generation: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.failed_generation = failed_generation


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Set when the model sent arguments that were not valid JSON.
    raw_arguments: str | None = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    model: str | None = None
    tier: ModelTier = ModelTier.SMART
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    temperature: float = 0.1
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    # For attribution in the usage ledger (agent name).
    caller: str = "system"


class ChatResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    provider: str = ""
    model: str = ""

    def assistant_message(self) -> dict[str, Any]:
        """This response as an OpenAI-format assistant message, for the next turn."""
        import json

        msg: dict[str, Any] = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments or json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


class LLMProvider(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    async def complete(self, request: ChatRequest, model: str) -> ChatResponse: ...

    async def aclose(self) -> None: ...
