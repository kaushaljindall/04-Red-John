import asyncio

import httpx
import pytest

from app.core.errors import ExternalServiceError
from app.infrastructure.llm.gateway import LLMGateway, Route
from app.infrastructure.llm.openai_compat import OpenAICompatibleProvider
from app.infrastructure.llm.types import ChatRequest, ModelTier


def _ok(content="", tool_calls=None):
    return httpx.Response(200, json={
        "model": "m",
        "choices": [{"message": {"content": content, "tool_calls": tool_calls}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    })


async def test_rate_limited_key_is_benched_and_next_key_used():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["authorization"]
        seen.append(key)
        if key == "Bearer key-a":
            return httpx.Response(429, text="Rate limit reached. Please try again in 7.5s.")
        return _ok("hello")

    provider = OpenAICompatibleProvider("groq", "https://x", ["key-a", "key-b"], transport=httpx.MockTransport(handler))
    for _ in range(3):
        response = await provider.complete(ChatRequest(messages=[{"role": "user", "content": "hi"}]), "m")
        assert response.content == "hello"
    # key-a is tried at most once; after its 429 it stays benched.
    assert seen.count("Bearer key-a") <= 1


async def test_tool_calls_are_parsed_and_cleaned():
    calls = [{"id": "c1", "type": "function",
              "function": {"name": "read_file<|channel|>commentary", "arguments": '{"path": "a.py"}'}},
             {"id": "c2", "type": "function", "function": {"name": "run_tests", "arguments": "{not json"}}]
    provider = OpenAICompatibleProvider("groq", "https://x", ["k"], transport=httpx.MockTransport(lambda r: _ok("<think>x</think>ok", calls)))
    response = await provider.complete(ChatRequest(messages=[]), "m")
    assert response.content == "ok"
    assert response.tool_calls[0].name == "read_file" and response.tool_calls[0].arguments == {"path": "a.py"}
    assert response.tool_calls[1].raw_arguments == "{not json"
    assert response.assistant_message()["tool_calls"][0]["function"]["name"] == "read_file"


async def test_gateway_falls_back_and_records_usage():
    broken = OpenAICompatibleProvider("groq", "https://x", ["k"],
                                      transport=httpx.MockTransport(lambda r: httpx.Response(400, text="bad")))
    backup = OpenAICompatibleProvider("gemini", "https://y", ["k"], transport=httpx.MockTransport(lambda r: _ok("from backup")))
    gateway = LLMGateway({ModelTier.SMART: [Route(broken, "a"), Route(backup, "b")]})
    response = await gateway.complete(ChatRequest(messages=[], caller="Atlas"))
    assert response.provider == "gemini" and response.content == "from backup"
    assert gateway.ledger.tokens_by_caller["Atlas"] == 15


async def test_gateway_without_keys_fails_clearly():
    gateway = LLMGateway({ModelTier.SMART: [Route(OpenAICompatibleProvider("groq", "https://x", []), "a")]})
    assert not gateway.available
    with pytest.raises(ExternalServiceError):
        await gateway.complete(ChatRequest(messages=[]))


async def test_groq_tool_use_failed_is_a_rejection_not_an_outage():
    from app.infrastructure.llm.types import ToolCallRejected

    body = {"error": {"message": "Tool call validation failed", "code": "tool_use_failed",
                      "failed_generation": '{"name": "run_tests", "arguments": {"command": null}}'}}
    provider = OpenAICompatibleProvider("groq", "https://x", ["k"],
                                        transport=httpx.MockTransport(lambda r: httpx.Response(400, json=body)))
    backup = OpenAICompatibleProvider("gemini", "https://y", ["k"], transport=httpx.MockTransport(lambda r: _ok("x")))
    gateway = LLMGateway({ModelTier.SMART: [Route(provider, "a"), Route(backup, "b")]})
    with pytest.raises(ToolCallRejected) as caught:
        await gateway.complete(ChatRequest(messages=[]))
    assert "command" in caught.value.failed_generation  # no silent fail-over: the agent must correct itself


async def test_long_rate_limits_fail_fast_so_the_gateway_can_fall_back():
    limited = OpenAICompatibleProvider(
        "gemini", "https://x", ["k1", "k2"], max_cooldown_wait_s=5,
        transport=httpx.MockTransport(lambda r: httpx.Response(429, text="Please try again in 55s.")))
    backup = OpenAICompatibleProvider("groq", "https://y", ["k"], transport=httpx.MockTransport(lambda r: _ok("served")))
    gateway = LLMGateway({ModelTier.SMART: [Route(limited, "a"), Route(backup, "b")]})
    response = await asyncio.wait_for(gateway.complete(ChatRequest(messages=[])), timeout=5)
    assert response.content == "served"
