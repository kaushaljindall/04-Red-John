"""
OpenAI-compatible chat provider with an API-key pool.

Ported from Mycel's `RobustGroqClient`: on a 429 the key is benched for the
time the provider asks for and the next healthy key is tried at once; dead
keys (401/403) are benched for a day. Works for Groq, Gemini's OpenAI
endpoint, OpenAI, or a local server — anything speaking /chat/completions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass

import httpx

from app.core.errors import ExternalServiceError
from app.infrastructure.llm.types import ChatRequest, ChatResponse, ContextTooLarge, ToolCall, ToolCallRejected, Usage

log = logging.getLogger(__name__)

_RETRY_AFTER = re.compile(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)
# Groq's tool-use models sometimes append channel markers to tool names.
_TOOL_NAME_JUNK = re.compile(r"<\|.*$")


@dataclass
class _Key:
    secret: str
    cooldown_until: float = 0.0

    @property
    def preview(self) -> str:
        return f"{self.secret[:6]}…"


class OpenAICompatibleProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_keys: list[str],
        *,
        timeout_s: float = 90.0,
        max_attempts: int = 12,
        max_cooldown_wait_s: float = 20.0,
        rate_limit_budget_s: float = 90.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        # Longer than this and every key is effectively out: fail so the gateway can try another provider.
        self._max_cooldown_wait_s = max_cooldown_wait_s
        self._rate_limit_budget_s = rate_limit_budget_s
        self._keys = [_Key(k) for k in api_keys if k]
        self._max_attempts = max_attempts
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport
        )

    @property
    def available(self) -> bool:
        return bool(self._keys)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _healthy_key(self) -> _Key:
        now = time.monotonic()
        healthy = [k for k in self._keys if k.cooldown_until <= now]
        if healthy:
            # Scatter traffic so concurrent agents do not hammer one key.
            return random.choice(healthy)
        soonest = min(self._keys, key=lambda k: k.cooldown_until)
        wait = soonest.cooldown_until - now
        if wait > self._max_cooldown_wait_s:
            raise ExternalServiceError(
                f"{self.name}: all {len(self._keys)} key(s) are rate limited or rejected for another {wait:.0f}s")
        log.info("[%s] all %d keys cooling down; waiting %.1fs", self.name, len(self._keys), wait)
        await asyncio.sleep(wait)
        return soonest

    @staticmethod
    def _retry_after(response: httpx.Response) -> float:
        header = response.headers.get("retry-after")
        if header:
            try:
                return float(header) + 0.5
            except ValueError:
                pass
        match = _RETRY_AFTER.search(response.text)
        if match:
            minutes = float(match.group(1) or 0)
            return minutes * 60 + float(match.group(2)) + 0.5
        return 10.0

    @staticmethod
    def _tool_call_rejection(response: httpx.Response) -> ToolCallRejected | None:
        if response.status_code != 400:
            return None
        try:
            error = response.json().get("error") or {}
        except ValueError:
            return None
        # Both mean the model's own output was malformed, not the request: the agent should try again.
        if error.get("code") not in ("tool_use_failed", "output_parse_failed"):
            return None
        return ToolCallRejected(str(error.get("message", "tool call rejected"))[:800],
                                str(error.get("failed_generation", ""))[:800])

    def _payload(self, request: ChatRequest, model: str) -> dict:
        body: dict = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.tools:
            body["tools"] = request.tools
            body["tool_choice"] = request.tool_choice or "auto"
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.response_format:
            body["response_format"] = request.response_format
        return body

    @staticmethod
    def _parse(data: dict, provider: str, model: str) -> ChatResponse:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        calls: list[ToolCall] = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            args_text = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_text) if isinstance(args_text, str) else dict(args_text)
                bad = None
            except ValueError:
                args, bad = {}, args_text
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{len(calls)}",
                    name=_TOOL_NAME_JUNK.sub("", fn.get("name", "")).strip(),
                    arguments=args if isinstance(args, dict) else {},
                    raw_arguments=bad,
                )
            )
        usage = data.get("usage") or {}
        return ChatResponse(
            content=_THINK.sub("", message.get("content") or "").strip(),
            tool_calls=calls,
            finish_reason=choice.get("finish_reason"),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
            provider=provider,
            model=data.get("model", model),
        )

    async def complete(self, request: ChatRequest, model: str) -> ChatResponse:
        if not self._keys:
            raise ExternalServiceError(f"{self.name}: no API keys configured")
        payload = self._payload(request, model)
        last_error = ""
        # Rate limits are bounded by total waiting time, not by attempts: with a big key pool a burst of
        # 429s with short cooldowns is normal and must not look like an outage.
        attempt, rate_limited_since = 0, None
        while attempt < self._max_attempts:
            if rate_limited_since is not None and time.monotonic() - rate_limited_since > self._rate_limit_budget_s:
                raise ExternalServiceError(f"{self.name}: still rate limited after {self._rate_limit_budget_s:.0f}s")
            key = await self._healthy_key()
            try:
                response = await self._http.post(
                    "/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key.secret}"},
                )
            except httpx.TransportError as exc:
                attempt += 1
                last_error = f"transport error: {exc}"
                await asyncio.sleep(min(2**attempt, 10))
                continue

            if response.status_code == 200:
                return self._parse(response.json(), self.name, model)
            if response.status_code == 429:
                wait = self._retry_after(response)
                key.cooldown_until = time.monotonic() + wait
                rate_limited_since = rate_limited_since or time.monotonic()
                log.info("[%s] key %s rate limited; benched %.1fs", self.name, key.preview, wait)
                last_error = "rate limited"
                continue
            attempt += 1
            if response.status_code in (401, 403):
                key.cooldown_until = time.monotonic() + 86_400
                log.error("[%s] key %s rejected (%s); benched", self.name, key.preview, response.status_code)
                last_error = f"auth error {response.status_code}"
                continue
            if response.status_code >= 500:
                last_error = f"server error {response.status_code}"
                await asyncio.sleep(min(2**attempt, 10))
                continue
            if response.status_code == 413 or "context_length_exceeded" in response.text:
                raise ContextTooLarge(f"{self.name}: request too large — {response.text[:300]}")
            rejected = self._tool_call_rejection(response)
            if rejected is not None:
                raise rejected
            # 400-class: the request itself is wrong; retrying will not help.
            raise ExternalServiceError(
                f"{self.name} rejected the request ({response.status_code})",
                details={"body": response.text[:1000]},
            )
        raise ExternalServiceError(
            f"{self.name}: gave up after {self._max_attempts} attempts ({last_error})"
        )
