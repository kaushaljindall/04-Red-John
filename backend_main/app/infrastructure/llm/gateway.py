"""
LLM gateway: the one door every agent uses to reach a model.

- Routes a tier (smart / fast) to a provider + model.
- Falls back to the next provider when one fails or has no keys.
- Caps concurrent in-flight calls (protects rate limits across all agents).
- Records token usage per caller, which feeds mission budgets.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass

from app.core.errors import ExternalServiceError
from app.infrastructure.llm.types import ChatRequest, ChatResponse, LLMProvider, ModelTier

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Route:
    provider: LLMProvider
    model: str


class UsageLedger:
    def __init__(self) -> None:
        self.tokens_by_caller: dict[str, int] = defaultdict(int)
        self.calls_by_caller: dict[str, int] = defaultdict(int)

    def record(self, caller: str, response: ChatResponse) -> None:
        self.tokens_by_caller[caller] += response.usage.total
        self.calls_by_caller[caller] += 1

    @property
    def total_tokens(self) -> int:
        return sum(self.tokens_by_caller.values())


class LLMGateway:
    def __init__(self, routes: dict[ModelTier, list[Route]], max_concurrency: int = 4) -> None:
        self._routes = routes
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.ledger = UsageLedger()

    @property
    def available(self) -> bool:
        return any(r.provider.available for chain in self._routes.values() for r in chain)

    def describe(self) -> dict[str, list[str]]:
        return {
            tier.value: [f"{r.provider.name}:{r.model}" for r in chain if r.provider.available]
            for tier, chain in self._routes.items()
        }

    async def complete(self, request: ChatRequest) -> ChatResponse:
        chain = self._routes.get(request.tier) or []
        if request.model:
            chain = [Route(r.provider, request.model) for r in chain[:1]] + chain[1:]
        errors: list[str] = []
        for route in chain:
            if not route.provider.available:
                continue
            try:
                async with self._semaphore:
                    response = await route.provider.complete(request, route.model)
            except ExternalServiceError as exc:
                log.warning("llm route %s:%s failed: %s %s", route.provider.name, route.model, exc.message,
                            str(exc.details.get("body", ""))[:300])
                errors.append(f"{route.provider.name}: {exc.message}")
                continue
            self.ledger.record(request.caller, response)
            return response
        raise ExternalServiceError(
            "No LLM provider could serve the request", details={"attempts": errors}
        )

    async def aclose(self) -> None:
        seen: set[int] = set()
        for chain in self._routes.values():
            for route in chain:
                if id(route.provider) not in seen:
                    seen.add(id(route.provider))
                    await route.provider.aclose()
