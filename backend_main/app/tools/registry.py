"""
Tool registry: declaration, schema export, argument repair, policy gating,
execution, and events. Ported from Jarvis `tools/registry.py`.

The registry — not the model — decides whether a call runs, needs a human,
or is refused. Agents only see the tools they were granted.
"""
from __future__ import annotations

import asyncio
import difflib
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.core.config import Settings
from app.core.errors import PolicyViolation
from app.domain.events import EventKind
from app.security.policy import Risk, Verdict
from app.security.sandbox import Workspace

log = logging.getLogger(__name__)

MAX_MODEL_TEXT = 12_000


@dataclass
class ToolResult:
    ok: bool
    content: Any = None
    error: str = ""
    display: str = ""
    # Set by tools whose output proves something (test runs, diffs, scans).
    evidence: dict[str, Any] | None = None
    duration_s: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_model_text(self, limit: int = MAX_MODEL_TEXT) -> str:
        """What the LLM sees as the tool message."""
        evidence_id = self.meta.get("evidence_id")
        prefix = f"[evidence_id: {evidence_id}]\n" if evidence_id else ""
        if self.ok and isinstance(self.content, str):
            body_text = self.content
            room = limit - len(prefix)
            if len(body_text) > room:
                body_text = body_text[: max(0, room - 60)] + f"\n… [truncated {len(body_text) - room + 60} chars]"
            return prefix + body_text
        body: dict[str, Any] = {"ok": self.ok}
        if evidence_id:
            body["evidence_id"] = evidence_id  # first, so truncation can never cut it off
        if not self.ok:
            body["error"] = self.error[: min(1500, limit // 3)]
        if self.content is not None:
            body["content"] = self.content
        text = json.dumps(body, default=str, indent=1)
        if len(text) > limit and self.content is not None:
            # Shorten the bulky part (output, file lists), never the verdict or the ids.
            inner = self.content if isinstance(self.content, str) else json.dumps(self.content, default=str, indent=1)
            keep = max(200, limit - (len(text) - len(inner)) - 80)
            body["content"] = inner[:keep] + f"… [truncated {len(inner) - keep} chars]"
            text = json.dumps(body, default=str, indent=1)
        if len(text) > limit:
            text = text[: limit - 60] + f"\n… [truncated {len(text) - limit + 60} chars]"
        return text


class Approver(Protocol):
    async def request(self, ctx: ToolContext, tool: str, args: dict[str, Any], verdict: Verdict) -> bool: ...


class DenyAllApprover:
    """Default until the approval gate (Phase 3) is wired: AMBER actions are refused."""

    async def request(self, ctx: ToolContext, tool: str, args: dict[str, Any], verdict: Verdict) -> bool:
        return False


class AllowAllApprover:
    """For tests and trusted local runs only."""

    async def request(self, ctx: ToolContext, tool: str, args: dict[str, Any], verdict: Verdict) -> bool:
        return True


class EventSink(Protocol):
    async def publish(self, mission_id: str, kind: EventKind, data: dict[str, Any] | None = None) -> Any: ...


@dataclass
class ToolContext:
    mission_id: str
    agent: str
    workspace: Workspace
    settings: Settings
    approver: Approver = field(default_factory=DenyAllApprover)
    events: EventSink | None = None
    # The task this agent is working on; evidence it produces is linked to it.
    task_id: str | None = None
    # Set by the agent runtime so board tools (messages, findings, verification) can reach the mission board.
    board: Any = None
    teams: set[str] = field(default_factory=set)
    inbox_seq: int = 0


Handler = Callable[..., Awaitable[ToolResult] | ToolResult]
Assessor = Callable[[ToolContext, dict[str, Any]], Verdict]


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    risk: Risk = Risk.SAFE
    category: str = "read"
    assess: Assessor | None = None
    evidence: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description,
                         "parameters": nullable_optionals(self.parameters)},
        }


def nullable_optionals(parameters: dict[str, Any]) -> dict[str, Any]:
    """Let optional arguments be null in the exported schema.

    Models often send `"cwd": null` for an argument they mean to leave out, and
    providers that validate tool calls server-side (Groq) then reject the whole
    response. Accepting null — and treating it as "not given" in normalize_args —
    removes that failure mode without weakening required arguments.
    """
    required = set(parameters.get("required") or [])
    props = {}
    for name, spec in (parameters.get("properties") or {}).items():
        spec = dict(spec)
        if name not in required:
            kind = spec.get("type")
            if isinstance(kind, str):
                spec["type"] = [kind, "null"]
            if "enum" in spec and None not in spec["enum"]:
                spec["enum"] = [*spec["enum"], None]
        props[name] = spec
    return {**parameters, "properties": props}


# ── argument repair (from Jarvis) ───────────────────────────────────────────
_ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("file", "filepath", "file_path", "filename", "location", "dir", "directory", "folder"),
    "content": ("text", "body", "data", "contents"),
    "old": ("old_str", "old_string", "search", "find", "original"),
    "new": ("new_str", "new_string", "replace", "replacement"),
    "query": ("q", "search", "pattern", "term", "keyword", "regex"),
    "limit": ("max_results", "max_matches", "max_lines", "num_results", "top_k", "count", "max"),
    "offset": ("line_start", "start_line", "start", "from_line", "line"),
    "command": ("cmd", "argv", "args"),
    "pattern": ("glob", "name", "filename"),
}


def _closest(word: str, options: Iterable[str]) -> str | None:
    hit = difflib.get_close_matches(word, list(options), n=1, cutoff=0.72)
    return hit[0] if hit else None


def _coerce(value: Any, spec: dict[str, Any], where: str) -> Any:
    """Make `value` fit a JSON-schema type, or raise ValueError saying why."""
    kind = spec.get("type")
    if "enum" in spec:
        options = spec["enum"]
        if value in options:
            return value
        lowered = {str(o).lower(): o for o in options}
        if str(value).lower() in lowered:
            return lowered[str(value).lower()]
        raise ValueError(f"{where} must be one of {options}, not {value!r}")
    if kind == "integer":
        if isinstance(value, bool):
            raise ValueError(f"{where} must be a whole number")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                number = float(value.strip())
            except ValueError:
                pass
            else:
                if number.is_integer():
                    return int(number)
        raise ValueError(f"{where} must be a whole number, not {value!r}")
    if kind == "number":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        try:
            return float(str(value).strip())
        except ValueError:
            raise ValueError(f"{where} must be a number, not {value!r}") from None
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "yes", "1", "on"):
            return True
        if text in ("false", "no", "0", "off", ""):
            return False
        raise ValueError(f"{where} must be true or false, not {value!r}")
    if kind == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
            return value[0]
        raise ValueError(f"{where} must be text, not {type(value).__name__}")
    if kind == "array":
        if isinstance(value, list):
            items = value
        elif isinstance(value, str) and value.strip().startswith("["):
            try:
                items = json.loads(value)
            except ValueError:
                items = [value]
        else:
            items = [value]
        item_spec = spec.get("items") or {}
        return [_coerce(x, item_spec, f"{where}[{i}]") if item_spec else x for i, x in enumerate(items)]
    if kind == "object":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        raise ValueError(f"{where} must be an object")
    return value


def normalize_args(spec: ToolSpec, args: Any) -> tuple[dict[str, Any], list[str]]:
    """Check a call against the tool's schema. Returns (clean_args, repair_notes)."""
    if args is None:
        args = {}
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except ValueError:
            raise ValueError("arguments must be a JSON object") from None
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")

    props: dict[str, Any] = spec.parameters.get("properties") or {}
    required: list[str] = list(spec.parameters.get("required") or [])
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key, value in args.items():
        if value is None:
            continue  # null means "not given"; a missing required argument is reported below
        if key in props:
            out[key] = value
            continue
        target = next(
            (p for p, aliases in _ARG_ALIASES.items() if p in props and key.lower() in aliases and p not in args),
            None,
        ) or _closest(key, [p for p in props if p not in args])
        if target and target not in out:
            out[target] = value
            notes.append(f"treated '{key}' as '{target}'")
        else:
            raise ValueError(
                f"'{spec.name}' has no argument '{key}'. Its arguments are: {', '.join(sorted(props)) or 'none'}"
            )
    for key in list(out):
        out[key] = _coerce(out[key], props[key], f"'{key}'")
    missing = [
        r for r in required
        if r not in out or (props.get(r, {}).get("type") == "string" and isinstance(out[r], str) and not out[r].strip())
    ]
    if missing:
        raise ValueError(f"missing required argument(s) for '{spec.name}': {', '.join(missing)}")
    return out, notes


# ── registry ────────────────────────────────────────────────────────────────
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} registered twice")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        *,
        risk: Risk = Risk.SAFE,
        category: str = "read",
        assess: Assessor | None = None,
        evidence: bool = False,
    ) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self.register(ToolSpec(name, description, parameters, fn, risk, category, assess, evidence))
            return fn

        return decorator

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "risk": t.risk.name, "category": t.category,
             "evidence": t.evidence, "parameters": t.parameters}
            for t in sorted(self._tools.values(), key=lambda t: t.name)
        ]

    def schemas(self, only: Iterable[str] | None = None) -> list[dict[str, Any]]:
        wanted = set(only) if only is not None else None
        return [t.schema() for n, t in sorted(self._tools.items()) if wanted is None or n in wanted]

    def assess(self, ctx: ToolContext, spec: ToolSpec, args: dict[str, Any]) -> Verdict:
        if spec.assess is not None:
            return spec.assess(ctx, args)
        return Verdict(spec.risk, f"{spec.name} ({spec.category})", spec.category)

    async def _emit(self, ctx: ToolContext, kind: EventKind, data: dict[str, Any]) -> None:
        if ctx.events is None:
            return
        try:
            await ctx.events.publish(ctx.mission_id, kind, data)
        except Exception:
            log.exception("failed to publish %s for %s", kind, data.get("tool"))

    async def execute(
        self, name: str, args: Any, ctx: ToolContext, *, allowed: Iterable[str] | None = None
    ) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None or (allowed is not None and name not in set(allowed)):
            pool = set(allowed) if allowed is not None else set(self._tools)
            hint = _closest(name, pool)
            return ToolResult(False, error=f"Unknown tool '{name}'." + (f" Did you mean '{hint}'?" if hint else ""),
                              meta={"outcome": "invalid"})

        try:
            clean, notes = normalize_args(spec, args)
        except ValueError as exc:
            return ToolResult(False, error=f"Invalid arguments: {exc}", meta={"outcome": "invalid"})

        try:
            verdict = self.assess(ctx, spec, clean)
        except PolicyViolation as exc:
            verdict = Verdict(Risk.FORBIDDEN, exc.message, "sandbox")

        base = {"agent": ctx.agent, "tool": name, "args": preview_args(clean), "class": verdict.action_class.value}
        meta: dict[str, Any] = {"action_class": verdict.action_class.value, "args": base["args"]}
        await self._emit(ctx, EventKind.TOOL_CALL, {**base, "reason": verdict.reason})

        if not verdict.allowed:
            result = ToolResult(False, error=f"Blocked by policy: {verdict.reason}",
                                meta={**meta, "verdict": verdict.describe(), "outcome": "blocked"})
            await self._emit(ctx, EventKind.TOOL_RESULT, {**base, "ok": False, "blocked": True, "error": result.error})
            return result
        if verdict.needs_approval:
            approved = await ctx.approver.request(ctx, name, clean, verdict)
            if not approved:
                result = ToolResult(False, error=f"Human approval was not granted: {verdict.reason}",
                                    meta={**meta, "verdict": verdict.describe(), "denied": True, "outcome": "denied"})
                await self._emit(ctx, EventKind.TOOL_RESULT, {**base, "ok": False, "denied": True, "error": result.error})
                return result
            meta["approved_by_human"] = True

        started = time.monotonic()
        try:
            outcome = spec.handler(ctx, **clean)
            result = await outcome if inspect.isawaitable(outcome) else outcome
        except PolicyViolation as exc:
            result = ToolResult(False, error=f"Blocked by policy: {exc.message}",
                                meta={**exc.details, "outcome": "failed" if exc.details.get("tool_unavailable") else "blocked"})
        except Exception as exc:  # a tool bug must not kill the agent loop
            log.exception("tool %s crashed", name)
            result = ToolResult(False, error=f"{type(exc).__name__}: {exc}")
        result.duration_s = round(time.monotonic() - started, 3)
        result.meta = {**meta, "outcome": "ok" if result.ok else "failed", **result.meta}
        if notes:
            result.meta.setdefault("repairs", notes)

        await self._emit(ctx, EventKind.TOOL_RESULT, {
            **base, "ok": result.ok, "display": result.display, "error": result.error[:500],
            "duration_s": result.duration_s,
        })
        # Evidence is stored (and announced) by the Governor, which owns the mission board.
        return result


def preview_args(args: dict[str, Any], limit: int = 300) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        out[key] = text if len(text) <= limit else text[:limit] + "…"
    return out


async def run_blocking(fn: Callable[..., Any], *args: Any) -> Any:
    return await asyncio.to_thread(fn, *args)
