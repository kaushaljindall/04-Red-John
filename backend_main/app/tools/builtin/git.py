"""
Read-only git tools over the mission workspace. Diffs are always against
the DevPilot baseline commit, so they show exactly what the agents changed.
"""
from __future__ import annotations

import re

from app.security.process import run_process, truncate_middle
from app.tools.registry import ToolContext, ToolRegistry, ToolResult

_STAT = re.compile(r"(\d+) files? changed(?:, (\d+) insertions?\(\+\))?(?:, (\d+) deletions?\(-\))?")


async def _git(ctx: ToolContext, *args: str) -> tuple[bool, str]:
    result = await run_process(["git", "-c", "core.quotepath=off", *args], ctx.workspace.root,
                               timeout_s=60, max_output=ctx.settings.command_max_output_chars)
    return result.ok, result.stdout if result.ok else result.failure_reason()


async def changed_files(ctx: ToolContext) -> list[str]:
    """Files changed since the baseline, including new untracked files."""
    await _git(ctx, "add", "--intent-to-add", "--all")
    ok, out = await _git(ctx, "diff", "--name-only", ctx.workspace.baseline_commit)
    return [line for line in out.splitlines() if line.strip()] if ok else []


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "git_status",
        "Show which files are modified, added or deleted in the workspace.",
        {"type": "object", "properties": {}},
        category="read")
    async def git_status(ctx: ToolContext) -> ToolResult:
        ok, out = await _git(ctx, "status", "--porcelain=v1", "--untracked-files=all")
        if not ok:
            return ToolResult(False, error=out)
        lines = [line for line in out.splitlines() if line.strip()]
        return ToolResult(True, content={"entries": lines, "clean": not lines}, display=f"{len(lines)} changed path(s)")

    @registry.tool(
        "git_diff",
        "Show the changes made in this mission (against the baseline), including new files. "
        "Use stat_only for a summary, or path to limit to one file or folder.",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "Limit the diff to this path"},
            "stat_only": {"type": "boolean", "description": "Only list files and line counts"}}},
        category="read", evidence=True)
    async def git_diff(ctx: ToolContext, path: str = "", stat_only: bool = False) -> ToolResult:
        await _git(ctx, "add", "--intent-to-add", "--all")
        base = ctx.workspace.baseline_commit
        scope = ["--", ctx.workspace.relative(ctx.workspace.resolve(path))] if path else []
        ok_stat, stat = await _git(ctx, "diff", "--stat", base, *scope)
        if not ok_stat:
            return ToolResult(False, error=stat)
        match = _STAT.search(stat)
        files, added, removed = (int(g or 0) for g in match.groups()) if match else (0, 0, 0)
        body = stat
        if not stat_only:
            ok, patch = await _git(ctx, "diff", "--unified=3", base, *scope)
            if not ok:
                return ToolResult(False, error=patch)
            body = truncate_middle(patch, ctx.settings.command_max_output_chars)
        return ToolResult(
            True,
            content={"diff": body or "(no changes)", "files_changed": files, "insertions": added, "deletions": removed},
            display=f"diff: {files} file(s), +{added} −{removed}",
            evidence={"kind": "diff", "files_changed": files, "insertions": added, "deletions": removed,
                      "stat": stat[-2000:]},
        )

    @registry.tool(
        "git_log",
        "Show recent commits of the repository (history before DevPilot started), optionally for one path.",
        {"type": "object", "properties": {
            "path": {"type": "string"},
            "limit": {"type": "integer", "description": "Number of commits (default 15)"}}},
        category="read")
    async def git_log(ctx: ToolContext, path: str = "", limit: int = 15) -> ToolResult:
        scope = ["--", ctx.workspace.relative(ctx.workspace.resolve(path))] if path else []
        ok, out = await _git(ctx, "log", f"-n{max(1, min(limit, 100))}", "--date=short",
                             "--pretty=format:%h %ad %an: %s", *scope)
        if not ok:
            return ToolResult(False, error=out)
        return ToolResult(True, content={"commits": out.splitlines()}, display=f"{len(out.splitlines())} commit(s)")
