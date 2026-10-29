"""
Content search across the workspace. Pure Python, so it behaves the same on
every host (Jarvis shelled out to ripgrep, which is not always installed).
"""
from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from app.security.policy import Verdict, classify_file_access
from app.security.sandbox import VENDOR_DIRS
from app.tools.registry import ToolContext, ToolRegistry, ToolResult, run_blocking

MAX_FILE_BYTES = 1_000_000
MAX_LINE_CHARS = 300


def _search(ctx: ToolContext, query: str, path: str, glob: str, limit: int, regex: bool) -> ToolResult:
    ws = ctx.workspace
    root = ws.resolve(path)
    flags = 0 if any(c.isupper() for c in query) else re.IGNORECASE  # smart case
    try:
        matcher = re.compile(query if regex else re.escape(query), flags)
    except re.error as exc:
        return ToolResult(False, error=f"Invalid regular expression: {exc}. Pass regex=false for a literal search.")

    files = [root] if root.is_file() else []
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in VENDOR_DIRS)
            files.extend(Path(dirpath, name) for name in sorted(filenames))

    matches: list[str] = []
    files_hit = 0
    for file in files:
        rel = ws.relative(file)
        if glob and not fnmatch.fnmatch(file.name, glob) and not fnmatch.fnmatch(rel, glob):
            continue
        # Never surface secrets through search results.
        if classify_file_access(rel, write=False).category == "secrets":
            continue
        try:
            if file.stat().st_size > MAX_FILE_BYTES:
                continue
            data = file.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8000]:
            continue
        hit = False
        for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            if matcher.search(line):
                hit = True
                text = line.strip()
                matches.append(f"{rel}:{number}: {text[:MAX_LINE_CHARS]}")
                if len(matches) >= limit:
                    return ToolResult(True, content={"matches": matches, "truncated": True},
                                      display=f"{len(matches)}+ matches for {query!r}")
        files_hit += hit
    return ToolResult(True, content={"matches": matches, "count": len(matches), "files": files_hit},
                      display=f"{len(matches)} match(es) for {query!r} in {files_hit} file(s)")


def _assess(ctx: ToolContext, args: dict) -> Verdict:
    target = ctx.workspace.resolve(args.get("path", "."))
    return classify_file_access(ctx.workspace.relative(target), write=False)


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "search_text",
        "Search file contents in the repository. Returns 'path:line: text'. Case-insensitive unless the query has capitals.",
        {"type": "object", "properties": {
            "query": {"type": "string", "description": "Text or regular expression to find"},
            "path": {"type": "string", "description": "File or folder to search (default '.')"},
            "glob": {"type": "string", "description": "Only files matching this glob, e.g. '*.py'"},
            "regex": {"type": "boolean", "description": "Treat query as a regular expression (default true)"},
            "limit": {"type": "integer", "description": "Maximum matches (default 80)"}},
         "required": ["query"]},
        category="search", assess=_assess)
    async def search_text(ctx: ToolContext, query: str, path: str = ".", glob: str = "",
                          regex: bool = True, limit: int = 80) -> ToolResult:
        return await run_blocking(_search, ctx, query, path, glob, limit, regex)
