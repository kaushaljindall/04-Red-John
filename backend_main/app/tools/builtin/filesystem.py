"""
File tools, jailed to the mission workspace. Ported from Jarvis `tools/fs.py`.

There is no delete tool and no journal: the workspace is a git repository
with a baseline commit, so every change is visible in `git_diff` and can be
reverted by DevPilot itself.
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from app.security.policy import Risk, Verdict, classify_file_access
from app.security.sandbox import VENDOR_DIRS
from app.tools.registry import ToolContext, ToolRegistry, ToolResult, run_blocking

MAX_LIST_ENTRIES = 500


def _assess(write: bool):
    def assess(ctx: ToolContext, args: dict[str, Any]) -> Verdict:
        target = ctx.workspace.resolve(args.get("path", "."))
        return classify_file_access(ctx.workspace.relative(target), write=write)

    return assess


def _read(ctx: ToolContext, path: str, offset: int = 1, limit: int = 400) -> ToolResult:
    ws = ctx.workspace
    file = ws.resolve(path)
    rel = ws.relative(file)
    if not file.exists():
        return ToolResult(False, error=f"File not found: {rel}")
    if file.is_dir():
        return ToolResult(False, error=f"{rel} is a directory; use list_dir")
    size = file.stat().st_size
    cap = ctx.settings.max_file_read_bytes
    data = file.read_bytes()[:cap]
    if b"\x00" in data[:8000]:
        return ToolResult(False, error=f"{rel} is a binary file")
    lines = data.decode("utf-8", "replace").splitlines()
    start = max(1, offset)
    chosen = lines[start - 1 : start - 1 + max(1, limit)]
    body = "\n".join(f"{start + i:>5}| {line}" for i, line in enumerate(chosen))
    more = start - 1 + len(chosen) < len(lines)
    if more:
        body += f"\n… [{len(lines) - (start - 1 + len(chosen))} more lines; continue with offset={start + len(chosen)}]"
    if size > cap:
        body += f"\n… [file truncated at {cap} of {size} bytes]"
    return ToolResult(True, content=body, display=f"read {rel} ({len(chosen)} lines)",
                      meta={"path": rel, "total_lines": len(lines)})


def _write(ctx: ToolContext, path: str, content: str) -> ToolResult:
    ws = ctx.workspace
    file = ws.resolve(path)
    rel = ws.relative(file)
    if file.is_dir():
        return ToolResult(False, error=f"{rel} is a folder — give a file name inside it")
    for parent in file.parents:
        if parent.exists():
            if not parent.is_dir():
                return ToolResult(False, error=f"{ws.relative(parent)} is a file, not a folder")
            break
    existed = file.exists()
    file.parent.mkdir(parents=True, exist_ok=True)
    # newline="" keeps the line endings the agent wrote instead of translating them.
    with open(file, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    verified = file.read_text(encoding="utf-8", errors="replace") == content
    return ToolResult(True, content={"path": rel, "bytes": len(content.encode()), "replaced": existed, "verified": verified},
                      display=f"{'updated' if existed else 'created'} {rel}")


def _edit(ctx: ToolContext, path: str, old: str, new: str, all: bool = False) -> ToolResult:
    ws = ctx.workspace
    file = ws.resolve(path)
    rel = ws.relative(file)
    if not file.is_file():
        return ToolResult(False, error=f"File not found: {rel}")
    if old == new:
        return ToolResult(False, error="'old' and 'new' are identical; nothing to change")
    with open(file, encoding="utf-8", errors="replace", newline="") as handle:
        text = handle.read()
    count = text.count(old)
    if count == 0 and "\r\n" in text and "\r\n" not in old:
        # Model wrote LF; file uses CRLF. Match on the file's own line endings.
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        count = text.count(old)
    if count == 0:
        return ToolResult(False, error="The 'old' text was not found. Read the file again and copy the text exactly.")
    if count > 1 and not all:
        return ToolResult(False, error=f"'old' appears {count} times; include more surrounding lines or pass all=true.")
    updated = text.replace(old, new) if all else text.replace(old, new, 1)
    with open(file, "w", encoding="utf-8", newline="") as handle:
        handle.write(updated)
    replaced = count if all else 1
    return ToolResult(True, content={"path": rel, "replacements": replaced}, display=f"edited {rel} ({replaced} replacement)")


def _list(ctx: ToolContext, path: str = ".", all: bool = False) -> ToolResult:
    ws = ctx.workspace
    folder = ws.resolve(path)
    rel = ws.relative(folder)
    if not folder.is_dir():
        return ToolResult(False, error=f"{rel} is not a directory")
    rows = []
    for entry in sorted(folder.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
        if not all and (entry.name.startswith(".") or entry.name in VENDOR_DIRS):
            continue
        try:
            rows.append({"name": entry.name + ("/" if entry.is_dir() else ""),
                         "size": entry.stat().st_size if entry.is_file() else None})
        except OSError:
            continue
    return ToolResult(True, content={"path": rel, "entries": rows[:MAX_LIST_ENTRIES], "count": len(rows),
                                     "truncated": len(rows) > MAX_LIST_ENTRIES},
                      display=f"listed {rel} ({len(rows)} entries)")


def _find(ctx: ToolContext, pattern: str, path: str = ".", limit: int = 200) -> ToolResult:
    ws = ctx.workspace
    root = ws.resolve(path)
    hits: list[str] = []
    match_path = "/" in pattern
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in VENDOR_DIRS)
        for name in sorted(filenames):
            rel = ws.relative(Path(dirpath, name))
            if fnmatch.fnmatch(rel if match_path else name, pattern):
                hits.append(rel)
                if len(hits) >= limit:
                    return ToolResult(True, content={"matches": hits, "truncated": True}, display=f"found {len(hits)}+ files")
    return ToolResult(True, content={"matches": hits, "count": len(hits)}, display=f"found {len(hits)} file(s)")


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "read_file",
        "Read a text file from the repository, with line numbers. Use offset/limit to page through large files.",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "offset": {"type": "integer", "description": "First line to read (1-based)"},
            "limit": {"type": "integer", "description": "Maximum lines to return (default 400)"}},
         "required": ["path"]},
        assess=_assess(write=False))
    async def read_file(ctx: ToolContext, path: str, offset: int = 1, limit: int = 400) -> ToolResult:
        return await run_blocking(_read, ctx, path, offset, limit)

    @registry.tool(
        "write_file",
        "Create a file or replace its whole content. Prefer edit_file for changes to existing files.",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "Path relative to the repository root"},
            "content": {"type": "string", "description": "The complete file content"}},
         "required": ["path", "content"]},
        risk=Risk.MEDIUM, category="write", assess=_assess(write=True))
    async def write_file(ctx: ToolContext, path: str, content: str) -> ToolResult:
        return await run_blocking(_write, ctx, path, content)

    @registry.tool(
        "edit_file",
        "Replace an exact piece of text in a file. 'old' must match the file exactly (read it first) and be unique unless all=true.",
        {"type": "object", "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact existing text to replace"},
            "new": {"type": "string", "description": "Replacement text"},
            "all": {"type": "boolean", "description": "Replace every occurrence"}},
         "required": ["path", "old", "new"]},
        risk=Risk.MEDIUM, category="write", assess=_assess(write=True))
    async def edit_file(ctx: ToolContext, path: str, old: str, new: str, all: bool = False) -> ToolResult:
        return await run_blocking(_edit, ctx, path, old, new, all)

    @registry.tool(
        "list_dir",
        "List a directory of the repository (dependency and cache folders are hidden unless all=true).",
        {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory relative to the repository root (default '.')"},
            "all": {"type": "boolean", "description": "Include dotfiles and vendor folders"}}},
        assess=_assess(write=False))
    async def list_dir(ctx: ToolContext, path: str = ".", all: bool = False) -> ToolResult:
        return await run_blocking(_list, ctx, path, all)

    @registry.tool(
        "find_files",
        "Find files by glob, e.g. '*.py', 'test_*.py' or 'src/**/*.ts' (patterns with '/' match the relative path).",
        {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Folder to search under (default '.')"},
            "limit": {"type": "integer"}},
         "required": ["pattern"]},
        category="search", assess=_assess(write=False))
    async def find_files(ctx: ToolContext, pattern: str, path: str = ".", limit: int = 200) -> ToolResult:
        return await run_blocking(_find, ctx, pattern, path, limit)
