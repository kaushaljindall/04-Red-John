"""
Command tools: `run_command` (general, policy-gated) and `run_tests`
(auto-detected test runner whose result is recorded as evidence).
"""
from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from app.core.errors import PolicyViolation
from app.security.policy import Risk, Verdict, classify_command
from app.security.process import ProcessResult, run_process, truncate_middle
from app.security.sandbox import Workspace
from app.tools.registry import ToolContext, ToolRegistry, ToolResult

_PATHISH = re.compile(r"^(\.{1,2}[/\\]|[/\\]|[A-Za-z]:[/\\])|[/\\]\.\.([/\\]|$)")


def parse_command(command: str | list[str]) -> list[str]:
    if isinstance(command, list):
        return [str(part) for part in command]
    # Forward slashes work for every tool we allow, and posix splitting would eat backslashes.
    try:
        return shlex.split(command.replace("\\", "/"), posix=True)
    except ValueError as exc:
        raise PolicyViolation(f"Cannot parse command: {exc}") from None


def check_argv_paths(workspace: Workspace, argv: list[str]) -> None:
    """Any argument that looks like a path must stay inside the workspace."""
    for token in argv[1:]:
        value = token.split("=", 1)[1] if token.startswith("-") and "=" in token else token
        if value and _PATHISH.search(value):
            workspace.resolve(value)  # raises PolicyViolation when it escapes


def _working_dir(ctx: ToolContext, cwd: str) -> Path:
    folder = ctx.workspace.resolve(cwd or ".")
    if not folder.is_dir():
        raise PolicyViolation(f"cwd '{cwd}' is not a directory in the workspace")
    return folder


def _assess_command(ctx: ToolContext, args: dict[str, Any]) -> Verdict:
    argv = parse_command(args.get("command", ""))
    _working_dir(ctx, args.get("cwd", "."))
    check_argv_paths(ctx.workspace, argv)
    return classify_command(argv)


def _process_content(result: ProcessResult, ctx: ToolContext) -> dict[str, Any]:
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "duration_s": round(result.duration_s, 2),
        "output": truncate_middle(result.combined(), ctx.settings.command_max_output_chars),
    }


# ── test-runner detection ───────────────────────────────────────────────────
def detect_test_command(root: Path) -> list[str] | None:
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
        except ValueError:
            scripts = {}
        test = scripts.get("test", "")
        if test and "no test specified" not in test:
            return ["npm", "test", "--silent"]
    python_markers = ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "setup.py", "conftest.py")
    if any((root / m).is_file() for m in python_markers) or any(root.glob("test*/**/*.py")) or any(root.glob("test_*.py")):
        return ["python", "-m", "pytest", "-q", "-rfE", "--no-header", "-p", "no:cacheprovider"]
    if (root / "go.mod").is_file():
        return ["go", "test", "./..."]
    if (root / "Cargo.toml").is_file():
        return ["cargo", "test"]
    return None


_PYTEST_COUNTS = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)")
_JEST = re.compile(r"Tests:\s+(.*?)\s+total", re.I)
_UNITTEST_RAN = re.compile(r"Ran (\d+) tests?")
_UNITTEST_FAIL = re.compile(r"FAILED \((.*?)\)")
_FAILED_TEST = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.M)


def summarize_test_output(output: str, exit_code: int) -> dict[str, Any]:
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    tail = output[-4000:]
    for number, label in _PYTEST_COUNTS.findall(tail):
        key = "errors" if label.startswith("error") else label
        if key in counts:
            counts[key] = max(counts[key], int(number))
    jest = _JEST.search(tail)
    if jest:
        for number, label in re.findall(r"(\d+) (passed|failed|skipped)", jest.group(1)):
            counts[label] = int(number)
    ran = _UNITTEST_RAN.search(tail)
    if ran and not any(counts.values()):
        total = int(ran.group(1))
        failed = _UNITTEST_FAIL.search(tail)
        if failed:
            for label, number in re.findall(r"(failures|errors)=(\d+)", failed.group(1)):
                counts["failed" if label == "failures" else "errors"] = int(number)
        counts["passed"] = total - counts["failed"] - counts["errors"]
    return {
        **counts,
        "failing_tests": _FAILED_TEST.findall(output)[:30],
        # pytest exit 5 = nothing collected; treat as not-passing so nobody claims success on it.
        "no_tests_collected": exit_code == 5,
        "passed_run": exit_code == 0,
    }


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "run_command",
        "Run one program in the repository, e.g. 'python -m pytest tests/test_api.py -q' or 'npm run lint'. "
        "No shell: pipes, '&&', redirection and globbing do not work. Destructive, network and install "
        "commands are blocked or need human approval.",
        {"type": "object", "properties": {
            "command": {"type": "string", "description": "Program and arguments"},
            "cwd": {"type": "string", "description": "Working directory relative to the repository root"},
            "timeout_s": {"type": "integer", "description": "Seconds before the command is stopped"}},
         "required": ["command"]},
        risk=Risk.MEDIUM, category="exec", assess=_assess_command)
    async def run_command(ctx: ToolContext, command: str, cwd: str = ".", timeout_s: int | None = None) -> ToolResult:
        argv = parse_command(command)
        timeout = min(timeout_s or ctx.settings.command_timeout_s, ctx.settings.command_timeout_s)
        result = await run_process(argv, _working_dir(ctx, cwd), timeout_s=timeout,
                                   max_output=ctx.settings.command_max_output_chars)
        return ToolResult(
            result.ok,
            content=_process_content(result, ctx),
            error="" if result.ok else result.failure_reason(),
            display=f"$ {' '.join(argv)[:80]}  (exit {result.exit_code})",
        )

    @registry.tool(
        "run_tests",
        "Run the project's test suite (auto-detected: pytest, npm test, go test, cargo test) or a given test "
        "command. The real output is recorded as evidence. Use 'command' to run a subset.",
        {"type": "object", "properties": {
            "command": {"type": "string", "description": "Optional explicit test command; default is auto-detected"},
            "cwd": {"type": "string", "description": "Working directory relative to the repository root"}}},
        risk=Risk.LOW, category="test", evidence=True,
        assess=lambda ctx, args: _assess_command(ctx, args) if args.get("command") else Verdict(Risk.LOW, "auto-detected test run", "test"))
    async def run_tests(ctx: ToolContext, command: str = "", cwd: str = ".") -> ToolResult:
        folder = _working_dir(ctx, cwd)
        argv = parse_command(command) if command else detect_test_command(folder)
        if not argv:
            # Recorded as evidence: "this project has no tests" is a fact Atlas plans around (it hires a test author).
            return ToolResult(False, error="No test runner detected (no pytest config, package.json test script, go.mod "
                                           "or Cargo.toml). Pass 'command' explicitly or report that the project has no tests.",
                              evidence={"kind": "test_run", "command": None, "exit_code": None, "passed": False,
                                        "summary": {"passed": 0, "failed": 0, "errors": 0, "no_runner": True},
                                        "output_tail": "", "duration_s": 0})
        result = await run_process(argv, folder, timeout_s=ctx.settings.command_timeout_s,
                                   max_output=ctx.settings.command_max_output_chars)
        summary = summarize_test_output(result.combined(), result.exit_code)
        passed = result.ok and not summary["no_tests_collected"]
        content = {**_process_content(result, ctx), "summary": summary}
        return ToolResult(
            passed,
            content=content,
            error="" if passed else ("no tests were collected" if summary["no_tests_collected"] else result.failure_reason()),
            display=f"tests: {summary['passed']} passed, {summary['failed']} failed, {summary['errors']} errors "
                    f"(exit {result.exit_code}{', timed out' if result.timed_out else ''}) — {' '.join(argv)[:80]}",
            evidence={
                "kind": "test_run",
                "command": " ".join(argv),
                "exit_code": result.exit_code,
                "passed": passed,
                "summary": summary,
                "output_tail": result.combined()[-3000:],
                "duration_s": round(result.duration_s, 2),
            },
        )
