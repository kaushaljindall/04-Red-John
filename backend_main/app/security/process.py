"""
Runs a command for an agent: no shell, jailed cwd, scrubbed environment,
hard timeout, bounded output. Ported from Jarvis `tools/shell.py`.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import PolicyViolation

# Environment variables that must never reach agent-run code.
_SECRET_ENV = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|MONGO|REDIS|GROQ|GEMINI|OPENAI|ANTHROPIC|AWS_|AZURE_|GCP_|GOOGLE_)",
    re.I,
)
# cmd.exe re-parses arguments of .bat/.cmd files; these characters would let an argument escape.
_BATCH_UNSAFE = re.compile(r'[&|<>^%"\r\n]')


@dataclass(frozen=True)
class ProcessResult:
    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def combined(self) -> str:
        out = self.stdout.strip()
        if self.stderr.strip():
            out += ("\n[stderr]\n" if out else "") + self.stderr.strip()
        return out

    def failure_reason(self, limit: int = 1500) -> str:
        """Why it failed, with the tail of the output — a bare 'exit 1' makes agents fix blind."""
        head = "timed out" if self.timed_out else f"exit {self.exit_code}"
        tail = (self.stderr or "").strip() or (self.stdout or "").strip()
        if not tail:
            return head
        return f"{head}: {'…' + tail[-limit:] if len(tail) > limit else tail}"


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n… [{len(text) - limit} chars elided] …\n{text[-half:]}"


def scrubbed_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    env.update(
        {
            "CI": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            "FORCE_COLOR": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "npm_config_yes": "false",
        }
    )
    return env


def resolve_executable(argv: list[str]) -> list[str]:
    exe = shutil.which(argv[0])
    if exe is None:
        raise PolicyViolation(f"'{argv[0]}' is not installed or not on PATH", details={"tool_unavailable": True})
    if exe.lower().endswith((".bat", ".cmd")):
        bad = [a for a in argv[1:] if _BATCH_UNSAFE.search(a)]
        if bad:
            raise PolicyViolation(
                "argument contains characters that are unsafe for a batch-file program",
                details={"arguments": bad},
            )
    return [exe, *argv[1:]]


def _run_sync(argv: list[str], cwd: Path, timeout_s: int, max_output: int) -> ProcessResult:
    started = time.monotonic()
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=scrubbed_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(
            argv=argv,
            exit_code=-1,
            stdout=truncate_middle(_text(exc.stdout), max_output),
            stderr=truncate_middle(_text(exc.stderr), max_output),
            duration_s=time.monotonic() - started,
            timed_out=True,
        )
    return ProcessResult(
        argv=argv,
        exit_code=proc.returncode,
        stdout=truncate_middle(proc.stdout or "", max_output),
        stderr=truncate_middle(proc.stderr or "", max_output),
        duration_s=time.monotonic() - started,
    )


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


async def run_process(argv: list[str], cwd: Path, *, timeout_s: int, max_output: int) -> ProcessResult:
    # A worker thread rather than asyncio subprocesses: those need the Proactor
    # loop on Windows, which uvicorn does not always use.
    return await asyncio.to_thread(_run_sync, resolve_executable(argv), cwd, timeout_s, max_output)
