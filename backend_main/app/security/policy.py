"""
Action policy: classify every command and file access before it happens.

Ported from Jarvis `safety/policy.py` and tightened for autonomous agents:

- Commands are argv lists, executed without a shell. Shell interpreters are
  refused outright, so pipes, `&&`, redirection and globbing cannot sneak in.
- Destructive, system, network and history-rewriting commands are FORBIDDEN.
  Agents change files through the file tools, which stay inside the workspace.
- Test runners, linters, build tools and read-only git are allowed.
- Installing dependencies or running arbitrary code strings needs a human (HIGH).
- Anything unrecognised needs a human too: unknown is never "safe".

Risk maps onto Mycel's approval classes: below HIGH is GREEN (autonomous),
HIGH is AMBER (human approval), FORBIDDEN is RED (blocked).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import PurePosixPath


class Risk(IntEnum):
    SAFE = 0        # read-only
    LOW = 1         # creates or runs project-local things (tests, builds)
    MEDIUM = 2      # modifies existing files inside the workspace
    HIGH = 3        # needs human approval
    FORBIDDEN = 4   # never


class ActionClass(StrEnum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"


@dataclass(frozen=True)
class Verdict:
    risk: Risk
    reason: str
    category: str = "generic"

    @property
    def action_class(self) -> ActionClass:
        if self.risk >= Risk.FORBIDDEN:
            return ActionClass.RED
        if self.risk >= Risk.HIGH:
            return ActionClass.AMBER
        return ActionClass.GREEN

    @property
    def allowed(self) -> bool:
        return self.risk < Risk.FORBIDDEN

    @property
    def needs_approval(self) -> bool:
        return self.risk == Risk.HIGH

    def describe(self) -> str:
        return f"[{self.action_class}/{self.category}] {self.reason}"


# ── command tables ──────────────────────────────────────────────────────────
_SHELLS = {"sh", "bash", "zsh", "fish", "dash", "ksh", "cmd", "powershell", "pwsh", "wsl"}
_DESTRUCTIVE = {
    "rm", "rmdir", "del", "erase", "rd", "shred", "srm", "mkfs", "format", "dd",
    "diskpart", "diskutil", "fdisk", "killall", "pkill", "kill", "taskkill",
    "shutdown", "reboot", "halt", "poweroff", "remove-item", "truncate",
}
_SYSTEM = {
    "sudo", "su", "doas", "runas", "chmod", "chown", "chgrp", "chflags", "icacls",
    "takeown", "attrib", "reg", "regedit", "sc", "net", "netsh", "setx", "bcdedit",
    "launchctl", "systemctl", "service", "crontab", "schtasks", "mount", "umount",
    "useradd", "usermod", "passwd", "visudo", "csrutil", "spctl", "nvram",
}
_NETWORK = {
    "curl", "wget", "scp", "sftp", "rsync", "ssh", "ftp", "nc", "ncat", "netcat",
    "telnet", "invoke-webrequest", "iwr", "invoke-restmethod", "certutil", "bitsadmin",
}
_READ_ONLY = {
    "ls", "dir", "cat", "type", "head", "tail", "grep", "findstr", "wc", "echo",
    "pwd", "which", "where", "tree", "diff", "fc", "sort", "uniq", "file", "stat",
    "du", "env", "printenv", "whoami", "hostname", "date", "uname",
}
_TEST_AND_LINT = {
    "pytest", "py.test", "tox", "nox", "ruff", "flake8", "pylint", "mypy", "pyright",
    "black", "isort", "bandit", "eslint", "prettier", "tsc", "jest", "vitest",
    "mocha", "phpunit", "rspec", "ctest",
}
_GIT_READ = {"status", "diff", "log", "show", "blame", "ls-files", "rev-parse", "grep", "shortlog", "describe"}
_GIT_LOCAL_WRITE = {"add", "commit", "stash", "restore", "checkout", "switch", "branch", "tag", "mv", "apply"}
_GIT_FORBIDDEN = {
    "push", "pull", "fetch", "clone", "remote", "reset", "clean", "rebase", "filter-branch",
    "filter-repo", "gc", "prune", "reflog", "update-ref", "config", "submodule", "worktree", "rm",
}
_PKG_MANAGERS = {"npm", "pnpm", "yarn", "bun"}
_PKG_RUN = {"test", "t", "run", "run-script", "lint", "build", "typecheck", "tsc"}
_PKG_INSTALL = {"install", "i", "ci", "add", "uninstall", "remove", "rm", "update", "upgrade", "link"}
_PKG_FORBIDDEN = {"publish", "unpublish", "deprecate", "login", "logout", "adduser", "token", "owner"}
_PY_MODULES_OK = {
    "pytest", "unittest", "doctest", "mypy", "ruff", "flake8", "pylint", "black", "isort",
    "compileall", "py_compile", "bandit", "coverage", "tox", "nox",
}
_PY_MODULES_READ = {"pip": {"list", "show", "freeze", "check"}}

# Catastrophic patterns caught even inside arguments (Jarvis forbidden_patterns).
_FORBIDDEN_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"rm\s+-rf\s+/",
        r":\(\)\s*\{.*\};\s*:",
        r"mkfs(\.|\s)",
        r"dd\s+if=.*of=/dev/",
        r"format\s+[a-z]:",
        r"del\s+/[sq]",
        r"rd\s+/s",
    )
]


def normalize_binary(token: str) -> str:
    name = PurePosixPath(token.replace("\\", "/")).name.lower()
    for ext in (".exe", ".cmd", ".bat", ".ps1", ".com"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _sub(argv: list[str], index: int = 1) -> str:
    """First non-flag argument at or after `index` (the subcommand)."""
    for token in argv[index:]:
        if not token.startswith("-"):
            return token.lower()
    return ""


def _classify_python(argv: list[str]) -> Verdict:
    args = argv[1:]
    if not args:
        return Verdict(Risk.HIGH, "interactive python interpreter", "exec")
    if args[0] in ("-c",) or "-c" in args[:2]:
        return Verdict(Risk.HIGH, "runs an inline code string", "exec")
    if args[0] == "-m" and len(args) > 1:
        module = args[1].split(".")[0].lower()
        if module in _PY_MODULES_OK:
            return Verdict(Risk.LOW, f"python -m {module}", "test")
        if module in _PY_MODULES_READ:
            if _sub(args, 2) in _PY_MODULES_READ[module]:
                return Verdict(Risk.SAFE, f"python -m {module} (read-only)", "read")
            return Verdict(Risk.HIGH, f"python -m {module} changes the environment", "install")
        return Verdict(Risk.HIGH, f"python -m {module} is not on the allowlist", "exec")
    if args[0] in ("--version", "-V"):
        return Verdict(Risk.SAFE, "version check", "read")
    # `python path/to/script.py`: the sandbox checks the path stays in the workspace.
    return Verdict(Risk.LOW, "runs a project script", "exec")


def classify_command(argv: list[str]) -> Verdict:
    if not argv or not argv[0].strip():
        return Verdict(Risk.FORBIDDEN, "empty command", "invalid")

    joined = " ".join(argv)
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern.search(joined):
            return Verdict(Risk.FORBIDDEN, f"matches forbidden pattern {pattern.pattern!r}", "catastrophic")

    binary = normalize_binary(argv[0])
    sub = _sub(argv)

    if binary in _SHELLS:
        return Verdict(Risk.FORBIDDEN, f"shell interpreter '{binary}' — pass the program and its arguments directly", "shell")
    if binary in _DESTRUCTIVE:
        return Verdict(Risk.FORBIDDEN, f"destructive command '{binary}' — use the file tools", "delete")
    if binary in _SYSTEM:
        return Verdict(Risk.FORBIDDEN, f"system-level command '{binary}'", "system")
    if binary in _NETWORK:
        return Verdict(Risk.FORBIDDEN, f"network command '{binary}' — the sandbox has no egress", "network")

    if binary == "git":
        if sub in _GIT_READ:
            return Verdict(Risk.SAFE, f"git {sub} (read-only)", "read")
        if sub in _GIT_FORBIDDEN:
            return Verdict(Risk.FORBIDDEN, f"git {sub} rewrites history or reaches the network", "vcs")
        if sub in _GIT_LOCAL_WRITE:
            return Verdict(Risk.HIGH, f"git {sub} changes repository state", "vcs")
        return Verdict(Risk.HIGH, f"git {sub or '(no subcommand)'} is not on the allowlist", "vcs")

    if binary in _PKG_MANAGERS:
        if sub in _PKG_FORBIDDEN:
            return Verdict(Risk.FORBIDDEN, f"{binary} {sub} publishes or touches credentials", "publish")
        if sub in _PKG_INSTALL:
            return Verdict(Risk.HIGH, f"{binary} {sub} changes dependencies", "install")
        if sub in _PKG_RUN or sub == "":
            return Verdict(Risk.LOW, f"{binary} {sub or 'test'}", "test")
        return Verdict(Risk.HIGH, f"{binary} {sub} is not on the allowlist", "exec")
    if binary in ("npx", "pnpx", "bunx"):
        return Verdict(Risk.HIGH, f"{binary} may download and run remote code", "install")
    if binary in ("pip", "pip3", "uv", "poetry", "pipenv", "conda"):
        if sub in ("list", "show", "freeze", "check", "--version"):
            return Verdict(Risk.SAFE, f"{binary} {sub} (read-only)", "read")
        return Verdict(Risk.HIGH, f"{binary} {sub} changes the environment", "install")

    if binary in ("python", "python3", "py", "pypy", "pypy3"):
        return _classify_python(argv)
    if binary in ("node", "deno", "ts-node", "tsx"):
        if any(a in ("-e", "--eval", "-p", "--print") for a in argv[1:]):
            return Verdict(Risk.HIGH, f"{binary} evaluates an inline code string", "exec")
        return Verdict(Risk.LOW, f"{binary} runs a project script", "exec")

    if binary in _TEST_AND_LINT:
        return Verdict(Risk.LOW, f"{binary} (test / lint)", "test")
    if binary == "go":
        return (Verdict(Risk.LOW, f"go {sub}", "test") if sub in ("test", "build", "vet", "fmt", "version")
                else Verdict(Risk.HIGH, f"go {sub} is not on the allowlist", "exec"))
    if binary == "cargo":
        return (Verdict(Risk.LOW, f"cargo {sub}", "test") if sub in ("test", "build", "check", "clippy", "fmt")
                else Verdict(Risk.HIGH, f"cargo {sub} is not on the allowlist", "exec"))
    if binary in ("mvn", "gradle", "gradlew", "dotnet"):
        return (Verdict(Risk.LOW, f"{binary} {sub}", "test") if sub in ("test", "build", "verify", "check", "compile")
                else Verdict(Risk.HIGH, f"{binary} {sub} is not on the allowlist", "exec"))
    if binary == "make":
        return Verdict(Risk.MEDIUM, "make runs project-defined targets", "build")
    if binary in _READ_ONLY:
        return Verdict(Risk.SAFE, f"{binary} (read-only)", "read")

    return Verdict(Risk.HIGH, f"unrecognised command '{binary}'", "generic")


# ── file access ─────────────────────────────────────────────────────────────
_SECRET_FILES = re.compile(
    r"(^|/)(\.env(\..+)?|.*\.pem|.*\.key|id_rsa.*|id_ed25519.*|\.npmrc|\.pypirc|credentials(\.json)?|secrets?\.(json|ya?ml|toml))$",
    re.I,
)
_CI_FILES = re.compile(r"(^|/)\.(github|gitlab|circleci)/", re.I)


def classify_file_access(relative_path: str, *, write: bool) -> Verdict:
    """`relative_path` is already jailed to the workspace (see sandbox.Workspace)."""
    rel = relative_path.replace("\\", "/") or "."
    while rel.startswith("./"):
        rel = rel[2:]
    if rel == ".git" or rel.startswith(".git/"):
        return (Verdict(Risk.FORBIDDEN, "git internals are managed by DevPilot", "vcs") if write
                else Verdict(Risk.SAFE, "read git internals", "read"))
    if _SECRET_FILES.search(rel):
        # Reading a secret would send it to an LLM provider; writing one is never needed.
        return Verdict(Risk.HIGH, f"'{rel}' may contain secrets", "secrets")
    if write and _CI_FILES.search(rel):
        return Verdict(Risk.HIGH, f"'{rel}' changes CI configuration", "ci")
    if write:
        return Verdict(Risk.MEDIUM, f"writes '{rel}' inside the workspace", "write")
    return Verdict(Risk.SAFE, f"reads '{rel}'", "read")
