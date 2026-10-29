"""
Mission workspaces: an isolated copy of the target repository per mission.

Agents never touch the user's repository. Each mission gets a copy with a
baseline git commit, so every change is a diff against the baseline and the
real repo is only updated after a human approves (Phase 8).
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import NotFoundError, PolicyViolation, ValidationFailed

log = logging.getLogger(__name__)

# Never copied: caches and virtualenvs (absolute paths inside, not relocatable).
COPY_IGNORE = (
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".nox",
    ".venv", "venv", "env", ".next", ".turbo", ".cache", "coverage", "htmlcov",
)
# Skipped by search and listing tools (still present on disk if copied).
VENDOR_DIRS = frozenset({*COPY_IGNORE, ".git", "node_modules", "dist", "build", "site-packages", ".idea", ".vscode"})

BASELINE_REF = "refs/devpilot/baseline"
_GIT_IDENTITY = ["-c", "user.name=DevPilot", "-c", "user.email=devpilot@localhost", "-c", "commit.gpgsign=false"]


@dataclass(frozen=True)
class Workspace:
    mission_id: str
    root: Path
    source: Path
    baseline_commit: str

    def resolve(self, path: str | Path = ".") -> Path:
        """Resolve `path` (relative to the root, or absolute) and refuse anything outside the root."""
        raw = Path(str(path).strip() or ".")
        candidate = raw if raw.is_absolute() else self.root / raw
        resolved = candidate.resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise PolicyViolation(
                f"'{path}' is outside the mission workspace", details={"path": str(path)}
            )
        return resolved

    def relative(self, path: Path) -> str:
        rel = path.resolve().relative_to(self.root).as_posix()
        return rel or "."


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *_GIT_IDENTITY, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


class WorkspaceManager:
    def __init__(self, base_dir: Path, allowed_roots: list[Path] | None = None) -> None:
        self._base = base_dir.resolve()
        self._allowed_roots = [p.resolve() for p in allowed_roots or []]
        self._base.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Workspace] = {}

    def _validate_source(self, source: Path) -> Path:
        src = source.expanduser().resolve()
        if not src.is_dir():
            raise ValidationFailed(f"Repository not found: {source}")
        if self._allowed_roots and not any(src.is_relative_to(r) for r in self._allowed_roots):
            raise PolicyViolation(f"Repository {src} is outside the allowed roots")
        if src == self._base or src.is_relative_to(self._base) or self._base.is_relative_to(src):
            raise PolicyViolation("Repository overlaps the DevPilot workspace directory")
        return src

    def _create_sync(self, mission_id: str, src: Path) -> Workspace:
        root = self._base / mission_id
        if root.exists():
            raise ValidationFailed(f"Workspace for {mission_id} already exists")
        shutil.copytree(src, root, ignore=shutil.ignore_patterns(*COPY_IGNORE), symlinks=True)
        if not (root / ".git").exists():
            _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "--allow-empty", "-m", "DevPilot baseline")
        baseline = _git(root, "rev-parse", "HEAD")
        _git(root, "update-ref", BASELINE_REF, baseline)
        log.info("workspace %s created from %s at %s", mission_id, src, baseline[:10])
        return Workspace(mission_id=mission_id, root=root.resolve(), source=src, baseline_commit=baseline)

    async def create(self, mission_id: str, source: Path) -> Workspace:
        src = self._validate_source(source)
        workspace = await asyncio.to_thread(self._create_sync, mission_id, src)
        self._cache[mission_id] = workspace
        return workspace

    def get(self, mission_id: str) -> Workspace:
        if mission_id in self._cache:
            return self._cache[mission_id]
        root = self._base / mission_id
        if not root.is_dir():
            raise NotFoundError(f"No workspace for mission {mission_id}")
        # Another worker created it: rebuild the handle from disk.
        baseline = _git(root, "rev-parse", BASELINE_REF)
        workspace = Workspace(mission_id=mission_id, root=root.resolve(), source=root, baseline_commit=baseline)
        self._cache[mission_id] = workspace
        return workspace

    async def destroy(self, mission_id: str) -> None:
        self._cache.pop(mission_id, None)
        root = self._base / mission_id
        if root.is_dir():
            await asyncio.to_thread(shutil.rmtree, root, True)
