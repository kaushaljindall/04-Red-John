"""
Applying a completed mission's change to the real repository — the one action
in DevPilot that touches the user's files, so it always needs a human.

    request()      the change is verified; ask a human (an AMBER approval with the diff stat)
    on_decision()  approved → apply; denied → record the rejection

Applying is all-or-nothing and conservative:
- Every file the mission touched must still be exactly as it was at the
  mission's baseline. If anyone changed it since, nothing is written and the
  conflict is reported (we never overwrite someone's work).
- Originals are backed up first, and the full patch is kept for the record.
- Nothing is committed or pushed: the change lands in the working tree for the
  human to review in their own tools.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings
from app.core.errors import ConflictError, ValidationFailed
from app.core.ids import now_ms
from app.domain.events import EventKind, LogLevel
from app.domain.governance import Approval, AuditEntry, AuditOutcome
from app.domain.mission import ApplyRecord, ApplyStatus, Mission, MissionStatus
from app.infrastructure.storage.base import Storage
from app.security.sandbox import WorkspaceManager
from app.services.event_publisher import EventPublisher
from app.services.mission_board import MissionBoard

log = logging.getLogger(__name__)

APPLY_TOOL = "apply_changes"
APPROVAL_WINDOW_MS = 7 * 24 * 3600 * 1000


class ApplyConflict(Exception):
    def __init__(self, paths: list[str]) -> None:
        super().__init__(", ".join(paths))
        self.paths = paths


@dataclass(frozen=True)
class Change:
    kind: str   # A, M or D
    path: str


def _git(root: Path, *args: str, text: bool = True) -> str | bytes:
    result = subprocess.run(["git", "-c", "core.quotepath=off", "-c", "core.autocrlf=false", *args], cwd=root,
                            capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout.decode("utf-8", "replace") if text else result.stdout


def mission_changes(root: Path, baseline: str) -> list[Change]:
    _git(root, "add", "--intent-to-add", "--all")
    raw = _git(root, "diff", "--name-status", "--no-renames", "-z", baseline)
    parts = [p for p in str(raw).split("\0") if p]
    return [Change(kind=parts[i][0], path=parts[i + 1]) for i in range(0, len(parts) - 1, 2)]


def _baseline_bytes(root: Path, baseline: str, path: str) -> bytes | None:
    try:
        return _git(root, "cat-file", "-p", f"{baseline}:{path}", text=False)  # type: ignore[return-value]
    except RuntimeError:
        return None


def _inside(base: Path, rel: str) -> Path:
    target = (base / rel).resolve()
    if not target.is_relative_to(base):
        raise ValidationFailed(f"path {rel!r} escapes the repository")
    return target


class ApplyService:
    def __init__(self, board: MissionBoard, storage: Storage, events: EventPublisher, workspaces: WorkspaceManager,
                 settings: Settings) -> None:
        self._board = board
        self._storage = storage
        self._events = events
        self._workspaces = workspaces
        self._settings = settings

    async def patch(self, mission_id: str) -> str:
        mission = await self._board.get(mission_id)
        root = self._workspaces.get(mission_id).root
        await asyncio.to_thread(_git, root, "add", "--intent-to-add", "--all")
        return str(await asyncio.to_thread(_git, root, "diff", "--binary", "--no-renames", mission.baseline_commit))

    async def request(self, mission_id: str, *, by: str) -> Approval:
        mission = await self._board.get(mission_id)
        if mission.status != MissionStatus.COMPLETED:
            raise ConflictError(f"Only a completed mission can be applied; this one is {mission.status}")
        if mission.application and mission.application.status in (ApplyStatus.REQUESTED, ApplyStatus.APPLIED):
            raise ConflictError(f"Apply is already {mission.application.status}")
        root = self._workspaces.get(mission_id).root
        changes = await asyncio.to_thread(mission_changes, root, mission.baseline_commit)
        if not changes:
            raise ValidationFailed("The mission made no changes; there is nothing to apply")
        stat = str(await asyncio.to_thread(_git, root, "diff", "--stat", "--no-renames", mission.baseline_commit))
        approval = Approval(
            mission_id=mission_id, agent=by, tool=APPLY_TOOL, action_class="AMBER", category="apply",
            reason=f"Apply {len(changes)} verified file change(s) to {mission.repo_path} (working tree only; "
                   "no commit or push)",
            args_preview=stat[-600:], expires_at=now_ms() + APPROVAL_WINDOW_MS,
        )
        await self._storage.approvals.create(approval)
        await self._events.publish(mission_id, EventKind.APPROVAL_REQUEST, approval.request_payload())
        await self._board.record_application(mission_id, ApplyRecord(
            status=ApplyStatus.REQUESTED, approval_id=approval.approval_id, requested_by=by,
            files=[c.path for c in changes]))
        await self._events.log(mission_id, f"Apply requested by {by}: {len(changes)} file(s) wait for human "
                                           "approval.", LogLevel.ARMOR)
        return approval

    async def on_decision(self, approval: Approval) -> ApplyRecord:
        mission = await self._board.get(approval.mission_id)
        current = mission.application
        if current is None or current.approval_id != approval.approval_id or current.status != ApplyStatus.REQUESTED:
            raise ConflictError("This approval does not match a pending apply request")
        decided = {"decided_by": approval.decided_by, "decided_at": approval.decided_at}
        if not approval.approved:
            record = current.model_copy(update={"status": ApplyStatus.REJECTED, **decided})
            await self._board.record_application(mission.mission_id, record)
            await self._events.log(mission.mission_id, f"Apply rejected by {approval.decided_by}; the repository "
                                                       "was not touched.", LogLevel.WARN)
            await self._audit(mission, AuditOutcome.DENIED, record)
            return record
        try:
            files, patch_path, backup = await asyncio.to_thread(self._apply_sync, mission)
        except ApplyConflict as exc:
            record = current.model_copy(update={
                "status": ApplyStatus.FAILED, **decided,
                "error": "These files changed in the repository since the mission started, so nothing was "
                         f"written: {', '.join(exc.paths)}"})
            await self._board.record_application(mission.mission_id, record)
            await self._events.log(mission.mission_id, f"Apply stopped: {record.error}", LogLevel.ERROR)
            await self._audit(mission, AuditOutcome.FAILED, record)
            return record
        record = current.model_copy(update={"status": ApplyStatus.APPLIED, **decided, "files": files,
                                            "patch_path": str(patch_path), "backup_path": str(backup)})
        await self._board.record_application(mission.mission_id, record)
        await self._events.log(mission.mission_id, f"Applied {len(files)} file(s) to {mission.repo_path} after "
                                                   f"{approval.decided_by} approved. Nothing was committed.",
                               LogLevel.SUCCESS)
        await self._audit(mission, AuditOutcome.OK, record)
        return record

    def _apply_sync(self, mission: Mission) -> tuple[list[str], Path, Path]:
        workspace = self._workspaces.get(mission.mission_id).root
        target = Path(mission.repo_path).resolve()
        if not target.is_dir():
            raise ValidationFailed(f"The repository {target} no longer exists")
        baseline = mission.baseline_commit or ""
        changes = mission_changes(workspace, baseline)

        # 1. Check everything before writing anything.
        conflicts = []
        for change in changes:
            path = _inside(target, change.path)
            current = path.read_bytes() if path.is_file() else None
            if current != _baseline_bytes(workspace, baseline, change.path):
                conflicts.append(change.path)
        if conflicts:
            raise ApplyConflict(conflicts)

        # 2. Keep the patch and back up what will be overwritten or removed.
        records = self._settings.data_dir / "applied" / mission.mission_id
        backup = records / "backup"
        backup.mkdir(parents=True, exist_ok=True)
        patch_path = records / "change.patch"
        patch_path.write_text(str(_git(workspace, "diff", "--binary", "--no-renames", baseline)), encoding="utf-8")
        for change in changes:
            original = _inside(target, change.path)
            if original.is_file():
                copy = backup / change.path
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original, copy)

        # 3. Write.
        for change in changes:
            destination = _inside(target, change.path)
            if change.kind == "D":
                destination.unlink(missing_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_inside(workspace, change.path), destination)
        log.info("applied %d file(s) from %s to %s", len(changes), mission.mission_id, target)
        return [c.path for c in changes], patch_path, backup

    async def _audit(self, mission: Mission, outcome: AuditOutcome, record: ApplyRecord) -> None:
        await self._storage.audit.append(AuditEntry(
            mission_id=mission.mission_id, agent=record.decided_by or "operator", tool=APPLY_TOOL,
            args={"repo": mission.repo_path, "files": ", ".join(record.files)[:300]}, action_class="AMBER",
            outcome=outcome, approved_by_human=record.status == ApplyStatus.APPLIED, error=record.error or "",
            display=f"apply {record.status}"))
