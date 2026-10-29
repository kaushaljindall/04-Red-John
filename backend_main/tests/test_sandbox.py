from pathlib import Path

import pytest

from app.core.errors import PolicyViolation, ValidationFailed
from app.security.sandbox import WorkspaceManager


async def test_workspace_is_isolated_copy_with_baseline(settings, sample_repo: Path):
    manager = WorkspaceManager(settings.workspaces_dir)
    ws = await manager.create("msn_a", sample_repo)

    assert ws.root != sample_repo.resolve()
    assert (ws.root / "calc.py").read_text() == (sample_repo / "calc.py").read_text()
    assert len(ws.baseline_commit) == 40
    # A second handle (as another worker would build it) finds the same baseline.
    manager._cache.clear()
    assert manager.get("msn_a").baseline_commit == ws.baseline_commit


async def test_paths_cannot_escape(settings, sample_repo: Path):
    ws = await WorkspaceManager(settings.workspaces_dir).create("msn_b", sample_repo)
    assert ws.resolve("tests/test_calc.py").is_file()
    for bad in ("../outside.txt", "..", str(sample_repo / "calc.py"), "tests/../../x"):
        with pytest.raises(PolicyViolation):
            ws.resolve(bad)


async def test_rejects_missing_repo(settings, tmp_path: Path):
    with pytest.raises(ValidationFailed):
        await WorkspaceManager(settings.workspaces_dir).create("msn_c", tmp_path / "nope")


async def test_allowed_roots(settings, sample_repo: Path, tmp_path: Path):
    manager = WorkspaceManager(settings.workspaces_dir, allowed_roots=[tmp_path / "elsewhere"])
    with pytest.raises(PolicyViolation):
        await manager.create("msn_d", sample_repo)
