from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from app.core.config import Environment, Settings
from app.security.sandbox import WorkspaceManager
from app.tools.builtin import build_registry
from app.tools.registry import AllowAllApprover, ToolContext, ToolRegistry


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """A tiny Python project with one seeded bug (add subtracts) and a planted secret."""
    repo = tmp_path / "repo"
    _write(repo, "calc.py", """
        API_KEY = "gsk_abcdefghijklmnopqrstuvwxyz123456"


        def add(a, b):
            return a - b


        def mul(a, b):
            return a * b
    """)
    _write(repo, "tests/test_calc.py", """
        from calc import add, mul


        def test_add():
            assert add(2, 3) == 5


        def test_mul():
            assert mul(2, 3) == 6
    """)
    _write(repo, "pytest.ini", "[pytest]\npythonpath = .\n")
    _write(repo, ".env", "SECRET=hunter2\n")
    return repo


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment=Environment.TEST,
        data_dir=tmp_path / "var",
        command_timeout_s=120,
        groq_api_keys=[],
        gemini_api_keys=[],
    )


@pytest.fixture
def registry() -> ToolRegistry:
    return build_registry()


@pytest.fixture
async def ctx(settings: Settings, sample_repo: Path) -> ToolContext:
    manager = WorkspaceManager(settings.workspaces_dir)
    workspace = await manager.create("msn_test", sample_repo)
    return ToolContext(mission_id="msn_test", agent="Takeshi", workspace=workspace, settings=settings,
                       approver=AllowAllApprover())
