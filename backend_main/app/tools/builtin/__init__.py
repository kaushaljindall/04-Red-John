"""Built-in tools. `build_registry()` returns a registry with all of them."""
from __future__ import annotations

from app.tools.builtin import board, commands, filesystem, git, search, security_scan
from app.tools.registry import ToolRegistry


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for module in (filesystem, search, commands, git, security_scan, board):
        module.register(registry)
    return registry
