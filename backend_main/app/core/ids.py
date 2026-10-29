"""Identifier and clock helpers, kept in one place so tests can reason about them."""
from __future__ import annotations

import time
import uuid


def new_id(prefix: str) -> str:
    """Short, prefixed, sortable-enough id, e.g. ``msn_3f9a1c2b7d4e``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_ms() -> int:
    return int(time.time() * 1000)
