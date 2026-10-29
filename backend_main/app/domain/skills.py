"""
Skill taxonomy: one canonical name per skill, plus the words that imply it.

Maya matches agents to work on canonical skills, so "pytest", "unit tests"
and "testing" all mean the same thing, and a request that mentions "XSS" or
"leaked key" is understood to need security. Deterministic on purpose:
hiring decisions must be explainable and reproducible.
"""
from __future__ import annotations

import re

from app.domain.mission import TaskKind

# canonical skill → words/phrases that imply it (lower-case; matched on word boundaries)
SKILL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "orchestration": ("orchestrate", "coordinate", "coordination"),
    "hiring": ("hire", "staffing", "recruit"),
    "requirements": ("requirement", "requirements", "acceptance criteria", "ambiguous", "ambiguity", "clarify",
                     "user story", "spec", "specification"),
    "recon": ("reconnaissance", "codebase", "conventions", "project structure", "explore", "investigate repo"),
    "planning": ("plan", "decompose", "decomposition", "breakdown", "dependencies", "schedule", "replan"),
    "implementation": ("implement", "build", "add", "feature", "create", "endpoint", "function", "support"),
    "python": ("python", "py", "pytest", "flask", "django", "fastapi", "pip", "pandas"),
    "javascript": ("javascript", "js", "typescript", "ts", "node", "nodejs", "npm", "jest", "vitest"),
    "frontend": ("frontend", "front-end", "ui", "ux", "react", "vue", "svelte", "html", "css", "component",
                 "page", "button", "form", "layout", "dom"),
    "backend": ("backend", "back-end", "api", "server", "endpoint", "route", "database", "sql", "service", "rest",
                "http"),
    "debugging": ("debug", "debugging", "bug", "defect", "error", "exception", "traceback", "crash", "broken",
                  "fix", "failing", "fails", "wrong", "incorrect"),
    "refactoring": ("refactor", "refactoring", "cleanup", "clean up", "style", "lint", "linting", "readability",
                    "formatting", "maintainability", "quality issue"),
    "test-execution": ("run tests", "test run", "test suite", "ci", "build check", "lint check"),
    "test-authoring": ("write tests", "write a test", "add tests", "add a test", "new tests", "unit test",
                       "unit tests", "test coverage", "coverage", "regression test", "regression tests", "tdd",
                       "test case", "test cases"),
    "verification": ("verify", "verification", "acceptance", "validate", "confirm", "evidence"),
    "code-review": ("review", "code review", "correctness", "consistency", "maintainable"),
    "security": ("security", "secure", "vulnerability", "vulnerable", "exploit", "injection", "sql injection",
                 "xss", "csrf", "secret", "secrets", "api key", "leaked", "credential", "password", "token",
                 "unsafe", "sanitize", "sanitise", "auth", "authentication", "authorization", "eval"),
    "arbitration": ("dispute", "disagreement", "arbitrate", "conflict", "escalate"),
    "reporting": ("report", "summary", "summarize", "document", "documentation", "readme", "changelog", "run log"),
}

CANONICAL_SKILLS = frozenset(SKILL_SYNONYMS)

# What a task of each kind needs at minimum. Language/area skills are added from the task text.
KIND_SKILLS: dict[TaskKind, tuple[str, ...]] = {
    TaskKind.ANALYSIS: (),            # analysis varies (requirements, recon…): its explicit skills decide
    TaskKind.PLAN: ("planning",),
    TaskKind.IMPLEMENT: ("implementation",),
    TaskKind.FIX: ("debugging",),
    TaskKind.TEST: ("test-execution",),
    TaskKind.REVIEW: ("code-review",),
    TaskKind.SECURITY: ("security",),
    TaskKind.VERIFY: ("verification",),
    TaskKind.REPORT: ("reporting",),
}

# Skills that describe *where* work happens rather than *what* kind of work it is.
AREA_SKILLS = frozenset({"python", "javascript", "frontend", "backend"})

_PATTERNS = {
    skill: re.compile(r"(?<![\w-])(" + "|".join(re.escape(w) for w in sorted({skill, *words}, key=len, reverse=True))
                      + r")(?![\w-])", re.I)
    for skill, words in SKILL_SYNONYMS.items()
}


def normalize_skill(name: str) -> str | None:
    """Map any skill word to its canonical name, or None if unknown."""
    key = name.strip().lower().replace("_", "-")
    if key in CANONICAL_SKILLS:
        return key
    for skill, pattern in _PATTERNS.items():
        if pattern.fullmatch(key):
            return skill
    return None


def normalize_skills(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        skill = normalize_skill(name)
        if skill and skill not in out:
            out.append(skill)
    return out


def infer_skills(text: str, *, only: frozenset[str] | None = None) -> list[str]:
    """Skills a piece of text implies, in taxonomy order."""
    return [skill for skill, pattern in _PATTERNS.items()
            if (only is None or skill in only) and pattern.search(text)]
