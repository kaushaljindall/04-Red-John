"""
Security inspection tools: secret detection and a rule-based static scan for
unsafe operations and injection risks. Findings are evidence for the Shadow
Council; matched secrets are redacted before they reach any model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from app.security.sandbox import VENDOR_DIRS
from app.tools.builtin.git import changed_files
from app.tools.registry import ToolContext, ToolRegistry, ToolResult, run_blocking

MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern[str]
    severity: str
    message: str
    extensions: tuple[str, ...] = ()


SECRET_RULES = [
    Rule("aws-access-key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"), "critical", "AWS access key id"),
    Rule("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "critical", "GitHub token"),
    Rule("groq-key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"), "critical", "Groq API key"),
    Rule("openai-key", re.compile(r"\bsk-(proj-)?[A-Za-z0-9_-]{20,}\b"), "critical", "OpenAI-style API key"),
    Rule("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), "critical", "Google API key"),
    Rule("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), "high", "Slack token"),
    Rule("private-key", re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"), "critical", "Private key block"),
    Rule("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "high", "Hard-coded JWT"),
    Rule("connection-string", re.compile(r"\b(mongodb(\+srv)?|postgres(ql)?|mysql|redis|amqp)://[^\s:/'\"]+:[^\s@'\"]+@"), "high",
         "Connection string with embedded password"),
    Rule("generic-secret", re.compile(
        r"""(?i)\b(api[_-]?key|secret[_-]?key|secret|password|passwd|auth[_-]?token|access[_-]?token)\b\s*[:=]\s*['"][^'"\s]{8,}['"]"""),
        "medium", "Hard-coded credential assignment"),
]

_PY = (".py",)
_JS = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
STATIC_RULES = [
    Rule("py-eval", re.compile(r"(?<![\w.])(eval|exec)\s*\("), "high", "eval/exec on dynamic input enables code injection", _PY),
    Rule("py-shell-true", re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True"), "high", "subprocess with shell=True enables command injection", _PY),
    Rule("py-os-system", re.compile(r"\bos\.(system|popen)\s*\("), "high", "os.system/os.popen runs through a shell", _PY),
    Rule("py-pickle", re.compile(r"\b(pickle|cPickle|dill)\.loads?\s*\("), "high", "unpickling untrusted data executes code", _PY),
    Rule("py-yaml-load", re.compile(r"\byaml\.load\s*\((?![^)]*SafeLoader)"), "medium", "yaml.load without SafeLoader", _PY),
    Rule("py-sql-format", re.compile(
        r"""\.execute(many)?\s*\(\s*(f['"]|['"][^'"]*['"]\s*(%|\.format\s*\()|[\w.]+\s*\+)"""),
        "high", "SQL built by string formatting — use parameters", _PY),
    Rule("py-weak-hash", re.compile(r"\bhashlib\.(md5|sha1)\s*\("), "low", "MD5/SHA-1 are unsuitable for passwords or signatures", _PY),
    Rule("py-tls-off", re.compile(r"verify\s*=\s*False"), "medium", "TLS certificate verification disabled", _PY),
    Rule("py-debug", re.compile(r"\b(app\.run\([^)]*debug\s*=\s*True|DEBUG\s*=\s*True)"), "low", "debug mode enabled", _PY),
    Rule("py-tempfile", re.compile(r"\btempfile\.mktemp\s*\("), "low", "tempfile.mktemp is race-prone", _PY),
    Rule("js-eval", re.compile(r"(?<![\w.])(eval\s*\(|new\s+Function\s*\()"), "high", "eval/new Function enables code injection", _JS),
    Rule("js-innerhtml", re.compile(r"\.(innerHTML|outerHTML)\s*=|dangerouslySetInnerHTML"), "medium", "raw HTML injection (XSS risk)", _JS),
    Rule("js-doc-write", re.compile(r"document\.write\s*\("), "medium", "document.write (XSS risk)", _JS),
    Rule("js-child-exec", re.compile(r"child_process['\"]?\)?\.exec(Sync)?\s*\(|\bexec(Sync)?\s*\(\s*`"), "high", "shell command execution", _JS),
    Rule("js-sql-template", re.compile(r"\.(query|execute)\s*\(\s*`[^`]*\$\{"), "high", "SQL built with template literals", _JS),
]


def _redact(text: str) -> str:
    return text if len(text) <= 8 else f"{text[:4]}…{text[-2:]} ({len(text)} chars)"


def _iter_files(root: Path, only: list[str] | None) -> list[Path]:
    if only is not None:
        return [root / rel for rel in only if (root / rel).is_file()]
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in VENDOR_DIRS)
        files.extend(Path(dirpath, name) for name in sorted(filenames))
    return files


def _scan(ctx: ToolContext, rules: list[Rule], only: list[str] | None, redact: bool, limit: int = 200) -> list[dict]:
    ws = ctx.workspace
    findings: list[dict] = []
    for file in _iter_files(ws.root, only):
        try:
            if file.stat().st_size > MAX_FILE_BYTES:
                continue
            data = file.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8000]:
            continue
        rel = ws.relative(file)
        suffix = file.suffix.lower()
        applicable = [r for r in rules if not r.extensions or suffix in r.extensions]
        if not applicable:
            continue
        for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//")) and not redact:
                continue  # commented-out code is not an active risk (secrets in comments still are)
            for rule in applicable:
                match = rule.pattern.search(line)
                if not match:
                    continue
                snippet = _redact(match.group(0)) if redact else stripped[:200]
                findings.append({"rule": rule.id, "severity": rule.severity, "file": rel, "line": number,
                                 "message": rule.message, "match": snippet})
                if len(findings) >= limit:
                    return findings
    return findings


def _result(kind: str, findings: list[dict], scope: str) -> ToolResult:
    by_severity: dict[str, int] = {}
    for finding in findings:
        by_severity[finding["severity"]] = by_severity.get(finding["severity"], 0) + 1
    return ToolResult(
        True,
        content={"findings": findings, "count": len(findings), "by_severity": by_severity, "scope": scope},
        display=f"{kind}: {len(findings)} finding(s) in {scope}",
        evidence={"kind": kind, "count": len(findings), "by_severity": by_severity, "scope": scope,
                  "findings": findings[:50]},
    )


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        "secret_scan",
        "Scan for exposed secrets (API keys, tokens, private keys, passwords, connection strings). "
        "Matches are redacted. By default scans only files changed in this mission.",
        {"type": "object", "properties": {
            "changed_only": {"type": "boolean", "description": "Only files changed since the baseline (default true)"}}},
        category="security", evidence=True)
    async def secret_scan(ctx: ToolContext, changed_only: bool = True) -> ToolResult:
        only = await changed_files(ctx) if changed_only else None
        findings = await run_blocking(_scan, ctx, SECRET_RULES, only, True)
        return _result("secret_scan", findings, "changed files" if changed_only else "whole repository")

    @registry.tool(
        "static_scan",
        "Rule-based scan for unsafe operations and injection risks (eval, shell=True, SQL string building, "
        "unsafe deserialisation, XSS sinks, disabled TLS). By default scans only files changed in this mission.",
        {"type": "object", "properties": {
            "changed_only": {"type": "boolean", "description": "Only files changed since the baseline (default true)"}}},
        category="security", evidence=True)
    async def static_scan(ctx: ToolContext, changed_only: bool = True) -> ToolResult:
        only = await changed_files(ctx) if changed_only else None
        findings = await run_blocking(_scan, ctx, STATIC_RULES, only, False)
        return _result("static_scan", findings, "changed files" if changed_only else "whole repository")
