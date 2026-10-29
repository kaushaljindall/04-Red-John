from app.domain.events import EventKind
from app.tools.registry import DenyAllApprover


async def test_catalog_has_core_tools(registry):
    names = set(registry.names())
    assert {"read_file", "write_file", "edit_file", "list_dir", "find_files", "search_text", "run_command",
            "run_tests", "git_status", "git_diff", "git_log", "secret_scan", "static_scan"} <= names
    assert all(s["type"] == "function" for s in registry.schemas(["read_file", "run_tests"]))


async def test_read_and_argument_repair(registry, ctx):
    result = await registry.execute("read_file", {"file_path": "calc.py"}, ctx)
    assert result.ok and "def add" in result.content
    assert result.meta["repairs"] == ["treated 'file_path' as 'path'"]


async def test_unknown_tool_and_grants(registry, ctx):
    assert "Did you mean 'read_file'" in (await registry.execute("read_fil", {}, ctx)).error
    denied = await registry.execute("write_file", {"path": "x.py", "content": ""}, ctx, allowed=["read_file"])
    assert not denied.ok and "Unknown tool" in denied.error


async def test_path_escape_is_blocked(registry, ctx):
    result = await registry.execute("read_file", {"path": "../../etc/passwd"}, ctx)
    assert not result.ok and "Blocked by policy" in result.error


async def test_secret_file_needs_approval(registry, ctx):
    ctx.approver = DenyAllApprover()
    result = await registry.execute("read_file", {"path": ".env"}, ctx)
    assert not result.ok and result.meta.get("denied")


async def test_feedback_loop_tests_fail_then_pass_after_edit(registry, ctx):
    failing = await registry.execute("run_tests", {}, ctx)
    assert not failing.ok
    assert failing.evidence["kind"] == "test_run"
    assert failing.evidence["summary"]["failed"] == 1
    assert failing.evidence["summary"]["passed"] == 1
    assert any("test_add" in t for t in failing.evidence["summary"]["failing_tests"])

    edit = await registry.execute("edit_file", {"path": "calc.py", "old": "return a - b", "new": "return a + b"}, ctx)
    assert edit.ok, edit.error

    passing = await registry.execute("run_tests", {}, ctx)
    assert passing.ok, passing.error
    assert passing.evidence["summary"]["passed"] == 2

    diff = await registry.execute("git_diff", {}, ctx)
    assert diff.ok and diff.content["files_changed"] == 1
    assert "+    return a + b" in diff.content["diff"]


async def test_write_new_file_shows_in_diff(registry, ctx):
    assert (await registry.execute("write_file", {"path": "pkg/new.py", "content": "X = 1\n"}, ctx)).ok
    status = await registry.execute("git_status", {}, ctx)
    assert any("pkg/new.py" in line for line in status.content["entries"])
    diff = await registry.execute("git_diff", {"stat_only": True}, ctx)
    assert diff.content["files_changed"] == 1


async def test_edit_requires_unique_match(registry, ctx):
    result = await registry.execute("edit_file", {"path": "calc.py", "old": "def ", "new": "def  "}, ctx)
    assert not result.ok and "appears" in result.error


async def test_search_and_find(registry, ctx):
    found = await registry.execute("search_text", {"query": "def add"}, ctx)
    assert found.ok and found.content["matches"][0].startswith("calc.py:4:")
    # .env is never surfaced by search
    assert not (await registry.execute("search_text", {"query": "hunter2"}, ctx)).content["matches"]
    files = await registry.execute("find_files", {"pattern": "test_*.py"}, ctx)
    assert files.content["matches"] == ["tests/test_calc.py"]


async def test_run_command_policy(registry, ctx):
    assert "Blocked by policy" in (await registry.execute("run_command", {"command": "rm -rf ."}, ctx)).error
    assert "Blocked by policy" in (await registry.execute("run_command", {"command": "python ../../evil.py"}, ctx)).error
    ctx.approver = DenyAllApprover()
    assert "approval" in (await registry.execute("run_command", {"command": "pip install requests"}, ctx)).error
    ok = await registry.execute("run_command", {"command": "python -m pytest -q tests/test_calc.py::test_mul"}, ctx)
    assert ok.ok and ok.content["exit_code"] == 0


async def test_security_scans(registry, ctx):
    secrets = await registry.execute("secret_scan", {"changed_only": False}, ctx)
    assert secrets.ok
    groq = [f for f in secrets.content["findings"] if f["rule"] == "groq-key"]
    assert groq and "abcdefghij" not in groq[0]["match"]  # redacted

    await registry.execute("write_file", {"path": "db.py", "content":
        "def find(cur, name):\n    cur.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n"}, ctx)
    static = await registry.execute("static_scan", {}, ctx)
    assert [f["rule"] for f in static.content["findings"]] == ["py-sql-format"]


async def test_events_are_published(registry, ctx):
    published = []

    class Sink:
        async def publish(self, mission_id, kind, data=None):
            published.append((kind, data))

    ctx.events = Sink()
    await registry.execute("run_tests", {}, ctx)
    kinds = [k for k, _ in published]
    assert kinds == [EventKind.TOOL_CALL, EventKind.TOOL_RESULT]  # evidence is the Governor's job


def test_truncated_tool_output_keeps_the_evidence_id():
    """Found when tool output was capped to fit Groq's request limit: the id at the end was cut off."""
    import json

    from app.tools.registry import ToolResult

    result = ToolResult(False, content={"output": "x" * 5000, "exit_code": 1}, error="exit 1: " + "y" * 3000,
                        meta={"evidence_id": "evd_123"})
    text = result.to_model_text(limit=1500)
    assert len(text) <= 1500
    body = json.loads(text)
    assert body["evidence_id"] == "evd_123" and body["ok"] is False and "truncated" in body["content"]
    ok_text = ToolResult(True, content="z" * 5000, meta={"evidence_id": "evd_9"}).to_model_text(limit=800)
    assert ok_text.startswith("[evidence_id: evd_9]") and len(ok_text) <= 800
