import pytest

from app.agents.runtime import AgentRuntime, trim_history
from app.core.errors import ExternalServiceError
from app.domain.agents import AgentDefinition, RunStatus
from app.domain.events import EventKind
from app.domain.governance import AuditOutcome, MissionLimits
from app.domain.mission import AgentMessage, MessageKind, MissionStatus
from app.infrastructure.bus.memory import InMemoryEventBus
from app.infrastructure.storage.memory import InMemoryStorage
from app.security.sandbox import WorkspaceManager
from app.services.event_publisher import EventPublisher
from app.services.governance import Governor
from app.services.mission_board import MissionBoard
from app.tools.registry import AllowAllApprover
from tests.fake_llm import ScriptedProvider, call, finish, gateway, reply

DEV_TOOLS = ["read_file", "edit_file", "search_text", "run_tests", "git_diff", "send_message", "record_finding",
             "read_inbox", "mission_state"]


def definition(name="Takeshi", team="forge", tools=DEV_TOOLS, **kw) -> AgentDefinition:
    return AgentDefinition(agent_id=f"{team}_{name.lower()}", name=name, team=team, role="Backend Engineer",
                           skills=["python"], tools=tools, system_prompt="Write correct, idiomatic Python.", **kw)


class Harness:
    def __init__(self, settings, sample_repo):
        self.settings = settings
        self.repo = sample_repo
        self.storage = InMemoryStorage()
        self.events = EventPublisher(self.storage.events, InMemoryEventBus())
        self.workspaces = WorkspaceManager(settings.workspaces_dir)
        self.board = MissionBoard(self.storage, self.events, self.workspaces)

    async def mission(self):
        m = await self.board.create_mission("Fix add() so the test suite passes", str(self.repo))
        await self.board.set_requirements(m.mission_id, ["add() adds"], ["All tests pass"], by="Kenji")
        return m.mission_id

    def runtime(self, provider, registry):
        return AgentRuntime(llm=gateway(provider), tools=registry, board=self.board, events=self.events,
                            workspaces=self.workspaces, settings=self.settings, approver=AllowAllApprover())

    def governor(self, mission_id, registry, **limits):
        return Governor(mission_id, registry, self.storage.audit, self.events, MissionLimits(**limits), board=self.board)


@pytest.fixture
def h(settings, sample_repo):
    return Harness(settings, sample_repo)


async def test_developer_fixes_bug_end_to_end(h, registry):
    mid = await h.mission()
    provider = ScriptedProvider([
        reply(call("run_tests"), content="First, reproduce the failure."),
        reply(call("read_file", path="calc.py")),
        reply(call("edit_file", path="calc.py", old="return a - b", new="return a + b")),
        reply(call("run_tests")),
        lambda req: finish(summary="Fixed add(); tests pass", evidence=[_evidence_id(req)]),
    ])
    result = await h.runtime(provider, registry).run(definition(), mid, h.governor(mid, registry))

    assert result.status == RunStatus.DONE and result.tool_calls == 4 and result.iterations == 5
    assert len(result.evidence_ids) == 2 and result.tokens == 500
    # The model saw the real failure, then the real pass, with evidence ids it could cite.
    first_test = provider.requests[1].messages[-1]["content"]
    assert '"evidence_id"' in first_test and "test_add" in first_test
    assert result.data["evidence"] == [result.evidence_ids[1]]

    [failed, passed] = await h.board.evidence(mid, kind="test_run")
    assert failed.passed is False and passed.passed is True
    kinds = [e.kind for e in await h.storage.events.list(mid)]
    assert kinds.count(EventKind.START) == 1 and kinds.count(EventKind.FINISH) == 1
    assert kinds.index(EventKind.START) < kinds.index(EventKind.TOOL_CALL) < kinds.index(EventKind.FINISH)
    logs = [e.data["text"] for e in await h.storage.events.list(mid) if e.kind == EventKind.LOG]
    assert "Takeshi: First, reproduce the failure." in logs
    assert any(t.startswith("Takeshi → edit_file: edited calc.py") for t in logs)
    assert len(await h.storage.audit.list(mid, agent="Takeshi")) == 4


async def test_tools_outside_the_grant_are_refused(h, registry):
    mid = await h.mission()
    provider = ScriptedProvider([reply(call("run_command", command="python -V")), finish()])
    result = await h.runtime(provider, registry).run(definition(tools=["read_file"]), mid, h.governor(mid, registry))
    assert result.status == RunStatus.DONE
    assert "Unknown tool 'run_command'" in provider.last_tool_results()[0]
    names = [t["function"]["name"] for t in provider.requests[0].tools]
    assert names == ["read_file", "finish"]


async def test_finish_contract_and_output_schema(h, registry):
    mid = await h.mission()
    kenji = definition("Kenji", "planning", tools=["read_file"],
                       output_schema={"required": ["objectives", "acceptance_criteria"]})
    provider = ScriptedProvider([
        reply(content="I think we are done."),                       # no tool call → nudged
        reply(call("finish", status="maybe", summary="x")),          # bad status
        finish(summary="analysed", objectives=["a"]),                # missing a required key
        finish(summary="analysed", objectives=["a"], acceptance_criteria=["b"]),
    ])
    result = await h.runtime(provider, registry).run(kenji, mid, h.governor(mid, registry))
    assert result.status == RunStatus.DONE and result.data["acceptance_criteria"] == ["b"]
    assert "must act through your tools" in provider.requests[1].messages[-1]["content"]
    assert "status must be one of" in provider.requests[2].messages[-1]["content"]
    assert "missing required field(s): acceptance_criteria" in provider.requests[3].messages[-1]["content"]
    assert "data` must include: objectives, acceptance_criteria" in provider.requests[0].messages[0]["content"]


async def test_limits_end_the_run(h, registry):
    mid = await h.mission()
    looping = ScriptedProvider([reply(call("read_file", path="calc.py"))] * 3)
    result = await h.runtime(looping, registry).run(definition(max_iterations=3), mid, h.governor(mid, registry))
    assert result.status == RunStatus.INCOMPLETE and result.stop_reason == "iteration_limit"

    silent = ScriptedProvider([reply(content="hmm")] * 3)
    result = await h.runtime(silent, registry).run(definition(), mid, h.governor(mid, registry))
    assert result.status == RunStatus.INCOMPLETE and result.stop_reason == "no_finish"

    broke = ScriptedProvider([reply(call("read_file", path="calc.py")), reply(call("read_file", path="calc.py"))])
    result = await h.runtime(broke, registry).run(definition(), mid, h.governor(mid, registry, max_tool_calls=1))
    assert result.status == RunStatus.BLOCKED and result.stop_reason == "budget"


async def test_llm_outage_and_bad_json(h, registry):
    mid = await h.mission()
    bad = call("read_file")
    bad.raw_arguments = "{path: calc.py"
    provider = ScriptedProvider([reply(bad), ExternalServiceError("groq down")])
    result = await h.runtime(provider, registry).run(definition(), mid, h.governor(mid, registry))
    assert result.status == RunStatus.FAILED and result.stop_reason == "llm_unavailable"
    assert "not valid JSON" in provider.requests[1].messages[-1]["content"]


async def test_repeated_failing_call_hits_loop_guard(h, registry):
    mid = await h.mission()
    same = reply(call("edit_file", path="calc.py", old="nope", new="x"))
    provider = ScriptedProvider([same, same, same, same, finish(status="blocked", summary="cannot find the code")])
    governor = h.governor(mid, registry, max_identical_failures=3)
    result = await h.runtime(provider, registry).run(definition(), mid, governor)
    assert result.status == RunStatus.BLOCKED
    assert "Repeating it will not help" in provider.requests[4].messages[-1]["content"]
    outcomes = [e.outcome for e in await h.storage.audit.list(mid)]
    assert outcomes[-1] == AuditOutcome.REFUSED_LOOP


async def test_agents_communicate_through_the_board(h, registry):
    """Aiko reports a failure to the Forge team; Takeshi gets it in his briefing and replies."""
    mid = await h.mission()
    aiko = definition("Aiko", "sentinel", tools=["run_tests", "send_message", "record_finding"])
    aiko_llm = ScriptedProvider([
        reply(call("run_tests")),
        lambda req: reply(call("send_message", to="@forge", kind="finding", subject="test_add fails",
                               body="add(2, 3) returned -1", evidence_ids=[_evidence_id(req)])),
        reply(call("record_finding", kind="test_failure", severity="high", title="add() subtracts",
                   file="calc.py", line=5)),
        finish(summary="reported"),
    ])
    aiko_result = await h.runtime(aiko_llm, registry).run(aiko, mid, h.governor(mid, registry))
    assert aiko_result.status == RunStatus.DONE

    takeshi_llm = ScriptedProvider([
        reply(call("send_message", to="Aiko", kind="answer", subject="on it", body="fixing add()")),
        reply(call("read_inbox")),
        finish(),
    ])
    result = await h.runtime(takeshi_llm, registry).run(definition(), mid, h.governor(mid, registry))
    briefing = takeshi_llm.requests[0].messages[1]["content"]
    assert "finding from Aiko to @forge: test_add fails" in briefing
    assert "add() subtracts calc.py:5" in briefing                     # open finding on the board
    assert '"messages": []' in takeshi_llm.requests[2].messages[-1]["content"]  # already read in the briefing
    assert result.inbox_seq == 1

    [aiko_msg, reply_msg] = await h.board.messages(mid)
    assert aiko_msg.evidence_ids == aiko_result.evidence_ids and reply_msg.to == "Aiko"
    assert [m.subject for m in await h.board.inbox(mid, "Aiko")] == ["on it"]


async def test_stopped_mission_ends_the_run(h, registry):
    mid = await h.mission()
    await h.board.set_status(mid, MissionStatus.STOPPED, by="operator")
    provider = ScriptedProvider([finish()])
    result = await h.runtime(provider, registry).run(definition(), mid, h.governor(mid, registry))
    assert result.status == RunStatus.BLOCKED and result.stop_reason == "mission_stopped"
    assert provider.requests == []


def test_trim_history_keeps_tool_pairs():
    head = [{"role": "system", "content": "s"}, {"role": "user", "content": "brief"}]
    steps = []
    for i in range(10):
        steps.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        steps.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 1000})
    trimmed = trim_history(head + steps, budget=4000)
    assert trimmed[:2] == head and trimmed[2]["content"].startswith("[context note]")
    body = trimmed[3:]
    assert body[0]["role"] == "assistant"
    ids = [m["tool_call_id"] for m in body if m["role"] == "tool"]
    calls = [m["tool_calls"][0]["id"] for m in body if m["role"] == "assistant"]
    assert ids == calls and calls[-1] == "c9"
    # Trimming again replaces the note instead of stacking another one.
    again = trim_history(trimmed + steps[-2:], budget=4000)
    assert sum(1 for m in again if str(m.get("content", "")).startswith("[context note]")) == 1


def _evidence_id(request) -> str:
    import json
    return json.loads(request.messages[-1]["content"])["evidence_id"]


async def test_finish_written_as_text_is_accepted(h, registry):
    """gpt-oss sometimes writes its finish call as JSON text instead of calling the tool (seen live on Groq)."""
    mid = await h.mission()
    provider = ScriptedProvider([reply(content='{\n  "status": "done",\n  "summary": "Fixed add(); tests pass"\n}')])
    result = await h.runtime(provider, registry).run(definition(), mid, h.governor(mid, registry))
    assert result.status == RunStatus.DONE and result.summary == "Fixed add(); tests pass"

    # Text that is not a valid finish still gets nudged.
    provider = ScriptedProvider([reply(content='{"status": "great"}'), finish()])
    result = await h.runtime(provider, registry).run(definition(), mid, h.governor(mid, registry))
    assert result.status == RunStatus.DONE and len(provider.requests) == 2


async def test_null_optional_arguments_are_ignored(h, registry):
    """Models send null for optional arguments; the exported schema allows it and null means 'not given'."""
    mid = await h.mission()
    schema = registry.get("run_tests").schema()["function"]["parameters"]["properties"]
    assert schema["command"]["type"] == ["string", "null"]
    provider = ScriptedProvider([reply(call("run_tests", command=None, cwd=None)),
                                 reply(call("search_text", query="def add", max_results=5)), finish()])
    await h.runtime(provider, registry).run(definition(), mid, h.governor(mid, registry))
    audit = await h.storage.audit.list(mid)
    assert audit[0].args == {} and audit[0].outcome == AuditOutcome.FAILED  # tests ran (and failed)
    assert audit[1].outcome == AuditOutcome.OK and audit[1].args["limit"] == "5"


async def test_rejected_tool_call_is_retried_not_fatal(h, registry):
    from app.infrastructure.llm.types import ToolCallRejected

    mid = await h.mission()
    provider = ScriptedProvider([ToolCallRejected("parameters for tool run_tests did not match schema"), finish()])
    result = await h.runtime(provider, registry).run(definition(), mid, h.governor(mid, registry))
    assert result.status == RunStatus.DONE
    assert "rejected before it ran" in provider.requests[1].messages[-1]["content"]


async def test_rejected_call_carrying_a_complete_result_is_accepted(h, registry):
    """Seen live: gpt-oss 'called' a tool named json with Kenji's whole result; Groq rejected it 4 times."""
    from app.infrastructure.llm.types import ToolCallRejected

    mid = await h.mission()
    kenji = definition("Kenji", "planning", tools=["read_file"],
                       output_schema={"required": ["objectives", "acceptance_criteria"]})
    generation = '{"name": "json", "arguments": {"objectives": ["fix add"], "acceptance_criteria": ["tests pass"]}}'
    provider = ScriptedProvider([ToolCallRejected("attempted to call tool 'json'", generation)])
    result = await h.runtime(provider, registry).run(kenji, mid, h.governor(mid, registry))
    assert result.status == RunStatus.DONE and result.data["acceptance_criteria"] == ["tests pass"]

    # A rejected call that is not a complete result gets the list of real tool names back.
    provider = ScriptedProvider([ToolCallRejected("attempted to call tool 'json'", '{"name": "json", "arguments": {}}'),
                                 finish(summary="ok", objectives=["a"], acceptance_criteria=["b"])])
    await h.runtime(provider, registry).run(kenji, mid, h.governor(mid, registry))
    assert "Your tools are: read_file, finish" in provider.requests[1].messages[-1]["content"]


async def test_namespaced_tool_names_are_mapped_to_real_tools(h, registry):
    """Seen live: gpt-oss called `repo_browser.list_dir`."""
    from app.agents.runtime import canonical_tool_name

    assert canonical_tool_name("repo_browser.list_dir", ["list_dir"]) == "list_dir"
    assert canonical_tool_name("functions.finish", []) == "finish"
    assert canonical_tool_name("repo_browser.open_file", ["read_file"]) == "repo_browser.open_file"
    mid = await h.mission()
    provider = ScriptedProvider([reply(call("repo_browser.list_dir", path=".")), finish()])
    result = await h.runtime(provider, registry).run(definition(tools=["list_dir"]), mid, h.governor(mid, registry))
    assert result.status == RunStatus.DONE and result.tool_calls == 1
    assert '"entries"' in provider.requests[1].messages[-1]["content"]


async def test_requests_fit_the_provider_limit_and_413_shrinks_context(h, registry, settings):
    """Seen live: Groq's on_demand tier rejects any request over 8,000 tokens (413), which looked like an outage."""
    import json

    from app.infrastructure.llm.types import ContextTooLarge

    settings.llm_max_request_tokens, settings.llm_max_output_tokens = 6000, 1500
    budget_chars = int((6000 - 1500) * settings.llm_chars_per_token)
    mid = await h.mission()
    reads = [reply(call("read_file", path="calc.py")) for _ in range(12)]
    provider = ScriptedProvider([*reads, ContextTooLarge("request too large"), finish()])
    result = await h.runtime(provider, registry).run(definition(tools=DEV_TOOLS), mid, h.governor(mid, registry))
    assert result.status == RunStatus.DONE
    for request in provider.requests:
        size = len(json.dumps(request.messages)) + len(json.dumps(request.tools))
        assert size <= budget_chars * 1.05, size
        assert request.max_tokens == 1500
    # After the 413 the next request was smaller than the one that was refused.
    assert len(json.dumps(provider.requests[-1].messages)) < len(json.dumps(provider.requests[-2].messages))
    # Schemas were compacted to fit: no per-parameter descriptions.
    assert "description" not in json.dumps(provider.requests[0].tools[0]["function"]["parameters"])


async def test_trimmed_agents_keep_a_digest_and_get_a_step_warning(h, registry, settings):
    """Seen live: with history trimmed, Hana forgot her work and redid it until the iteration limit."""
    settings.llm_max_request_tokens, settings.llm_max_output_tokens = 5000, 1500
    mid = await h.mission()
    steps = [reply(call("run_tests"))] + [reply(call("read_file", path="calc.py")) for _ in range(6)] + [finish()]
    provider = ScriptedProvider(steps)
    result = await h.runtime(provider, registry).run(definition(tools=DEV_TOOLS, max_iterations=9), mid,
                                                     h.governor(mid, registry))
    assert result.status == RunStatus.DONE
    notes = [m["content"] for r in provider.requests for m in r.messages
             if m["role"] == "user" and str(m["content"]).startswith("[context note]")]
    assert notes and "- run_tests() → FAILED: tests: 1 passed, 1 failed" in notes[-1]
    assert "evd_" in notes[-1]                                # ids survive in the digest
    warnings = [m["content"] for m in provider.requests[-1].messages if "step(s) left" in str(m.get("content"))]
    assert len(warnings) == 1
