import asyncio

import pytest

from app.core.errors import ConflictError, NotFoundError
from app.domain.events import EventKind
from app.domain.governance import ApprovalStatus, AuditOutcome, MissionLimits
from app.infrastructure.bus.memory import InMemoryEventBus
from app.infrastructure.storage.memory import InMemoryStorage
from app.security.policy import Risk, Verdict
from app.services.approval_gate import ApprovalGate
from app.services.event_publisher import EventPublisher
from app.services.governance import Governor

AMBER = Verdict(Risk.HIGH, "pip install changes the environment", "install")


@pytest.fixture
def storage():
    return InMemoryStorage()


@pytest.fixture
def events(storage):
    return EventPublisher(storage.events, InMemoryEventBus())


def make_gate(storage, events, **kwargs):
    return ApprovalGate(storage.approvals, events._bus, events, **kwargs)


async def _pending(storage, mission_id="msn_test"):
    for _ in range(200):
        pending = await storage.approvals.list(mission_id, status=ApprovalStatus.PENDING)
        if pending:
            return pending[0]
        await asyncio.sleep(0.01)
    raise AssertionError("no approval was requested")


async def test_human_approves(storage, events, ctx):
    gate = make_gate(storage, events)
    waiting = asyncio.create_task(gate.request(ctx, "run_command", {"command": "pip install x"}, AMBER))
    approval = await _pending(storage)
    assert approval.agent == "Takeshi" and approval.action_class == "AMBER"

    decided = await gate.resolve(approval.approval_id, True, by="kaushal", note="ok for demo")
    assert decided.status == ApprovalStatus.APPROVED
    assert await waiting is True

    kinds = [e.kind for e in await storage.events.list("msn_test")]
    assert kinds == [EventKind.APPROVAL_REQUEST, EventKind.APPROVAL_RESPONSE]
    response = (await storage.events.list("msn_test"))[-1].data
    assert response["approved"] is True and response["decided_by"] == "kaushal"


async def test_human_denies_and_cannot_answer_twice(storage, events, ctx):
    gate = make_gate(storage, events)
    waiting = asyncio.create_task(gate.request(ctx, "run_command", {}, AMBER))
    approval = await _pending(storage)
    await gate.resolve(approval.approval_id, False, by="kaushal")
    assert await waiting is False
    with pytest.raises(ConflictError):
        await gate.resolve(approval.approval_id, True, by="someone-else")
    with pytest.raises(NotFoundError):
        await gate.resolve("apr_missing", True, by="x")


async def test_no_answer_expires_as_denial(storage, events, ctx):
    gate = make_gate(storage, events, timeout_s=0.05)
    assert await gate.request(ctx, "run_command", {}, AMBER) is False
    [approval] = await storage.approvals.list("msn_test")
    assert approval.status == ApprovalStatus.EXPIRED and approval.decided_by == "system:timeout"


async def test_auto_approved_categories(storage, events, ctx):
    gate = make_gate(storage, events, auto_approve_categories={"install"})
    assert await gate.request(ctx, "run_command", {}, AMBER) is True
    assert await storage.approvals.list("msn_test") == []


async def test_expire_pending_on_stop(storage, events, ctx):
    gate = make_gate(storage, events)
    waiting = asyncio.create_task(gate.request(ctx, "run_command", {}, AMBER))
    await _pending(storage)
    assert await gate.expire_pending("msn_test", "mission stopped") == 1
    assert await waiting is False


async def test_gate_through_registry(storage, events, ctx, registry):
    ctx.approver = gate = make_gate(storage, events)
    ctx.events = events
    run = asyncio.create_task(registry.execute("run_command", {"command": "pip install requests"}, ctx))
    approval = await _pending(storage)
    await gate.resolve(approval.approval_id, False, by="kaushal")
    result = await run
    assert not result.ok and result.meta["outcome"] == "denied"


async def test_loop_guard_and_audit(storage, events, ctx, registry):
    ctx.events = events
    governor = Governor("msn_test", registry, storage.audit, events, MissionLimits(max_identical_failures=2))
    bad_edit = {"path": "calc.py", "old": "does not exist", "new": "x"}

    first = await governor.execute("edit_file", bad_edit, ctx)
    second = await governor.execute("edit_file", bad_edit, ctx)
    third = await governor.execute("edit_file", bad_edit, ctx)
    assert not first.ok and not second.ok
    assert third.meta["outcome"] == AuditOutcome.REFUSED_LOOP
    assert "already failed 2 times" in third.error and "not found" in third.error

    # A different call is still allowed, and success resets the agent's failure streak.
    good = await governor.execute("edit_file", {"path": "calc.py", "old": "return a - b", "new": "return a + b"}, ctx)
    assert good.ok
    assert await governor.execute("run_command", {"command": "rm -rf ."}, ctx)

    outcomes = [e.outcome for e in await storage.audit.list("msn_test")]
    assert outcomes == [AuditOutcome.FAILED, AuditOutcome.FAILED, AuditOutcome.REFUSED_LOOP,
                        AuditOutcome.OK, AuditOutcome.BLOCKED]
    blocked = (await storage.audit.list("msn_test"))[-1]
    assert blocked.action_class == "RED" and blocked.args == {"command": "rm -rf ."}


async def test_consecutive_failure_guard(storage, events, ctx, registry):
    governor = Governor("msn_test", registry, storage.audit, events,
                        MissionLimits(max_consecutive_failures=2, max_identical_failures=10))
    for path in ("a.py", "b.py"):
        assert not (await governor.execute("read_file", {"path": path}, ctx)).ok
    refused = await governor.execute("read_file", {"path": "c.py"}, ctx)
    assert refused.meta["outcome"] == AuditOutcome.REFUSED_LOOP and "in a row" in refused.error


async def test_budget_limits(storage, events, ctx, registry):
    governor = Governor("msn_test", registry, storage.audit, events, MissionLimits(max_tool_calls=2))
    assert (await governor.execute("list_dir", {}, ctx)).ok
    assert (await governor.execute("list_dir", {}, ctx)).ok
    refused = await governor.execute("list_dir", {}, ctx)
    assert refused.meta["outcome"] == AuditOutcome.REFUSED_BUDGET
    await governor.execute("list_dir", {}, ctx)
    decisions = [e for e in await storage.events.list("msn_test") if e.kind == EventKind.DECISION]
    assert len(decisions) == 1 and decisions[0].data["decision"] == "stop"  # announced once

    tokens = Governor("msn_test", registry, storage.audit, events, MissionLimits(max_llm_tokens=100))
    tokens.record_llm_usage("Atlas", 150)
    assert "token limit" in tokens.exhausted
    assert tokens.budget.snapshot()["tokens_by_agent"] == {"Atlas": 150}
