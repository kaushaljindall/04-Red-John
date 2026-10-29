import asyncio

from fastapi.testclient import TestClient

from app.domain.events import EventKind
from app.main import create_app


def test_health_tools_and_event_stream(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/api/v1/health/live").json() == {"status": "ok"}
        ready = client.get("/api/v1/health/ready")
        assert ready.status_code == 200 and ready.json()["checks"]["llm_configured"] is False
        assert client.get("/api/v1/tools").json()["count"] >= 13
        assert "x-request-id" in client.get("/api/v1/health/live").headers

        events = app.state.container.events
        portal = client.portal  # run coroutines on the app's event loop
        portal.call(events.log, "msn_1", "stored before connect")

        with client.websocket_connect("/api/v1/realtime/missions/msn_1") as ws:
            replayed = ws.receive_json()
            assert replayed["seq"] == 1 and replayed["data"]["text"] == "stored before connect"

            portal.call(events.publish, "msn_1", EventKind.HIRE, {"agent": "Daichi"})
            live = ws.receive_json()
            assert live["seq"] == 2 and live["kind"] == "hire" and live["data"]["agent"] == "Daichi"

        # Resume after seq 1 replays only what was missed.
        with client.websocket_connect("/api/v1/realtime/missions/msn_1?after_seq=1") as ws:
            assert ws.receive_json()["seq"] == 2

        listed = client.get("/api/v1/realtime/missions/msn_1/events").json()["events"]
        assert [e["seq"] for e in listed] == [1, 2]


def test_mission_lifecycle_over_http(settings, sample_repo):
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post("/api/v1/missions", json={"request": "Fix add()", "repo_path": str(sample_repo)})
        assert created.status_code == 201
        mid = created.json()["mission"]["mission_id"]
        assert client.post("/api/v1/missions", json={"request": "x y z", "repo_path": "Z:/nope"}).status_code == 422

        listed = client.get("/api/v1/missions").json()["missions"]
        assert [m["mission_id"] for m in listed] == [mid]
        assert client.get(f"/api/v1/missions/{mid}").json()["mission"]["status"] == "created"
        assert client.get("/api/v1/missions/msn_nope").status_code == 404

        board = app.state.container.board
        portal = client.portal
        portal.call(lambda: board.raise_issue(mid, __import__("app.domain.mission", fromlist=["OpenIssue"]).OpenIssue(
            raised_by="Kenji", kind="clarification", text="floats?", needs_human=True)))
        issue_id = client.get(f"/api/v1/missions/{mid}").json()["mission"]["issues"][0]["id"]
        resolved = client.post(f"/api/v1/missions/{mid}/issues/{issue_id}/resolve", json={"resolution": "yes"})
        assert resolved.json()["issue"]["status"] == "resolved"

        # Mycel's snapshot path returns the control-room shape plus mission state.
        snap = client.get(f"/api/v1/realtime/snapshot/{mid}").json()
        assert {"clock", "logs", "hires", "agents", "complete", "pendingApprovals"} <= snap.keys()
        assert snap["mission"]["status"] == "created" and snap["agents"]["Atlas"]["phase"] == "working"
        assert any("needs a human" in log["text"] for log in snap["logs"])

        # Mycel's WebSocket path streams the same events.
        with client.websocket_connect(f"/api/v1/realtime/sessions/{mid}") as ws:
            assert ws.receive_json()["kind"] == "mission_created"

        stopped = client.post(f"/api/v1/missions/{mid}/stop", json={"reason": "demo over"}).json()
        assert stopped["mission"]["status"] == "stopped"
        again = client.post(f"/api/v1/missions/{mid}/stop", json={})  # idempotent
        assert again.status_code == 200 and again.json()["mission"]["status"] == "stopped"
        kinds = [e["kind"] for e in client.get(f"/api/v1/realtime/missions/{mid}/events").json()["events"]]
        assert kinds.count("mission_status") == 1
        # An issue can only be answered once.
        assert client.post(f"/api/v1/missions/{mid}/issues/{issue_id}/resolve",
                           json={"resolution": "again"}).status_code == 422


def test_roster_and_hiring_over_http(settings, sample_repo):
    app = create_app(settings)
    with TestClient(app) as client:
        agents = client.get("/api/v1/agents").json()["agents"]
        assert len(agents) == 16 and agents[0]["name"] == "Atlas"
        mid = client.post("/api/v1/missions", json={"request": "Fix add()", "repo_path": str(sample_repo)}).json()["mission"]["mission_id"]

        hired = client.post(f"/api/v1/missions/{mid}/hires", json={"skills": ["xss", "javascript"], "reason": "audit UI"})
        assert hired.status_code == 200 and hired.json()["hired"][0]["agent"] == "Kaito"
        assert client.post(f"/api/v1/missions/{mid}/hires", json={"skills": ["cooking"], "reason": "lunch"}).status_code == 422
        assert client.get(f"/api/v1/missions/{mid}/hires").json()["active"] == ["Kaito"]

        released = client.post(f"/api/v1/missions/{mid}/hires/Kaito/release", json={})
        assert released.json()["released"]["release_reason"] == "released by operator"
        assert client.get(f"/api/v1/missions/{mid}/hires").json()["active"] == []
        assert client.post(f"/api/v1/missions/{mid}/hires/Kaito/release", json={}).status_code == 422

        snap = client.get(f"/api/v1/missions/{mid}/snapshot").json()
        assert snap["hires"][0]["agent"] == "Kaito" and snap["agents"]["Kaito"]["phase"] == "done"


def test_approval_over_http_and_audit(settings, sample_repo):
    from app.security.policy import Risk, Verdict
    from app.tools.registry import ToolContext

    app = create_app(settings)
    with TestClient(app) as client:
        container = app.state.container
        portal = client.portal
        workspace = portal.call(container.workspaces.create, "msn_2", sample_repo)
        ctx = ToolContext(mission_id="msn_2", agent="Riku", workspace=workspace, settings=settings,
                          approver=container.approvals, events=container.events)
        governor = container.governor("msn_2")

        # The agent blocks on approval in the background while the operator answers over HTTP.
        future = portal.start_task_soon(governor.execute, "read_file", {"path": ".env"}, ctx)
        pending = []
        for _ in range(100):
            pending = client.get("/api/v1/missions/msn_2/approvals?status=pending").json()["approvals"]
            if pending:
                break
            portal.call(asyncio.sleep, 0.02)
        assert pending and pending[0]["category"] == "secrets"

        # Mycel's modal path still works.
        answer = client.post(f"/api/v1/realtime/approvals/{pending[0]['approval_id']}/respond", json={"approved": True})
        assert answer.status_code == 200 and answer.json()["approved"] is True
        result = future.result(timeout=10)
        assert result.ok and "SECRET=" in result.content

        again = client.post(f"/api/v1/approvals/{pending[0]['approval_id']}/respond", json={"approved": False})
        assert again.status_code == 409 and again.json()["error"]["code"] == "conflict"
        assert client.post("/api/v1/approvals/apr_nope/respond", json={"approved": True}).status_code == 404

        audit = client.get("/api/v1/missions/msn_2/audit").json()
        assert audit["count"] == 1 and audit["entries"][0]["approved_by_human"] is True
        assert audit["by_outcome"] == {"ok": 1}

        kinds = [e["kind"] for e in client.get("/api/v1/realtime/missions/msn_2/events").json()["events"]]
        assert kinds == ["tool_call", "approval_request", "approval_response", "tool_result"]
