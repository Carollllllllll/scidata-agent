from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from scidata_agent.api import main as api_main
from scidata_agent.api import task_manager as task_manager_module
from scidata_agent.api.task_manager import TaskManager


class _FakeResult:
    def __init__(self, task_id: str, research_question: str, files: list[str], figure_path: Path):
        self.task_id = task_id
        self.status = "completed"
        self.research_question = research_question
        self.files = files
        self.figure_path = figure_path

    def model_dump(self, mode: str = "json", by_alias: bool = True) -> dict:
        source_path = self.files[0] if self.files else None
        return {
            "task_id": self.task_id,
            "status": self.status,
            "research_question": self.research_question,
            "summary": {"records_extracted": 1, "dynamic_records_extracted": 1},
            "quality_report": {"record_count": 1, "warning_count": 0},
            "dynamic_records": [
                {
                    "record_id": "dyn_1",
                    "table_name": "results",
                    "fields": {"metric": "PCE", "value": 25.1},
                    "source_file": source_path,
                    "confidence": 0.94,
                    "warnings": [],
                }
            ],
            "figures": [
                {
                    "figure_id": "fig_1",
                    "source_file": source_path,
                    "page": 1,
                    "image_path": str(self.figure_path),
                }
            ],
            "source_catalog": [
                {
                    "source_id": "src_1",
                    "title": "Uploaded source",
                    "artifacts": [{"artifact_id": "artifact_1", "local_path": source_path}],
                }
            ],
            "export_files": {"csv": str(self.figure_path.parent / "result.csv")},
        }


class _FakeAgent:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def run(self, research_question: str, files: list[str], *, task_id: str, **kwargs):
        task_dir = self.output_dir / task_id
        figures_dir = task_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "result.csv").write_text("metric,value\nPCE,25.1\n", encoding="utf-8")
        figure_path = figures_dir / "figure_1.png"
        figure_path.write_bytes(b"fake-png")
        (task_dir / "agent_monitor.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": "2026-08-20T12:00:00+00:00",
                    "event_type": "step",
                    "step": "export",
                    "status": "completed",
                    "message": "Export completed.",
                    "data": {"files": [{"name": "metrics.csv", "path": files[0] if files else None}]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return _FakeResult(task_id, research_question, files, figure_path)


def _wait_until_terminal(client: TestClient, task_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        response = client.get(f"/api/tasks/{task_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"task {task_id} did not finish")


def test_request_body_limit_rejects_before_calling_application() -> None:
    application_called = False
    sent: list[dict] = []

    async def inner(_scope, _receive, _send):
        nonlocal application_called
        application_called = True

    async def receive():
        return {"type": "http.request", "body": b"12345", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = api_main.RequestBodyLimitMiddleware(
        inner,
        max_body_bytes=4,
        paths={"/api/analyze"},
    )
    asyncio.run(
        middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/analyze",
                "headers": [(b"content-length", b"5")],
            },
            receive,
            send,
        )
    )

    assert application_called is False
    assert next(message for message in sent if message["type"] == "http.response.start")["status"] == 413


def test_api_task_lifecycle_and_public_assets(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(task_manager_module, "SciDataAgent", _FakeAgent)
    upload_dir = tmp_path / "uploads"
    manager = TaskManager(
        output_dir=tmp_path / "outputs",
        state_dir=tmp_path / "tasks",
        upload_dir=upload_dir,
        max_workers=1,
    )
    monkeypatch.setattr(api_main, "TASK_MANAGER", manager)
    monkeypatch.setattr(api_main, "UPLOAD_DIR", upload_dir)
    client = TestClient(api_main.app)

    try:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        submitted = client.post(
            "/api/analyze",
            data={"research_question": "提取上传表格中的光伏效率和证据"},
            files={"files": ("metrics.csv", b"metric,value\nPCE,25.1\n", "text/csv")},
        )
        assert submitted.status_code == 202
        task_id = submitted.json()["task_id"]

        completed = _wait_until_terminal(client, task_id)
        assert completed["status"] == "completed"
        assert completed["result"]["dynamic_records"][0]["fields"]["metric"] == "PCE"
        assert completed["result"]["dynamic_records"][0]["source_file"] == "metrics.csv"
        assert completed["uploads"][0]["asset_url"].startswith(f"/api/tasks/{task_id}/assets/upload/")
        assert completed["result"]["figures"][0]["image_url"].startswith(
            f"/api/tasks/{task_id}/assets/output/"
        )
        serialized = json.dumps(completed, ensure_ascii=False)
        assert str(tmp_path) not in serialized

        tasks = client.get("/api/tasks")
        assert tasks.status_code == 200
        assert tasks.json()["count"] == 1
        assert tasks.json()["tasks"][0]["result"] is None

        events = client.get(f"/api/tasks/{task_id}/events?tail=10")
        assert events.status_code == 200
        assert events.json()["events"][0]["step"] == "export"
        assert "data" not in events.json()["events"][0]
        assert str(tmp_path) not in json.dumps(events.json(), ensure_ascii=False)

        detailed_events = client.get(f"/api/tasks/{task_id}/events?tail=10&include_data=true")
        assert detailed_events.status_code == 200
        assert "data" in detailed_events.json()["events"][0]
        assert str(tmp_path) not in json.dumps(detailed_events.json(), ensure_ascii=False)

        review = client.post(
            f"/api/tasks/{task_id}/reviews/dyn_1",
            json={"decision": "approved", "note": "Evidence checked against page 1."},
        )
        assert review.status_code == 200
        assert review.json()["decision"] == "approved"
        refreshed = client.get(f"/api/tasks/{task_id}").json()
        assert refreshed["review_decisions"]["dyn_1"]["note"] == "Evidence checked against page 1."

        exported = client.get(f"/api/tasks/{task_id}/export?format=csv")
        assert exported.status_code == 200
        assert b"PCE,25.1" in exported.content

        figure_url = completed["result"]["figures"][0]["image_url"]
        figure = client.get(figure_url)
        assert figure.status_code == 200
        assert figure.content == b"fake-png"
    finally:
        manager.shutdown()


def test_api_validation_and_http_errors(tmp_path, monkeypatch) -> None:
    manager = TaskManager(
        output_dir=tmp_path / "outputs",
        state_dir=tmp_path / "tasks",
        upload_dir=tmp_path / "uploads",
        max_workers=1,
    )
    monkeypatch.setattr(api_main, "TASK_MANAGER", manager)
    monkeypatch.setattr(api_main, "UPLOAD_DIR", tmp_path / "uploads")
    client = TestClient(api_main.app)

    try:
        missing = client.get("/api/tasks/not_a_real_task")
        assert missing.status_code == 404
        assert missing.json()["detail"]["code"] == "TASK_NOT_FOUND"

        unsupported_upload = client.post(
            "/api/analyze",
            data={"research_question": "这是一个有效长度的科研问题"},
            files={"files": ("notes.txt", b"not supported", "text/plain")},
        )
        assert unsupported_upload.status_code == 415
        assert unsupported_upload.json()["detail"]["code"] == "UNSUPPORTED_FILE_TYPE"

        invalid_magic = client.post(
            "/api/analyze",
            data={"research_question": "这是一个有效长度的科研问题"},
            files=[
                ("files", ("valid.csv", b"metric,value\nFID,1.2\n", "text/csv")),
                ("files", ("fake.pdf", b"this is not a PDF", "application/pdf")),
            ],
        )
        assert invalid_magic.status_code == 415
        assert invalid_magic.json()["detail"]["code"] == "INVALID_FILE_CONTENT"
        assert not any((tmp_path / "uploads").iterdir())

        invalid_question = client.post("/api/discover", data={"research_question": "x"})
        assert invalid_question.status_code == 422

        unknown_review = client.post(
            "/api/tasks/not_a_real_task/reviews/dyn_1",
            json={"decision": "approved"},
        )
        assert unknown_review.status_code == 404
    finally:
        manager.shutdown()


def test_api_security_handles_non_ascii_and_cors_preflight(monkeypatch) -> None:
    monkeypatch.setenv("SCIDATA_API_TOKEN", "expected-token")
    api_main._RATE_LIMIT_BUCKETS.clear()
    client = TestClient(api_main.app)

    preflight = client.options(
        "/api/analyze",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"

    unauthorized = client.get(
        "/api/tasks/not_a_real_task",
        headers=[(b"origin", b"http://localhost:5173"), (b"authorization", b"Bearer \xff")],
    )
    assert unauthorized.status_code == 401
    assert unauthorized.json()["detail"]["code"] == "UNAUTHORIZED"
    assert unauthorized.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_invalid_token_attempts_are_rate_limited_before_auth(monkeypatch) -> None:
    monkeypatch.setenv("SCIDATA_API_TOKEN", "expected-token")
    monkeypatch.setattr(api_main, "RATE_LIMIT_PER_MINUTE", 1)
    api_main._RATE_LIMIT_BUCKETS.clear()
    client = TestClient(api_main.app)

    first = client.get("/api/tasks/not_a_real_task", headers={"Authorization": "Bearer wrong-one"})
    second = client.get("/api/tasks/not_a_real_task", headers={"Authorization": "Bearer wrong-two"})

    assert first.status_code == 401
    assert second.status_code == 429
    assert second.json()["detail"]["code"] == "RATE_LIMITED"
