from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scidata_agent.api import task_manager as task_manager_module
from scidata_agent.api.task_manager import TaskManager


class _FakeResult:
    task_id = "20260819_120000_000_abcd"
    status = "completed"

    def model_dump(self, mode: str = "json", by_alias: bool = True) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "summary": {"records_extracted": 1},
            "quality_report": {"record_count": 1},
            "export_files": {"csv": "server-only\\result.csv"},
        }


class _FakeAgent:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def run(self, research_question: str, files: list[str], *, task_id: str, **kwargs):
        assert research_question == "test question"
        assert task_id
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "result.csv").write_text("field,value\nanswer,1\n", encoding="utf-8")
        (task_dir / "agent_monitor.jsonl").write_text(
            json.dumps({
                "timestamp": "2026-08-19T12:00:00+00:00",
                "step": "export",
                "status": "completed",
                "message": "export completed",
                "data": {},
            })
            + "\n",
            encoding="utf-8",
        )
        return _FakeResult()


class _FakeFailedResult:
    status = "failed"

    def __init__(self, task_id: str):
        self.task_id = task_id

    def model_dump(self, mode: str = "json", by_alias: bool = True) -> dict:
        return {
            "task_id": self.task_id,
            "status": "failed",
            "summary": {"records_extracted": 0},
            "quality_report": {"record_count": 0},
            "processing_log": ["Task failed: Qwen/Bailian API key not configured."],
            "export_files": {},
        }


class _FakeFailedAgent:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def run(self, research_question: str, files: list[str], *, task_id: str, **kwargs):
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "agent_monitor.jsonl").write_text(
            json.dumps(
                {
                    "timestamp": "2026-08-20T12:00:00+00:00",
                    "event_type": "error",
                    "step": "ensure_llm_ready",
                    "status": "failed",
                    "message": "Qwen/Bailian API key not configured.",
                    "data": {},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return _FakeFailedResult(task_id)


class _FakePartialResult:
    status = "partial"

    def __init__(self, task_id: str):
        self.task_id = task_id

    def model_dump(self, mode: str = "json", by_alias: bool = True) -> dict:
        return {
            "task_id": self.task_id,
            "status": "partial",
            "summary": {"records_extracted": 1},
            "coverage_report": {
                "decision": "continue",
                "coverage_score": 0.4,
                "missing_requirements": ["experimental setup"],
            },
            "processing_log": [
                "Agent task produced partial results: coverage remains incomplete after all configured action iterations."
            ],
            "export_files": {},
        }


class _FakePartialAgent:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def run(self, research_question: str, files: list[str], *, task_id: str, **kwargs):
        return _FakePartialResult(task_id)


def _wait_for_status(manager: TaskManager, task_id: str, expected: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        task = manager.get_task(task_id)
        if task.get("status") == expected:
            return task
        time.sleep(0.01)
    raise AssertionError(f"task did not reach {expected}: {manager.get_task(task_id)}")


def test_task_manager_persists_completion_and_download_allowlist(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(task_manager_module, "SciDataAgent", _FakeAgent)
    manager = TaskManager(tmp_path / "outputs", tmp_path / "tasks", max_workers=1)
    try:
        task = manager.submit(
            task_id="20260819_120000_000_abcd",
            research_question="test question",
            files=[],
            run_options={},
        )
        completed = _wait_for_status(manager, task["task_id"], "completed")

        assert completed["result"]["summary"]["records_extracted"] == 1
        assert manager.download_path(task["task_id"], "csv").name == "result.csv"
        assert manager.download_path(task["task_id"], "../task_state.json") is None
        assert "csv" in manager.download_urls(task["task_id"])
        assert (tmp_path / "tasks" / task["task_id"] / "task_state.json").is_file()
        assert completed["owner_pid"] == os.getpid()
    finally:
        manager.shutdown()


def test_reconciliation_preserves_live_owner_and_fails_orphan_only(tmp_path) -> None:
    state_dir = tmp_path / "tasks"
    live_dir = state_dir / "20260820_120000_000_live"
    orphan_dir = state_dir / "20260820_120000_000_orphan"
    live_dir.mkdir(parents=True)
    orphan_dir.mkdir(parents=True)
    (live_dir / "task_state.json").write_text(
        json.dumps({"task_id": live_dir.name, "status": "running", "owner_pid": os.getpid()}),
        encoding="utf-8",
    )
    (orphan_dir / "task_state.json").write_text(
        json.dumps({"task_id": orphan_dir.name, "status": "queued"}),
        encoding="utf-8",
    )

    manager = TaskManager(tmp_path / "outputs", state_dir, max_workers=1)
    try:
        # Construction/import is side-effect free; reconciliation is a server-start action.
        assert manager.get_task(orphan_dir.name)["status"] == "queued"
        manager.reconcile_interrupted_tasks()
        assert manager.get_task(live_dir.name)["status"] == "running"
        orphan = manager.get_task(orphan_dir.name)
        assert orphan["status"] == "failed"
        assert orphan["error"]["code"] == "TASK_INTERRUPTED"
    finally:
        manager.shutdown()


def test_task_manager_lists_tasks_and_resolves_scoped_assets(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    state_dir = tmp_path / "tasks"
    upload_dir = tmp_path / "uploads"
    manager = TaskManager(output_dir, state_dir, max_workers=1, upload_dir=upload_dir)
    task_id = "20260820_120000_000_abcd"
    manager._write_state(
        task_id,
        {
            "task_id": task_id,
            "status": "completed",
            "research_question": "question",
            "created_at": "2026-08-20T12:00:00+00:00",
        },
    )
    figure = output_dir / task_id / "figures" / "figure.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    figure.write_bytes(b"png")
    upload = upload_dir / task_id / "paper.pdf"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(b"pdf")

    try:
        tasks = manager.list_tasks()
        assert [task["task_id"] for task in tasks] == [task_id]
        assert manager.asset_url(task_id, figure).endswith("/assets/output/figures/figure.png")
        assert manager.asset_url(task_id, upload).endswith("/assets/upload/paper.pdf")
        assert manager.asset_path(task_id, "output", "figures/figure.png") == figure
        assert manager.asset_path(task_id, "output", "../task_state.json") is None
        assert manager.asset_url(task_id, tmp_path / "secret.txt") is None
    finally:
        manager.shutdown()


def test_task_manager_preserves_failed_step_and_message(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(task_manager_module, "SciDataAgent", _FakeFailedAgent)
    manager = TaskManager(tmp_path / "outputs", tmp_path / "tasks", max_workers=1)
    try:
        task = manager.submit(
            task_id="20260820_120000_000_fail",
            research_question="test question",
            files=[],
            run_options={},
        )
        failed = _wait_for_status(manager, task["task_id"], "failed")
        assert failed["current_step"] == "ensure_llm_ready"
        assert failed["error"] == {
            "code": "AGENT_TASK_FAILED",
            "message": "Qwen/Bailian API key not configured.",
        }
        assert failed["message"] == "Qwen/Bailian API key not configured."
    finally:
        manager.shutdown()


def test_task_manager_persists_partial_status_and_coverage_message(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(task_manager_module, "SciDataAgent", _FakePartialAgent)
    manager = TaskManager(tmp_path / "outputs", tmp_path / "tasks", max_workers=1)
    try:
        task = manager.submit(
            task_id="20260824_120000_000_partial",
            research_question="test partial result",
            files=[],
            run_options={},
        )
        partial = _wait_for_status(manager, task["task_id"], "partial")
        expected = "Agent task partially completed; coverage requirements remain unsatisfied."
        assert partial["current_step"] == "partial"
        assert partial["message"] == expected
        assert partial["error"] == {"code": "AGENT_TASK_PARTIAL", "message": expected}
        assert partial["result"]["coverage_report"]["missing_requirements"] == ["experimental setup"]
    finally:
        manager.shutdown()


def test_task_list_uses_latest_monitor_step_for_active_task(tmp_path) -> None:
    output_dir = tmp_path / "outputs"
    manager = TaskManager(output_dir, tmp_path / "tasks", max_workers=1)
    task_id = "20260820_120000_000_live"
    manager._write_state(
        task_id,
        {
            "task_id": task_id,
            "status": "running",
            "research_question": "question",
            "current_step": "starting",
        },
    )
    monitor = output_dir / task_id / "agent_monitor.jsonl"
    monitor.parent.mkdir(parents=True, exist_ok=True)
    monitor.write_text(
        json.dumps(
            {
                "timestamp": "2026-08-20T12:00:01+00:00",
                "event_type": "step",
                "step": "dynamic_extraction",
                "status": "started",
                "message": "dynamic_extraction started.",
                "data": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        listed = manager.list_tasks()
        assert listed[0]["current_step"] == "dynamic_extraction"
        assert listed[0]["message"] == "dynamic_extraction started."
    finally:
        manager.shutdown()


def test_concurrent_review_decisions_do_not_overwrite_each_other(tmp_path) -> None:
    manager = TaskManager(tmp_path / "outputs", tmp_path / "tasks", max_workers=1)
    task_id = "20260820_120000_000_reviews"
    records = [
        {"record_id": f"dyn_{index}", "table_name": "results", "fields": {"value": index}}
        for index in range(24)
    ]
    manager._write_state(task_id, {"task_id": task_id, "status": "completed"})
    manager._write_json(
        manager._task_state_dir(task_id) / "result_payload.json",
        {"dynamic_records": records},
    )

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(manager.set_review_decision, task_id, record["record_id"], "approved")
                for record in records
            ]
            for future in futures:
                future.result()

        assert set(manager.review_decisions(task_id)) == {record["record_id"] for record in records}
        assert manager.get_task(task_id)["status"] == "completed"
    finally:
        manager.shutdown()


def test_invalid_pending_task_environment_falls_back(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SCIDATA_MAX_PENDING_TASKS", "not-a-number")
    manager = TaskManager(tmp_path / "outputs", tmp_path / "tasks", max_workers=1)
    try:
        assert manager.max_pending_tasks == 8
    finally:
        manager.shutdown()


def test_task_list_only_reads_payloads_for_requested_page(tmp_path, monkeypatch) -> None:
    manager = TaskManager(tmp_path / "outputs", tmp_path / "tasks", max_workers=1)
    for index in range(30):
        task_id = f"20260820_120000_{index:03d}_task"
        manager._write_state(
            task_id,
            {
                "task_id": task_id,
                "status": "completed",
                "created_at": f"2026-08-20T12:00:{index:02d}+00:00",
            },
        )
        manager._write_json(
            manager._task_state_dir(task_id) / "result_payload.json",
            {"summary": {"records_extracted": index}},
        )

    original_read_json = manager._read_json
    payload_reads: list[Path] = []

    def tracking_read_json(path: Path):
        if path.name == "result_payload.json":
            payload_reads.append(path)
        return original_read_json(path)

    monkeypatch.setattr(manager, "_read_json", tracking_read_json)
    try:
        tasks = manager.list_tasks(limit=4)
        assert len(tasks) == 4
        assert len(payload_reads) == 4
    finally:
        manager.shutdown()


def test_running_task_cancels_at_agent_checkpoint(tmp_path, monkeypatch) -> None:
    class CancellableAgent:
        def __init__(self, output_dir: Path):
            self.output_dir = output_dir

        def run(self, _question, _files, *, task_id: str, cancel_check, **_kwargs):
            deadline = time.monotonic() + 2
            while not cancel_check() and time.monotonic() < deadline:
                time.sleep(0.005)
            return _FakeFailedResult(task_id)

    monkeypatch.setattr(task_manager_module, "SciDataAgent", CancellableAgent)
    manager = TaskManager(tmp_path / "outputs", tmp_path / "tasks", max_workers=1)
    task_id = "20260820_120000_000_cancel"
    try:
        manager.submit(
            task_id=task_id,
            research_question="test question",
            files=[],
            run_options={},
        )
        _wait_for_status(manager, task_id, "running")
        assert manager.cancel_task(task_id) is True
        cancelled = _wait_for_status(manager, task_id, "cancelled")
        assert cancelled["error"]["code"] == "TASK_CANCELLED"
    finally:
        manager.shutdown()
