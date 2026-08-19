from __future__ import annotations

import json
import time
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
    finally:
        manager.shutdown()
