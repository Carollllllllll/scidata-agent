from __future__ import annotations

import copy
import json
import re
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from scidata_agent.agent.schemas import timestamp_task_id
from scidata_agent.agent.scidata_agent import SciDataAgent


TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class TaskManager:
    """Run Agent jobs outside the request thread and persist task state.

    The scientific pipeline remains owned by ``SciDataAgent``. This class only
    manages API-facing lifecycle state, result snapshots, and safe file lookup.
    """

    EXPORT_FILES = {
        "csv": "result.csv",
        "json": "result.json",
        "quality_report": "quality_report.json",
        "processing_log": "processing_log.json",
        "source_discovery_plan": "source_discovery_plan.json",
        "source_catalog": "source_catalog.json",
        "source_selection": "source_selection_plan.json",
        "source_triage": "source_triage.json",
        "paper_survey": "paper_survey.json",
        "dynamic_schema": "dynamic_schema.json",
        "dynamic_records": "dynamic_records.json",
        "needs_review": "needs_review.json",
        "chart_extractions": "chart_extractions.json",
        "chart_validation": "chart_validation_report.json",
        "summary": "summary.json",
        "final_report": "final_report.md",
    }

    def __init__(self, output_dir: Path, state_dir: Path, max_workers: int = 2):
        self.output_dir = output_dir.resolve()
        self.state_dir = state_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scidata-agent")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = Lock()

    def submit(
        self,
        *,
        task_id: str | None = None,
        research_question: str,
        files: list[str],
        run_options: dict[str, Any],
        auto_fetch_arxiv: bool = True,
    ) -> dict[str, Any]:
        task_id = task_id or timestamp_task_id()
        self._write_state(
            task_id,
            {
                "task_id": task_id,
                "status": "queued",
                "research_question": research_question,
                "created_at": _now(),
                "updated_at": _now(),
                "current_step": "queued",
                "message": "Task queued.",
                "files_count": len(files),
            },
        )
        future = self._executor.submit(
            self._run,
            task_id,
            research_question,
            files,
            run_options,
            auto_fetch_arxiv,
        )
        with self._lock:
            self._futures[task_id] = future
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> dict[str, Any]:
        state = self._read_state(task_id)
        if state is None:
            return {"task_id": task_id, "status": "not_found"}

        event = self._latest_monitor_event(task_id)
        response = copy.deepcopy(state)
        if event:
            if event.get("event_type") in {"step", "progress"}:
                response["current_step"] = event.get("step") or response.get("current_step")
            response["message"] = event.get("message") or response.get("message")
            response["updated_at"] = event.get("timestamp") or response.get("updated_at")
            response["event"] = event
            event_data = event.get("data") or {}
            if "progress_index" in event_data or "progress_total" in event_data:
                response["progress"] = {
                    "current": event_data.get("progress_index"),
                    "total": event_data.get("progress_total"),
                }

        payload = self._read_json(self._task_state_dir(task_id) / "result_payload.json")
        if payload:
            response.update(
                {
                    "result": payload,
                    "summary": payload.get("summary"),
                    "quality_report": payload.get("quality_report"),
                    "download_urls": self.download_urls(task_id),
                }
            )
        return response

    def get_events(self, task_id: str, tail: int = 100) -> dict[str, Any]:
        state = self._read_state(task_id)
        if state is None:
            return {"task_id": task_id, "status": "not_found", "events": []}
        tail = max(1, min(int(tail), 500))
        log_path = self._monitor_path(task_id)
        events: list[dict[str, Any]] = []
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-tail:]:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return {"task_id": task_id, "status": state.get("status"), "events": events}

    def download_path(self, task_id: str, export_format: str) -> Path | None:
        task_dir = self._task_output_dir(task_id)
        filename = self.EXPORT_FILES.get(export_format)
        if not filename or task_dir is None:
            return None
        path = (task_dir / filename).resolve()
        if path.parent != task_dir or not path.is_file():
            return None
        return path

    def download_urls(self, task_id: str) -> dict[str, str]:
        return {
            export_format: f"/api/tasks/{task_id}/export?format={export_format}"
            for export_format in self.EXPORT_FILES
            if self.download_path(task_id, export_format) is not None
        }

    def shutdown(self, wait: bool = True) -> None:
        """Stop the worker pool during application shutdown or test teardown."""
        self._executor.shutdown(wait=wait)

    def _run(
        self,
        task_id: str,
        research_question: str,
        files: list[str],
        run_options: dict[str, Any],
        auto_fetch_arxiv: bool,
    ) -> None:
        self._update_state(task_id, status="running", current_step="starting", message="Agent task started.")
        try:
            agent = SciDataAgent(output_dir=self.output_dir)
            result = agent.run(
                research_question,
                files,
                task_id=task_id,
                auto_fetch_arxiv=auto_fetch_arxiv,
                **run_options,
            )
            payload = result.model_dump(mode="json", by_alias=True)
            self._write_json(self._task_state_dir(task_id) / "result_payload.json", payload)
            self._update_state(
                task_id,
                status=result.status,
                current_step="completed" if result.status == "completed" else "failed",
                message="Agent task completed." if result.status == "completed" else "Agent task failed.",
                result_status=result.status,
            )
        except Exception as exc:  # defensive boundary for background jobs
            self._update_state(
                task_id,
                status="failed",
                current_step="failed",
                message=f"Agent task failed: {exc}",
                error={"code": "TASK_EXECUTION_FAILED", "message": str(exc)},
            )

    def _task_state_dir(self, task_id: str) -> Path:
        _validate_task_id(task_id)
        path = (self.state_dir / task_id).resolve()
        if path.parent != self.state_dir:
            raise ValueError("Invalid task ID")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _task_output_dir(self, task_id: str) -> Path | None:
        try:
            _validate_task_id(task_id)
        except ValueError:
            return None
        path = (self.output_dir / task_id).resolve()
        if path.parent != self.output_dir:
            return None
        return path

    def _monitor_path(self, task_id: str) -> Path:
        task_dir = self._task_output_dir(task_id)
        if task_dir is None:
            raise ValueError("Invalid task ID")
        return task_dir / "agent_monitor.jsonl"

    def _latest_monitor_event(self, task_id: str) -> dict[str, Any] | None:
        path = self._monitor_path(task_id)
        if not path.exists():
            return None
        for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None

    def _read_state(self, task_id: str) -> dict[str, Any] | None:
        try:
            return self._read_json(self._task_state_dir(task_id) / "task_state.json")
        except ValueError:
            return None

    def _write_state(self, task_id: str, payload: dict[str, Any]) -> None:
        state = dict(payload)
        state.setdefault("created_at", _now())
        state["updated_at"] = _now()
        self._write_json(self._task_state_dir(task_id) / "task_state.json", state)

    def _update_state(self, task_id: str, **changes: Any) -> None:
        current = self._read_state(task_id) or {"task_id": task_id, "created_at": _now()}
        current.update(changes)
        self._write_state(task_id, current)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)


def _validate_task_id(task_id: str) -> None:
    if not task_id or not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError("Invalid task ID")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
