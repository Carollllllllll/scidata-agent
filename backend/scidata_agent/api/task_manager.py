from __future__ import annotations

import copy
import json
import os
import re
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, RLock
from typing import Any
from urllib.parse import quote

from scidata_agent.agent.schemas import timestamp_task_id
from scidata_agent.agent.scidata_agent import SciDataAgent


TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class TaskQueueFullError(RuntimeError):
    pass


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


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
        "dynamic_records_clean": "dynamic_records_clean.json",
        "dynamic_records_raw": "dynamic_records_raw.json",
        "needs_review": "needs_review.json",
        "needs_review_csv": "needs_review.csv",
        "review_queue": "review_queue.json",
        "chart_extractions": "chart_extractions.json",
        "chart_validation": "chart_validation_report.json",
        "chart_corrections": "chart_corrections.json",
        "cross_modal_validation": "cross_modal_validation.json",
        "connector_status": "connector_status.json",
        "agent_trace": "agent_trace.json",
        "decision_history": "decision_history.json",
        "tool_history": "tool_history.json",
        "discovered_sources": "discovered_sources.json",
        "summary": "summary.json",
        "final_report": "final_report.md",
    }

    def __init__(
        self,
        output_dir: Path,
        state_dir: Path,
        max_workers: int = 2,
        upload_dir: Path | None = None,
        max_pending_tasks: int | None = None,
    ):
        self.output_dir = output_dir.resolve()
        self.state_dir = state_dir.resolve()
        self.upload_dir = upload_dir.resolve() if upload_dir is not None else None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if self.upload_dir is not None:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="scidata-agent")
        self._futures: dict[str, Future[Any]] = {}
        self._cancel_events: dict[str, Event] = {}
        self._lock = RLock()
        self.owner_pid = os.getpid()
        configured_pending = max_pending_tasks if max_pending_tasks is not None else _positive_env_int("SCIDATA_MAX_PENDING_TASKS", 8)
        self.max_pending_tasks = max(1, configured_pending)

    def submit(
        self,
        *,
        task_id: str | None = None,
        research_question: str,
        files: list[str],
        run_options: dict[str, Any],
        auto_fetch_arxiv: bool = True,
        file_metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        task_id = task_id or timestamp_task_id()
        with self._lock:
            self._prune_futures_locked()
            if len(self._futures) >= self.max_pending_tasks:
                raise TaskQueueFullError(
                    f"Task queue is full ({len(self._futures)}/{self.max_pending_tasks})."
                )
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
                    "uploads": copy.deepcopy(file_metadata or []),
                    "internal_files": list(files),
                    "owner_pid": self.owner_pid,
                    "run_options": copy.deepcopy(run_options),
                    "auto_fetch_arxiv": auto_fetch_arxiv,
                },
            )
            cancel_event = Event()
            self._cancel_events[task_id] = cancel_event
            future = self._executor.submit(
                self._run,
                task_id,
                research_question,
                files,
                run_options,
                auto_fetch_arxiv,
                cancel_event,
            )
            self._futures[task_id] = future
            future.add_done_callback(lambda _future, identifier=task_id: self._discard_future(identifier))
        return self.get_task(task_id)

    def can_accept(self) -> bool:
        with self._lock:
            self._prune_futures_locked()
            return len(self._futures) < self.max_pending_tasks

    def cancel_task(self, task_id: str) -> bool:
        cancelled_before_start = False
        with self._lock:
            future = self._futures.get(task_id)
            cancel_event = self._cancel_events.get(task_id)
            if future is None or cancel_event is None or future.done():
                return False
            cancelled_before_start = future.cancel()
            if cancelled_before_start:
                self._futures.pop(task_id, None)
                self._cancel_events.pop(task_id, None)
            else:
                cancel_event.set()
        if cancelled_before_start:
            self._mark_cancelled(task_id, "Task cancelled before execution.")
        else:
            self._update_state(
                task_id,
                current_step="cancellation_requested",
                message="Cancellation requested; the current pipeline step will finish safely.",
            )
        return True

    def retry_task(self, task_id: str, new_task_id: str | None = None) -> dict[str, Any]:
        state = self._read_state(task_id)
        if state is None:
            raise KeyError(task_id)
        if state.get("status") in {"queued", "running"}:
            raise RuntimeError("Task is still active")
        new_task_id = new_task_id or timestamp_task_id()
        files, uploads = self._copy_retry_uploads(
            task_id,
            new_task_id,
            list(state.get("internal_files") or []),
            list(state.get("uploads") or []),
        )
        return self.submit(
            task_id=new_task_id,
            research_question=str(state.get("research_question") or ""),
            files=files,
            run_options=dict(state.get("run_options") or {}),
            auto_fetch_arxiv=bool(state.get("auto_fetch_arxiv", True)),
            file_metadata=uploads,
        )

    def resume_task(self, task_id: str) -> dict[str, Any]:
        """Resume a failed/interrupted task from its last valid Agent checkpoint."""
        state = self._read_state(task_id)
        if state is None:
            raise KeyError(task_id)
        if state.get("status") in {"queued", "running"}:
            raise RuntimeError("Task is still active")
        # Do not expose a stale result while the same task is being resumed.
        try:
            (self._task_state_dir(task_id) / "result_payload.json").unlink(missing_ok=True)
        except OSError:
            pass
        run_options = dict(state.get("run_options") or {})
        run_options["resume"] = True
        return self.submit(
            task_id=task_id,
            research_question=str(state.get("research_question") or ""),
            files=list(state.get("internal_files") or []),
            run_options=run_options,
            auto_fetch_arxiv=bool(state.get("auto_fetch_arxiv", True)),
            file_metadata=list(state.get("uploads") or []),
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        state = self._read_state(task_id)
        if state is None:
            return {"task_id": task_id, "status": "not_found"}

        event = self._latest_monitor_event(task_id)
        response = copy.deepcopy(state)
        if event:
            if event.get("event_type") in {"step", "progress"}:
                response["current_step"] = event.get("step") or response.get("current_step")
            if not (response.get("status") == "failed" and response.get("error")):
                response["message"] = event.get("message") or response.get("message")
            response["updated_at"] = event.get("timestamp") or response.get("updated_at")
            compact_event = _compact_monitor_event(event, include_data=False)
            response["event"] = compact_event
            event_data = event.get("data") or {}
            _apply_live_snapshot(response, compact_event.get("data"))
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
                    "review_decisions": self.review_decisions(task_id),
                }
            )
            # A final payload is authoritative. The last monitor event can be
            # an intermediate tool event written before the Agent updates its
            # terminal status and stop reason.
            response.update(_snapshot_from_result_payload(payload))
        else:
            response["review_decisions"] = self.review_decisions(task_id)
        return response

    def task_exists(self, task_id: str) -> bool:
        return self._read_state(task_id) is not None

    def list_tasks(self, limit: int = 20, status: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        allowed_statuses = {"queued", "running", "completed", "partial", "failed", "cancelled"}
        if status is not None and status not in allowed_statuses:
            raise ValueError("Invalid task status")

        candidates: list[tuple[float, str, Path]] = []
        try:
            entries = os.scandir(self.state_dir)
        except OSError:
            return []
        with entries:
            for entry in entries:
                try:
                    is_directory = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if not is_directory or not TASK_ID_PATTERN.fullmatch(entry.name):
                    continue
                task_dir = Path(entry.path)
                candidates.append((self._task_recency(task_dir), entry.name, task_dir))
        candidates.sort(reverse=True)

        tasks: list[dict[str, Any]] = []
        for _recency, task_id, task_dir in candidates:
            state = self._read_json(task_dir / "task_state.json")
            if state is None or (status is not None and state.get("status") != status):
                continue
            # Payloads, export checks and monitor tails are comparatively
            # expensive. Only materialize them for rows that can enter the
            # requested page rather than for the complete task history.
            payload = self._read_json(task_dir / "result_payload.json")
            item = copy.deepcopy(state)
            if item.get("status") in {"queued", "running"}:
                event = self._latest_monitor_event(task_id)
                if event:
                    if event.get("event_type") in {"step", "progress"}:
                        item["current_step"] = event.get("step") or item.get("current_step")
                    item["message"] = event.get("message") or item.get("message")
                    item["updated_at"] = event.get("timestamp") or item.get("updated_at")
                    _apply_live_snapshot(item, _compact_monitor_event(event, include_data=False).get("data"))
                    event_data = event.get("data") or {}
                    if "progress_index" in event_data or "progress_total" in event_data:
                        item["progress"] = {
                            "current": event_data.get("progress_index"),
                            "total": event_data.get("progress_total"),
                        }
            item["result"] = None
            item["summary"] = payload.get("summary") if payload else None
            item["quality_report"] = payload.get("quality_report") if payload else None
            if payload:
                item.update(_snapshot_from_result_payload(payload))
            item["download_urls"] = self.download_urls(task_id)
            tasks.append(item)
            if len(tasks) >= limit:
                break

        tasks.sort(key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""), reverse=True)
        return tasks

    def _task_recency(self, task_dir: Path) -> float:
        mtimes: list[float] = []
        for path in (
            task_dir / "task_state.json",
            self.output_dir / task_dir.name / "agent_monitor.jsonl",
        ):
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
        return max(mtimes, default=0.0)

    def get_events(
        self,
        task_id: str,
        tail: int = 100,
        *,
        include_data: bool = False,
    ) -> dict[str, Any]:
        state = self._read_state(task_id)
        if state is None:
            return {"task_id": task_id, "status": "not_found", "events": []}
        tail = max(1, min(int(tail), 500))
        log_path = self._monitor_path(task_id)
        events: list[dict[str, Any]] = []
        live_snapshot: dict[str, Any] = {}
        if log_path.exists():
            for line in _tail_text_lines(log_path, tail):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    compact_event = _compact_monitor_event(event, include_data=include_data)
                    events.append(compact_event)
                    _merge_live_snapshot(
                        live_snapshot,
                        event.get("data") if include_data else compact_event.get("data"),
                    )
        response: dict[str, Any] = {"task_id": task_id, "status": state.get("status"), "events": events}
        if live_snapshot:
            response.update(live_snapshot)
        payload = self._read_json(self._task_state_dir(task_id, create=False) / "result_payload.json")
        if payload:
            # Event tails are useful while a task runs, but the persisted
            # result is authoritative once the task reaches a terminal state.
            response.update(_snapshot_from_result_payload(payload))
        return response

    def review_decisions(self, task_id: str) -> dict[str, dict[str, Any]]:
        payload = self._read_json(self._task_state_dir(task_id, create=False) / "review_decisions.json")
        if not payload:
            return {}
        return {
            str(record_id): value
            for record_id, value in payload.items()
            if isinstance(value, dict)
        }

    def set_review_decision(
        self,
        task_id: str,
        record_id: str,
        decision: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        if decision not in {"approved", "needs_changes", "rejected"}:
            raise ValueError("Invalid review decision")
        with self._lock:
            payload = self._read_json(self._task_state_dir(task_id, create=False) / "result_payload.json")
            if payload is None:
                raise LookupError("Task result is not ready")
            valid_ids = _reviewable_review_ids(payload)
            if record_id not in valid_ids:
                raise KeyError(record_id)
            decisions = self.review_decisions(task_id)
            queue_item = valid_ids[record_id]
            review = {
                "record_id": str(queue_item.get("record_id") or record_id),
                "review_id": record_id,
                "subject_id": str(queue_item.get("subject_id") or record_id),
                "subject_type": queue_item.get("subject_type"),
                "decision": decision,
                "note": note.strip() if isinstance(note, str) and note.strip() else None,
                "updated_at": _now(),
            }
            decisions[record_id] = review
            legacy_record_id = queue_item.get("record_id")
            if legacy_record_id and str(legacy_record_id) != record_id:
                decisions[str(legacy_record_id)] = review
            self._write_json(self._task_state_dir(task_id) / "review_decisions.json", decisions)
            self._update_state(task_id, updated_at=_now())
            return review

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

    def asset_url(self, task_id: str, file_path: str | Path) -> str | None:
        """Convert an internal task file path into a scoped public URL."""

        try:
            resolved = Path(file_path).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        for scope, root in self._asset_roots(task_id).items():
            try:
                relative = resolved.relative_to(root)
            except ValueError:
                continue
            encoded = quote(relative.as_posix(), safe="/")
            return f"/api/tasks/{task_id}/assets/{scope}/{encoded}"
        return None

    def asset_path(self, task_id: str, scope: str, relative_path: str) -> Path | None:
        roots = self._asset_roots(task_id)
        root = roots.get(scope)
        if root is None or not relative_path or Path(relative_path).is_absolute():
            return None
        try:
            path = (root / relative_path).resolve()
            path.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            return None
        return path if path.is_file() else None

    def shutdown(self, wait: bool = True) -> None:
        """Stop the worker pool during application shutdown or test teardown."""
        with self._lock:
            active_task_ids = list(self._futures)
        for task_id in active_task_ids:
            self.cancel_task(task_id)
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _discard_future(self, task_id: str) -> None:
        with self._lock:
            self._futures.pop(task_id, None)
            self._cancel_events.pop(task_id, None)

    def _prune_futures_locked(self) -> None:
        completed = [task_id for task_id, future in self._futures.items() if future.done()]
        for task_id in completed:
            self._futures.pop(task_id, None)

    def reconcile_interrupted_tasks(self) -> None:
        """Fail orphaned active tasks while preserving work owned by a live process."""
        if not self.state_dir.is_dir():
            return
        for task_dir in self.state_dir.iterdir():
            if not task_dir.is_dir() or not TASK_ID_PATTERN.fullmatch(task_dir.name):
                continue
            state = self._read_json(task_dir / "task_state.json")
            if not state or state.get("status") not in {"queued", "running"}:
                continue
            if _pid_is_alive(state.get("owner_pid")):
                continue
            state.update(
                {
                    "status": "failed",
                    "current_step": "interrupted",
                    "message": "Task was interrupted by an API process restart and can be retried.",
                    "error": {
                        "code": "TASK_INTERRUPTED",
                        "message": "Task was interrupted by an API process restart and can be retried.",
                    },
                    "updated_at": _now(),
                }
            )
            self._write_json(task_dir / "task_state.json", state)

    def _copy_retry_uploads(
        self,
        old_task_id: str,
        new_task_id: str,
        files: list[str],
        uploads: list[dict[str, Any]],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if not files or self.upload_dir is None:
            return files, copy.deepcopy(uploads)
        old_root = (self.upload_dir / old_task_id).resolve()
        new_root = (self.upload_dir / new_task_id).resolve()
        if old_root.parent != self.upload_dir or new_root.parent != self.upload_dir:
            raise ValueError("Invalid retry upload path")
        new_root.mkdir(parents=True, exist_ok=True)
        copied_files: list[str] = []
        copied_uploads = copy.deepcopy(uploads)
        for index, value in enumerate(files):
            source = Path(value).resolve()
            try:
                source.relative_to(old_root)
            except ValueError:
                copied_files.append(str(source))
                continue
            target = (new_root / source.name).resolve()
            target.relative_to(new_root)
            shutil.copy2(source, target)
            copied_files.append(str(target))
            if index < len(copied_uploads):
                copied_uploads[index]["local_path"] = str(target)
        return copied_files, copied_uploads

    def _run(
        self,
        task_id: str,
        research_question: str,
        files: list[str],
        run_options: dict[str, Any],
        auto_fetch_arxiv: bool,
        cancel_event: Event,
    ) -> None:
        if cancel_event.is_set():
            self._mark_cancelled(task_id, "Task cancelled before execution.")
            return
        self._update_state(task_id, status="running", current_step="starting", message="Agent task started.")
        agent = None
        try:
            agent = SciDataAgent(output_dir=self.output_dir)
            result = agent.run(
                research_question,
                files,
                task_id=task_id,
                auto_fetch_arxiv=auto_fetch_arxiv,
                cancel_check=cancel_event.is_set,
                **run_options,
            )
            payload = result.model_dump(mode="json", by_alias=True)
            self._write_json(self._task_state_dir(task_id) / "result_payload.json", payload)
            if cancel_event.is_set():
                self._mark_cancelled(task_id, "Task cancelled at a safe pipeline checkpoint.")
                return
            failure_event = self._latest_failure_event(task_id) if result.status == "failed" else None
            failure_message = _result_failure_message(payload)
            if failure_event:
                failure_message = failure_event.get("message") or failure_message
            is_partial = result.status == "partial"
            partial_message = (
                "Agent task partially completed; coverage requirements remain unsatisfied."
            )
            self._update_state(
                task_id,
                status=result.status,
                current_step=(
                    "completed"
                    if result.status == "completed"
                    else "partial"
                    if is_partial
                    else (failure_event or {}).get("step") or "failed"
                ),
                message=(
                    "Agent task completed."
                    if result.status == "completed"
                    else partial_message
                    if is_partial
                    else failure_message
                ),
                error=(
                    None if result.status == "completed"
                    else {"code": "AGENT_TASK_PARTIAL", "message": partial_message}
                    if is_partial
                    else {"code": "AGENT_TASK_FAILED", "message": failure_message}
                ),
                result_status=result.status,
            )
        except Exception as exc:  # defensive boundary for background jobs
            if cancel_event.is_set():
                self._mark_cancelled(task_id, "Task cancelled at a safe pipeline checkpoint.")
            else:
                self._update_state(
                    task_id,
                    status="failed",
                    current_step="failed",
                    message=f"Agent task failed: {exc}",
                    error={"code": "TASK_EXECUTION_FAILED", "message": str(exc)},
                )
        finally:
            llm_client = getattr(agent, "llm_client", None)
            close = getattr(llm_client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def _mark_cancelled(self, task_id: str, message: str) -> None:
        self._update_state(
            task_id,
            status="cancelled",
            current_step="cancelled",
            message=message,
            error={"code": "TASK_CANCELLED", "message": message},
        )

    def _task_state_dir(self, task_id: str, *, create: bool = True) -> Path:
        _validate_task_id(task_id)
        path = (self.state_dir / task_id).resolve()
        if path.parent != self.state_dir:
            raise ValueError("Invalid task ID")
        if create:
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

    def _asset_roots(self, task_id: str) -> dict[str, Path]:
        try:
            _validate_task_id(task_id)
        except ValueError:
            return {}
        roots = {"output": (self.output_dir / task_id).resolve()}
        if self.upload_dir is not None:
            roots["upload"] = (self.upload_dir / task_id).resolve()
        return roots

    def _latest_monitor_event(self, task_id: str) -> dict[str, Any] | None:
        path = self._monitor_path(task_id)
        if not path.exists():
            return None
        for line in reversed(_tail_text_lines(path, 20)):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return None

    def _latest_failure_event(self, task_id: str) -> dict[str, Any] | None:
        path = self._monitor_path(task_id)
        if not path.exists():
            return None
        for line in reversed(_tail_text_lines(path, 200)):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("status") in {"failed", "error"} and event.get("step") != "task":
                return event
        return None

    def _read_state(self, task_id: str) -> dict[str, Any] | None:
        try:
            state_dir = self._task_state_dir(task_id, create=False)
        except ValueError:
            return None
        if not state_dir.is_dir():
            return None
        return self._read_json(state_dir / "task_state.json")

    def _write_state(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            state = dict(payload)
            state.setdefault("created_at", _now())
            state["updated_at"] = _now()
            self._write_json(self._task_state_dir(task_id) / "task_state.json", state)

    def _update_state(self, task_id: str, **changes: Any) -> None:
        with self._lock:
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


def _tail_text_lines(path: Path, limit: int, max_bytes: int = 4 * 1024 * 1024) -> list[str]:
    """Read the last JSONL lines without loading an ever-growing monitor file."""
    if limit <= 0 or not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            buffer = b""
            while position > 0 and buffer.count(b"\n") <= limit and len(buffer) < max_bytes:
                size = min(8192, position, max_bytes - len(buffer))
                if size <= 0:
                    break
                position -= size
                handle.seek(position)
                buffer = handle.read(size) + buffer
    except OSError:
        return []
    return [line.decode("utf-8", errors="replace") for line in buffer.splitlines()[-limit:]]


def _compact_monitor_event(event: dict[str, Any], *, include_data: bool) -> dict[str, Any]:
    if include_data:
        return event
    public_keys = (
        "timestamp",
        "event_type",
        "step",
        "status",
        "message",
        "duration_ms",
    )
    compact = {key: event[key] for key in public_keys if key in event}
    live_snapshot = _live_snapshot_from_event_data(event.get("data"))
    if live_snapshot:
        compact["data"] = live_snapshot
    return compact


def _apply_live_snapshot(target: dict[str, Any], data: Any) -> None:
    snapshot = _normalise_live_snapshot(data)
    if snapshot:
        target.update(snapshot)


def _merge_live_snapshot(target: dict[str, Any], data: Any) -> None:
    snapshot = _normalise_live_snapshot(data)
    if snapshot:
        target.update(snapshot)


def _normalise_live_snapshot(data: Any) -> dict[str, Any]:
    if isinstance(data, dict) and any(
        key in data for key in ("runtime", "coverage", "source_status")
    ):
        return {
            key: data[key]
            for key in ("runtime", "coverage", "source_status")
            if key in data
        }
    return _live_snapshot_from_event_data(data)


def _live_snapshot_from_event_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    runtime = _compact_runtime_snapshot(data.get("runtime"))
    runtime_event = _compact_runtime_event(data.get("runtime_event"))
    if runtime_event:
        runtime["latest_event"] = runtime_event
    coverage = _compact_coverage_snapshot(data.get("coverage_report"))
    source_status = _compact_source_status(data)
    snapshot: dict[str, Any] = {}
    if runtime:
        snapshot["runtime"] = runtime
    if coverage:
        snapshot["coverage"] = coverage
    if source_status:
        snapshot["source_status"] = source_status
    return snapshot


def _compact_runtime_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    event = {
        key: value[key]
        for key in ("event_type", "iteration", "call_id", "tool_name", "status")
        if key in value
    }
    payload = value.get("payload")
    if isinstance(payload, dict):
        result = payload.get("result")
        if isinstance(result, dict):
            for key in ("retry_count", "cached"):
                if key in result:
                    event[key] = result[key]
            for source_key, target_key in (
                ("evidence_refs", "evidence_count"),
                ("artifact_refs", "artifact_count"),
                ("warnings", "warning_count"),
                ("errors", "error_count"),
            ):
                values = result.get(source_key)
                if isinstance(values, list):
                    event[target_key] = len(values)
        reasons = payload.get("reasons")
        if isinstance(reasons, list):
            event["reason_count"] = len(reasons)
    return event


def _compact_runtime_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    decisions = []
    raw_decisions = value.get("recent_decisions", [])
    if not isinstance(raw_decisions, list):
        raw_decisions = []
    for raw in raw_decisions[-3:]:
        if not isinstance(raw, dict):
            continue
        calls = []
        raw_calls = raw.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raw_calls = []
        for call in raw_calls[:12]:
            if not isinstance(call, dict):
                continue
            item = {
                key: call[key]
                for key in ("call_id", "tool_name", "priority", "gap_ids")
                if key in call
            }
            for key in ("reason", "purpose"):
                if key in call:
                    item[key] = _public_text(call[key])
            for key in ("expected_evidence",):
                if isinstance(call.get(key), list):
                    item[key] = [_public_text(entry) for entry in call[key][:12]]
            calls.append(item)
        decision = {
            key: raw[key]
            for key in ("decision",)
            if key in raw
        }
        for key in ("reason", "stop_reason"):
            if key in raw:
                decision[key] = _public_text(raw[key])
        if isinstance(raw.get("expected_evidence"), list):
            decision["expected_evidence"] = [
                _public_text(entry) for entry in raw["expected_evidence"][:12]
            ]
        decisions.append(decision | {"tool_calls": calls})

    results = []
    raw_results = value.get("recent_tool_results", [])
    if not isinstance(raw_results, list):
        raw_results = []
    for raw in raw_results[-5:]:
        if not isinstance(raw, dict):
            continue
        item = {
            key: raw[key]
            for key in (
                "call_id",
                "tool_name",
                "status",
                "elapsed_ms",
                "retry_count",
                "cached",
            )
            if key in raw
        }
        for key in ("artifact_refs", "evidence_refs", "warnings", "errors"):
            if isinstance(raw.get(key), list):
                item[key] = [_public_text(entry) for entry in raw[key][:12]]
        results.append(item)

    snapshot = {
        key: value[key]
        for key in (
            "iteration",
            "iteration_budget",
            "status",
            "phase",
            "stop_reason",
            "no_progress_streak",
            "no_progress_limit",
            "last_progress_iteration",
            "decision_count",
            "tool_result_count",
            "trace_count",
        )
        if key in value
    }
    for key in ("stop_reason",):
        if key in snapshot:
            snapshot[key] = _public_text(snapshot[key])
    snapshot["recent_decisions"] = decisions
    snapshot["recent_tool_results"] = results
    raw_rejections = value.get("stop_rejections", [])
    if not isinstance(raw_rejections, list):
        raw_rejections = []
    snapshot["stop_rejections"] = [_public_text(item) for item in raw_rejections[-8:]]
    return snapshot


def _snapshot_from_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Project a completed Agent result into the live API snapshot shape."""
    decisions = payload.get("agent_decision_history")
    if not isinstance(decisions, list):
        decisions = []
    tool_results = payload.get("tool_result_history")
    if not isinstance(tool_results, list):
        tool_results = []
    trace = payload.get("agent_trace")
    if not isinstance(trace, list):
        trace = []
    rejections = payload.get("stop_rejections")
    if not isinstance(rejections, list):
        rejections = []
    runtime_data = {
        "iteration": payload.get("runtime_iteration", 0),
        "iteration_budget": payload.get("runtime_iteration_budget"),
        "status": payload.get("runtime_status") or payload.get("status"),
        "phase": payload.get("runtime_phase"),
        "stop_reason": payload.get("runtime_stop_reason"),
        "no_progress_streak": payload.get("runtime_no_progress_streak", 0),
        "no_progress_limit": payload.get("runtime_no_progress_limit", 4),
        "last_progress_iteration": payload.get("runtime_last_progress_iteration"),
        "decision_count": len(decisions),
        "tool_result_count": len(tool_results),
        "trace_count": len(trace),
        "recent_decisions": decisions[-3:],
        "recent_tool_results": tool_results[-5:],
        "stop_rejections": rejections[-8:],
    }

    catalog = payload.get("source_catalog")
    if not isinstance(catalog, list):
        catalog = []
    artifacts = [
        artifact
        for source in catalog
        if isinstance(source, dict)
        for artifact in (source.get("artifacts") or [])
        if isinstance(artifact, dict)
    ]

    def count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            value = str(item.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts

    source_data = {
        "source_catalog_count": len(catalog),
        "source_artifacts_count": len(artifacts),
        "source_catalog_statuses": count_by(
            [item for item in catalog if isinstance(item, dict)], "status"
        ),
        "source_artifact_statuses": count_by(artifacts, "status"),
        "connector_status": payload.get("connector_status", []),
    }
    return {
        "runtime": _compact_runtime_snapshot(runtime_data),
        "coverage": _compact_coverage_snapshot(payload.get("coverage_report")),
        "source_status": _compact_source_status(source_data),
    }


def _compact_coverage_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    gaps = value.get("gaps")
    missing = value.get("missing_requirements")
    unprocessed = value.get("unprocessed_relevant_artifacts")
    snapshot = {
        key: value[key]
        for key in (
            "decision",
            "coverage_score",
            "required_evidence_types",
            "covered_evidence_types",
            "reasons",
            "recommended_actions",
        )
        if key in value
    }
    for key in ("required_evidence_types", "covered_evidence_types", "reasons", "recommended_actions"):
        if isinstance(value.get(key), list):
            snapshot[key] = [_public_text(item) for item in value[key][:12]]
    return snapshot | {
        "gap_count": len(gaps) if isinstance(gaps, list) else 0,
        "missing_requirements": [_public_text(item) for item in missing[:12]] if isinstance(missing, list) else [],
        "unprocessed_relevant_artifacts_count": len(unprocessed) if isinstance(unprocessed, list) else 0,
    }


def _compact_source_status(data: dict[str, Any]) -> dict[str, Any]:
    catalog_count = data.get("source_catalog_count")
    artifact_count = data.get("source_artifacts_count")
    source_counts = data.get("source_catalog_statuses")
    artifact_counts = data.get("source_artifact_statuses")
    connector_items = []
    connector_status = data.get("connector_status", [])
    if not isinstance(connector_status, list):
        connector_status = []
    for raw in connector_status[:12]:
        if not isinstance(raw, dict):
            continue
        connector_items.append({
            key: _public_text(raw[key]) if key in {"query", "error", "message"} else raw[key]
            for key in (
                "connector",
                "connector_name",
                "query",
                "status",
                "attempt",
                "attempts",
                "retry_count",
                "added_sources_count",
                "error",
                "message",
            )
            if key in raw
        })
    if not any(value is not None for value in (catalog_count, artifact_count, source_counts, artifact_counts)) and not connector_items:
        return {}
    return {
        "catalog_count": catalog_count,
        "artifact_count": artifact_count,
        "source_status_counts": source_counts if isinstance(source_counts, dict) else {},
        "artifact_status_counts": artifact_counts if isinstance(artifact_counts, dict) else {},
        "connectors": connector_items,
    }


def _public_text(value: Any, max_length: int = 1000) -> str:
    """Bound diagnostic text and redact Windows paths from live snapshots."""
    text = str(value)
    text = re.sub(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\s,;]+", "[local path]", text)
    return text if len(text) <= max_length else text[:max_length] + "...[truncated]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_is_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _result_failure_message(payload: dict[str, Any]) -> str:
    logs = payload.get("processing_log")
    if isinstance(logs, list):
        for item in reversed(logs):
            if isinstance(item, str) and item.lower().startswith("task failed:"):
                return item.split(":", 1)[1].strip() or "Agent task failed."
    return "Agent task failed."


def _reviewable_review_ids(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    queue = payload.get("review_queue")
    if isinstance(queue, list):
        for item in queue:
            if isinstance(item, dict) and item.get("review_id"):
                result[str(item["review_id"])] = item
    for collection in ("records", "dynamic_records", "needs_review_records", "figures"):
        items = payload.get(collection)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            identifier = item.get("record_id") or item.get("figure_id")
            if identifier:
                identifier = str(identifier)
                result.setdefault(identifier, {
                    "review_id": identifier,
                    "subject_id": identifier,
                    "subject_type": "record" if item.get("record_id") else "figure",
                    "record_id": item.get("record_id"),
                })
    return result
