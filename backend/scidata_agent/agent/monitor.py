from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from pydantic import BaseModel


LOGGER = logging.getLogger(__name__)
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class AgentMonitor:
    """Console and JSONL monitor for Agent workflow checkpoints."""

    def __init__(
        self,
        task_id: str,
        output_dir: Path,
        console: bool = True,
        enabled: bool = True,
        cancel_check: Callable[[], bool] | None = None,
    ):
        if not task_id or not TASK_ID_PATTERN.fullmatch(task_id):
            raise ValueError("Invalid task ID")
        self.task_id = task_id
        self.console = console
        self.enabled = enabled
        self._cancel_check = cancel_check
        output_root = Path(output_dir).expanduser().resolve()
        self.task_dir = (output_root / task_id).resolve()
        if self.task_dir.parent != output_root:
            raise ValueError("Invalid task output path")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.task_dir / "agent_monitor.jsonl"
        self._step_started_at: dict[str, float] = {}
        self._emit_lock = Lock()

    def cancel_requested(self) -> bool:
        return bool(self._cancel_check and self._cancel_check())

    def start(self, step: str, message: str, data: dict[str, Any] | None = None) -> None:
        self._step_started_at[step] = time.perf_counter()
        self.emit("step", step, "started", message, data=data)

    def end(self, step: str, message: str, data: dict[str, Any] | None = None) -> None:
        started_at = self._step_started_at.pop(step, None)
        elapsed_ms = None
        if started_at is not None:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        payload = dict(data or {})
        if elapsed_ms is not None:
            payload["elapsed_ms"] = elapsed_ms
        self.emit("step", step, "completed", message, data=payload)

    def error(self, step: str, message: str, data: dict[str, Any] | None = None) -> None:
        started_at = self._step_started_at.pop(step, None)
        elapsed_ms = None
        if started_at is not None:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        payload = dict(data or {})
        if elapsed_ms is not None:
            payload["elapsed_ms"] = elapsed_ms
        self.emit("step", step, "failed", message, data=payload)

    def task(self, status: str, message: str, data: dict[str, Any] | None = None) -> None:
        self.emit("task", "task", status, message, data=data)

    def emit(
        self,
        event_type: str,
        step: str,
        status: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": self.task_id,
            "event_type": event_type,
            "step": step,
            "status": status,
            "message": message,
            "data": _jsonable(data or {}),
        }
        with self._emit_lock:
            try:
                line = json.dumps(event, ensure_ascii=False)
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except (OSError, TypeError, ValueError):
                LOGGER.warning("Agent monitor file write failed for task %s", self.task_id, exc_info=True)
            if self.console:
                try:
                    print(_format_console_event(event), flush=True)
                except (OSError, TypeError, ValueError):
                    LOGGER.warning("Agent monitor console write failed for task %s", self.task_id, exc_info=True)


def _format_console_event(event: dict[str, Any]) -> str:
    data = event.get("data") or {}
    compact_data = _compact_for_console(data)
    suffix = ""
    if compact_data:
        suffix = " | " + json.dumps(compact_data, ensure_ascii=False)
    return (
        f"[SciDataAgent][{event['task_id']}][{event['step']}][{event['status']}] "
        f"{event['message']}{suffix}"
    )


def _compact_for_console(data: Any) -> Any:
    if isinstance(data, dict):
        keys = [
            "elapsed_ms",
            "domain",
            "files_count",
            "candidate_sources_count",
            "arxiv_papers_count",
            "downloaded_pdfs_count",
            "text_blocks_count",
            "tables_count",
            "candidate_records_count",
            "final_records_count",
            "source_summaries_count",
            "source_catalog_count",
            "source_artifacts_count",
            "source_catalog_statuses",
            "source_artifact_statuses",
            "issue_count",
            "warning_count",
            "error_count",
            "conflict_count",
            "progress_index",
            "progress_total",
            "source_file",
            "page",
            "chars",
            "records_so_far",
            "export_files",
            "sample_sources",
            "sample_records",
        ]
        compact = {key: data[key] for key in keys if key in data}
        if "sample_sources" in compact:
            compact["sample_sources"] = _compact_sample_sources(compact["sample_sources"])
        if "sample_records" in compact:
            compact["sample_records"] = _compact_sample_records(compact["sample_records"])
        return compact
    return data


def _compact_sample_sources(sources: Any) -> Any:
    if not isinstance(sources, list):
        return sources
    compact_sources = []
    for source in sources[:3]:
        if isinstance(source, dict):
            compact_sources.append(
                {
                    "title": source.get("title"),
                    "source_type": source.get("source_type"),
                    "url": source.get("url"),
                    "downloaded": bool(source.get("downloaded_path")),
                }
            )
    return compact_sources


def _compact_sample_records(records: Any) -> Any:
    if not isinstance(records, list):
        return records
    compact_records = []
    for record in records[:3]:
        if isinstance(record, dict):
            compact_records.append(
                {
                    "paper_title": record.get("paper_title"),
                    "material": record.get("material"),
                    "metric_name": record.get("metric_name"),
                    "metric_value": record.get("metric_value"),
                    "unit": record.get("unit"),
                    "source_file": record.get("source_file"),
                    "page": record.get("page"),
                }
            )
    return compact_records


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, list):
        limited = [_jsonable(item) for item in value[:10]]
        if len(value) > 10:
            limited.append({"_truncated_count": len(value) - 10})
        return limited
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _jsonable(value.__dict__)
    return str(value)
