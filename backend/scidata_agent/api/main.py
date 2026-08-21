from __future__ import annotations

from contextlib import asynccontextmanager
import os
import re
import hmac
import time
from collections import defaultdict, deque
from pathlib import Path
from pathlib import PureWindowsPath
from threading import Lock
from typing import Any
from uuid import uuid4

try:
    from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
except ImportError:  # pragma: no cover - allows core agent tests without FastAPI installed.
    FastAPI = None  # type: ignore[assignment]
    File = Form = UploadFile = None  # type: ignore[assignment]
    HTTPException = Query = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
    FileResponse = JSONResponse = None  # type: ignore[assignment]

from scidata_agent.agent.schemas import timestamp_task_id
from scidata_agent.api.schemas import (
    HealthResponse,
    ReviewDecision,
    ReviewRequest,
    TaskEventsResponse,
    TaskListResponse,
    TaskResponse,
    TaskStatus,
    TaskSubmissionResponse,
)
from scidata_agent.api.task_manager import TaskManager, TaskQueueFullError
from scidata_agent.config import get_settings, load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[3]
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
TASK_STATE_DIR = RUNTIME_DIR / "tasks"
TASK_MANAGER = TaskManager(output_dir=OUTPUT_DIR, state_dir=TASK_STATE_DIR, upload_dir=UPLOAD_DIR)
API_VERSION = "0.3.0"

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".csv", ".tsv", ".xlsx", ".xls"}
PATH_URL_KEYS = {
    "path": "asset_url",
    "file_path": "asset_url",
    "downloaded_path": "asset_url",
    "local_path": "asset_url",
    "source_path": "source_url",
    "image_path": "image_url",
    "monitor_log_path": "monitor_url",
}


def _positive_env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


MAX_UPLOAD_FILES = _positive_env_int("SCIDATA_MAX_UPLOAD_FILES", 20)
MAX_UPLOAD_BYTES = _positive_env_int("SCIDATA_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
MAX_UPLOAD_TOTAL_BYTES = _positive_env_int("SCIDATA_MAX_UPLOAD_TOTAL_BYTES", 200 * 1024 * 1024)
RATE_LIMIT_PER_MINUTE = _positive_env_int("SCIDATA_RATE_LIMIT_PER_MINUTE", 60)
_RATE_LIMIT_LOCK = Lock()
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def _cors_origins() -> list[str]:
    configured = os.getenv("SCIDATA_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _save_uploads(
    files: list[UploadFile] | None,
    task_id: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    uploads = files or []
    if len(uploads) > MAX_UPLOAD_FILES:
        raise HTTPException(
            status_code=413,
            detail={"code": "TOO_MANY_FILES", "message": f"最多上传 {MAX_UPLOAD_FILES} 个文件。"},
        )
    task_upload_dir = (UPLOAD_DIR / task_id).resolve()
    if task_upload_dir.parent != UPLOAD_DIR.resolve():
        raise ValueError("Invalid upload task ID")
    task_upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    file_metadata: list[dict[str, Any]] = []
    saved_targets: list[Path] = []
    total_size = 0
    for uploaded in uploads:
        original_name = Path(uploaded.filename or "uploaded_file").name
        safe_name = original_name or "uploaded_file"
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            _cleanup_saved_uploads(saved_targets, task_upload_dir)
            raise HTTPException(
                status_code=415,
                detail={
                    "code": "UNSUPPORTED_FILE_TYPE",
                    "message": f"不支持 {suffix or '无扩展名'} 文件；仅支持 PDF、CSV、TSV、XLSX、XLS。",
                },
            )
        target = task_upload_dir / f"{uuid4().hex[:12]}_{safe_name}"
        size = 0
        try:
            with target.open("wb") as handle:
                while chunk := uploaded.file.read(1024 * 1024):
                    size += len(chunk)
                    total_size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "code": "FILE_TOO_LARGE",
                                "message": f"文件 {safe_name} 超过单文件大小限制。",
                            },
                        )
                    if total_size > MAX_UPLOAD_TOTAL_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "code": "UPLOAD_TOTAL_TOO_LARGE",
                                "message": "本次任务上传文件总大小超过限制。",
                            },
                        )
                    handle.write(chunk)
            _validate_uploaded_file(target, suffix)
        except Exception:
            target.unlink(missing_ok=True)
            _cleanup_saved_uploads(saved_targets, task_upload_dir)
            raise
        saved_targets.append(target)
        saved_files.append(str(target))
        file_metadata.append(
            {
                "name": safe_name,
                "size": size,
                "content_type": uploaded.content_type,
                "local_path": str(target),
            }
        )
    return saved_files, file_metadata


def _validate_uploaded_file(path: Path, suffix: str) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(4096)
    if not prefix:
        raise HTTPException(
            status_code=422,
            detail={"code": "EMPTY_FILE", "message": f"文件 {path.name} 为空。"},
        )
    if suffix == ".pdf" and not prefix.lstrip().startswith(b"%PDF-"):
        raise HTTPException(
            status_code=415,
            detail={"code": "INVALID_FILE_CONTENT", "message": "PDF 扩展名与文件内容不匹配。"},
        )
    if suffix == ".xlsx" and not prefix.startswith(b"PK"):
        raise HTTPException(
            status_code=415,
            detail={"code": "INVALID_FILE_CONTENT", "message": "XLSX 扩展名与文件内容不匹配。"},
        )
    if suffix == ".xls" and not prefix.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raise HTTPException(
            status_code=415,
            detail={"code": "INVALID_FILE_CONTENT", "message": "XLS 扩展名与文件内容不匹配。"},
        )
    if suffix in {".csv", ".tsv"} and b"\x00" in prefix:
        raise HTTPException(
            status_code=415,
            detail={"code": "INVALID_FILE_CONTENT", "message": "表格文本文件包含二进制内容。"},
        )


def _cleanup_saved_uploads(paths: list[Path], task_upload_dir: Path) -> None:
    for path in paths:
        path.unlink(missing_ok=True)
    try:
        task_upload_dir.rmdir()
    except OSError:
        pass


def _sanitize_public_value(
    value: Any,
    task_id: str,
    *,
    manager: TaskManager,
    key: str | None = None,
) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for child_key, child_value in value.items():
            if child_key == "export_files":
                continue
            public_key = PATH_URL_KEYS.get(child_key)
            if public_key is not None:
                asset_url = manager.asset_url(task_id, child_value) if child_value else None
                if asset_url:
                    sanitized[public_key] = asset_url
                continue
            sanitized[child_key] = _sanitize_public_value(
                child_value,
                task_id,
                manager=manager,
                key=child_key,
            )
        return sanitized
    if isinstance(value, list):
        return [_sanitize_public_value(item, task_id, manager=manager, key=key) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_public_value(item, task_id, manager=manager, key=key) for item in value]
    if isinstance(value, str) and key == "source_file":
        candidate = Path(value)
        if candidate.is_absolute() or "\\" in value:
            name = PureWindowsPath(value).name if "\\" in value else candidate.name
            return re.sub(r"^[0-9a-f]{12}_", "", name)
    return value


def _public_task_response(
    task: dict[str, Any],
    *,
    manager: TaskManager | None = None,
) -> dict[str, Any]:
    """Return one stable task envelope without exposing server filesystem paths."""

    manager = manager or TASK_MANAGER
    task_id = str(task["task_id"])
    raw_result = task.get("result")
    result = (
        _sanitize_public_value(raw_result, task_id, manager=manager)
        if isinstance(raw_result, dict)
        else None
    )
    download_urls = task.get("download_urls") or manager.download_urls(task_id)
    if result is not None:
        result["export_files"] = download_urls
        result["download_urls"] = download_urls

    summary = result.get("summary") if result else task.get("summary")
    quality_report = result.get("quality_report") if result else task.get("quality_report")
    return {
        "task_id": task_id,
        "status": task.get("status", "failed"),
        "research_question": task.get("research_question")
        or (result.get("research_question") if result else None),
        "current_step": task.get("current_step"),
        "message": task.get("message"),
        "progress": task.get("progress"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
        "error": task.get("error"),
        "uploads": _sanitize_public_value(task.get("uploads", []), task_id, manager=manager),
        "event": _sanitize_public_value(task.get("event"), task_id, manager=manager),
        "result": result,
        "summary": summary,
        "quality_report": quality_report,
        "download_urls": download_urls,
        "review_decisions": task.get("review_decisions") or manager.review_decisions(task_id),
    }


def _task_or_404(task_id: str) -> dict[str, Any]:
    task = TASK_MANAGER.get_task(task_id)
    if task.get("status") == "not_found":
        raise HTTPException(
            status_code=404,
            detail={"code": "TASK_NOT_FOUND", "message": "任务不存在。"},
        )
    return task


@asynccontextmanager
async def _app_lifespan(_app):
    TASK_MANAGER.reconcile_interrupted_tasks()
    try:
        yield
    finally:
        TASK_MANAGER.shutdown()


if FastAPI is not None:
    app = FastAPI(title="SciData Agent API", version=API_VERSION, lifespan=_app_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def api_security_guard(request: Request, call_next):
        if request.url.path.startswith("/api") and request.url.path != "/api/health":
            configured_token = os.getenv("SCIDATA_API_TOKEN", "").strip()
            if configured_token:
                supplied = request.headers.get("Authorization", "")
                expected = f"Bearer {configured_token}"
                if not hmac.compare_digest(supplied, expected):
                    return JSONResponse(
                        status_code=401,
                        content={"detail": {"code": "UNAUTHORIZED", "message": "缺少或无效的 API 访问令牌。"}},
                    )
            if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
                client_key = request.client.host if request.client else "unknown"
                now = time.monotonic()
                with _RATE_LIMIT_LOCK:
                    bucket = _RATE_LIMIT_BUCKETS[client_key]
                    while bucket and now - bucket[0] >= 60:
                        bucket.popleft()
                    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
                        return JSONResponse(
                            status_code=429,
                            content={"detail": {"code": "RATE_LIMITED", "message": "请求过于频繁，请稍后再试。"}},
                        )
                    bucket.append(now)
        return await call_next(request)

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "service": "SciData Agent API",
            "version": API_VERSION,
            "qwen_configured": bool(settings.dashscope_api_key),
            "model": settings.qwen_model,
            "vl_model": settings.qwen_vl_model,
            "text_model_pool": settings.qwen_models,
            "vl_model_pool": settings.qwen_vl_models,
            "cors_origins": _cors_origins(),
            "agent_loop": [
                "task_planning",
                "source_discovery",
                "source_parsing",
                "figure_chart_extraction",
                "record_extraction",
                "schema_alignment",
                "provenance_tracking",
                "quality_validation",
                "export",
            ],
        }

    @app.post("/api/analyze", status_code=202, response_model=TaskSubmissionResponse)
    async def analyze(
        research_question: str = Form(..., min_length=3, max_length=4000),
        files: list[UploadFile] | None = File(default=None),
        max_pdf_pages: int = Form(8, ge=1, le=200),
        max_arxiv_papers: int | None = Form(None, ge=0, le=100),
        max_auto_resources: int = Form(5, ge=0, le=100),
        enable_live_search: bool = Form(True),
        auto_download_sources: bool = Form(True),
        max_dynamic_text_blocks: int = Form(20, ge=1, le=500),
        max_record_text_blocks: int = Form(20, ge=1, le=500),
        max_figures_per_pdf: int = Form(6, ge=0, le=50),
        max_pdf_parse_workers: int | None = Form(None, ge=1, le=16),
        max_chart_workers: int | None = Form(None, ge=1, le=16),
        max_text_extraction_workers: int | None = Form(None, ge=1, le=16),
        max_table_extraction_workers: int | None = Form(None, ge=1, le=16),
        reuse_dynamic_records_for_metrics: bool = Form(True),
    ) -> dict[str, Any]:
        if not TASK_MANAGER.can_accept():
            raise HTTPException(
                status_code=503,
                detail={"code": "TASK_QUEUE_FULL", "message": "任务队列已满，请稍后再试。"},
            )
        task_id = timestamp_task_id()
        saved_files, file_metadata = _save_uploads(files, task_id)
        try:
            task = TASK_MANAGER.submit(
                task_id=task_id,
                research_question=research_question,
                files=saved_files,
                run_options={
                    "max_pdf_pages": max_pdf_pages,
                    "max_arxiv_papers": max_arxiv_papers,
                    "max_auto_resources": max_auto_resources,
                    "enable_live_search": enable_live_search,
                    "auto_download_sources": auto_download_sources,
                    "max_dynamic_text_blocks": max_dynamic_text_blocks,
                    "max_record_text_blocks": max_record_text_blocks,
                    "max_figures_per_pdf": max_figures_per_pdf,
                    "max_pdf_parse_workers": max_pdf_parse_workers,
                    "max_chart_workers": max_chart_workers,
                    "max_text_extraction_workers": max_text_extraction_workers,
                    "max_table_extraction_workers": max_table_extraction_workers,
                    "reuse_dynamic_records_for_metrics": reuse_dynamic_records_for_metrics,
                },
                auto_fetch_arxiv=enable_live_search,
                file_metadata=file_metadata,
            )
        except TaskQueueFullError:
            _cleanup_saved_uploads([Path(path) for path in saved_files], (UPLOAD_DIR / task_id).resolve())
            raise HTTPException(
                status_code=503,
                detail={"code": "TASK_QUEUE_FULL", "message": "任务队列已满，请稍后再试。"},
            )
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "status_url": f"/api/tasks/{task['task_id']}",
            "events_url": f"/api/tasks/{task['task_id']}/events",
        }

    @app.post("/api/discover", status_code=202, response_model=TaskSubmissionResponse)
    async def discover(
        research_question: str = Form(..., min_length=3, max_length=4000),
    ) -> dict[str, Any]:
        task_id = timestamp_task_id()
        try:
            task = TASK_MANAGER.submit(
                task_id=task_id,
                research_question=research_question,
                files=[],
                run_options={
                    "enable_live_search": True,
                    "auto_download_sources": False,
                    "discovery_only": True,
                },
                auto_fetch_arxiv=True,
            )
        except TaskQueueFullError:
            raise HTTPException(
                status_code=503,
                detail={"code": "TASK_QUEUE_FULL", "message": "任务队列已满，请稍后再试。"},
            )
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "status_url": f"/api/tasks/{task['task_id']}",
            "events_url": f"/api/tasks/{task['task_id']}/events",
        }

    @app.get("/api/tasks", response_model=TaskListResponse)
    def list_tasks(
        limit: int = Query(20, ge=1, le=100),
        status: TaskStatus | None = Query(None),
    ) -> dict[str, Any]:
        tasks = [
            _public_task_response(task)
            for task in TASK_MANAGER.list_tasks(limit=limit, status=status)
        ]
        return {"tasks": tasks, "count": len(tasks)}

    @app.get("/api/tasks/{task_id}", response_model=TaskResponse)
    def get_task(task_id: str) -> dict[str, Any]:
        return _public_task_response(_task_or_404(task_id))

    @app.get("/api/tasks/{task_id}/events", response_model=TaskEventsResponse)
    def get_task_events(
        task_id: str,
        tail: int = Query(100, ge=1, le=500),
        include_data: bool = Query(False),
    ) -> dict[str, Any]:
        _task_or_404(task_id)
        events = TASK_MANAGER.get_events(task_id, tail=tail, include_data=include_data)
        events["events"] = _sanitize_public_value(events["events"], task_id, manager=TASK_MANAGER)
        return events

    @app.post("/api/tasks/{task_id}/reviews/{record_id}", response_model=ReviewDecision)
    def review_record(task_id: str, record_id: str, request: ReviewRequest) -> dict[str, Any]:
        _task_or_404(task_id)
        try:
            return TASK_MANAGER.set_review_decision(
                task_id,
                record_id,
                request.decision,
                request.note,
            )
        except LookupError:
            raise HTTPException(
                status_code=409,
                detail={"code": "RESULT_NOT_READY", "message": "任务结果尚未生成，暂时不能复核。"},
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={"code": "RECORD_NOT_FOUND", "message": "待复核记录不存在。"},
            )

    @app.post("/api/tasks/{task_id}/cancel", response_model=TaskResponse)
    def cancel_task(task_id: str) -> dict[str, Any]:
        task = _task_or_404(task_id)
        if task.get("status") != "queued" or not TASK_MANAGER.cancel_task(task_id):
            raise HTTPException(
                status_code=409,
                detail={"code": "TASK_NOT_CANCELLABLE", "message": "任务已经开始执行或已经结束，无法直接取消。"},
            )
        return _public_task_response(_task_or_404(task_id))

    @app.post("/api/tasks/{task_id}/retry", status_code=202, response_model=TaskSubmissionResponse)
    def retry_task(task_id: str) -> dict[str, Any]:
        _task_or_404(task_id)
        try:
            task = TASK_MANAGER.retry_task(task_id, timestamp_task_id())
        except RuntimeError:
            raise HTTPException(
                status_code=409,
                detail={"code": "TASK_STILL_ACTIVE", "message": "任务仍在运行，不能重试。"},
            )
        except TaskQueueFullError:
            raise HTTPException(
                status_code=503,
                detail={"code": "TASK_QUEUE_FULL", "message": "任务队列已满，请稍后再试。"},
            )
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "status_url": f"/api/tasks/{task['task_id']}",
            "events_url": f"/api/tasks/{task['task_id']}/events",
        }
    @app.get("/api/tasks/{task_id}/export")
    def export_task(task_id: str, format: str = "csv"):
        _task_or_404(task_id)
        if format not in TASK_MANAGER.EXPORT_FILES:
            raise HTTPException(
                status_code=400,
                detail={"code": "UNSUPPORTED_EXPORT_FORMAT", "message": "不支持该导出格式。"},
            )
        file_path = TASK_MANAGER.download_path(task_id, format)
        if file_path is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "EXPORT_NOT_READY", "message": "导出文件尚未生成。"},
            )
        return FileResponse(file_path, filename=file_path.name)

    @app.get("/api/tasks/{task_id}/assets/{scope}/{asset_path:path}")
    def get_task_asset(task_id: str, scope: str, asset_path: str):
        _task_or_404(task_id)
        file_path = TASK_MANAGER.asset_path(task_id, scope, asset_path)
        if file_path is None:
            raise HTTPException(
                status_code=404,
                detail={"code": "ASSET_NOT_FOUND", "message": "任务资源不存在。"},
            )
        return FileResponse(file_path)
else:
    app = None
