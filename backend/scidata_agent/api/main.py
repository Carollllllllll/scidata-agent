from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError:  # pragma: no cover - allows core agent tests without FastAPI installed.
    FastAPI = None  # type: ignore[assignment]
    File = Form = UploadFile = None  # type: ignore[assignment]
    CORSMiddleware = None  # type: ignore[assignment]
    FileResponse = None  # type: ignore[assignment]

from scidata_agent.agent.schemas import timestamp_task_id
from scidata_agent.config import get_settings, load_dotenv
from scidata_agent.api.task_manager import TaskManager


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[3]
RUNTIME_DIR = BASE_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
TASK_STATE_DIR = RUNTIME_DIR / "tasks"
TASK_MANAGER = TaskManager(output_dir=OUTPUT_DIR, state_dir=TASK_STATE_DIR)


def _cors_origins() -> list[str]:
    configured = os.getenv("SCIDATA_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def _save_uploads(files: list[UploadFile] | None, task_id: str) -> list[str]:
    task_upload_dir = (UPLOAD_DIR / task_id).resolve()
    if task_upload_dir.parent != UPLOAD_DIR.resolve():
        raise ValueError("Invalid upload task ID")
    task_upload_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for uploaded in files or []:
        original_name = Path(uploaded.filename or "uploaded_file").name
        safe_name = original_name or "uploaded_file"
        target = task_upload_dir / f"{uuid4().hex[:12]}_{safe_name}"
        with target.open("wb") as handle:
            shutil.copyfileobj(uploaded.file, handle)
        saved_files.append(str(target))
    return saved_files


def _public_task_response(task: dict[str, Any]) -> dict[str, Any]:
    """Return API data without exposing server filesystem paths to the frontend."""
    result = task.get("result")
    if not isinstance(result, dict):
        return task

    response = copy.deepcopy(result)
    download_urls = task.get("download_urls") or TASK_MANAGER.download_urls(task["task_id"])
    response["status"] = task.get("status", response.get("status", "completed"))
    response["download_urls"] = download_urls
    response["task_state"] = {
        key: task.get(key)
        for key in (
            "task_id",
            "status",
            "current_step",
            "message",
            "progress",
            "created_at",
            "updated_at",
            "error",
        )
        if key in task
    }
    # Keep the established export_files key, but expose URLs rather than local paths.
    response["export_files"] = download_urls
    return response


if FastAPI is not None:
    app = FastAPI(title="SciData Agent API", version="0.2.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("shutdown")
    def shutdown_task_manager() -> None:
        TASK_MANAGER.shutdown()

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "service": "SciData Agent API",
            "version": "0.2.0",
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

    @app.post("/api/analyze", status_code=202)
    async def analyze(
        research_question: str = Form(...),
        files: list[UploadFile] | None = File(default=None),
        max_pdf_pages: int = Form(8),
        max_arxiv_papers: int | None = Form(None),
        max_dynamic_text_blocks: int = Form(20),
        max_record_text_blocks: int = Form(20),
        max_figures_per_pdf: int = Form(6),
    ) -> dict[str, Any]:
        task_id = timestamp_task_id()
        saved_files = _save_uploads(files, task_id)
        task = TASK_MANAGER.submit(
            task_id=task_id,
            research_question=research_question,
            files=saved_files,
            run_options={
                "max_pdf_pages": max_pdf_pages,
                "max_arxiv_papers": max_arxiv_papers,
                "max_dynamic_text_blocks": max_dynamic_text_blocks,
                "max_record_text_blocks": max_record_text_blocks,
                "max_figures_per_pdf": max_figures_per_pdf,
            },
            auto_fetch_arxiv=True,
        )
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "status_url": f"/api/tasks/{task['task_id']}",
            "events_url": f"/api/tasks/{task['task_id']}/events",
        }

    @app.post("/api/discover", status_code=202)
    async def discover(research_question: str = Form(...)) -> dict[str, Any]:
        task_id = timestamp_task_id()
        task = TASK_MANAGER.submit(
            task_id=task_id,
            research_question=research_question,
            files=[],
            run_options={},
            auto_fetch_arxiv=False,
        )
        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "status_url": f"/api/tasks/{task['task_id']}",
            "events_url": f"/api/tasks/{task['task_id']}/events",
        }

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, Any]:
        return _public_task_response(TASK_MANAGER.get_task(task_id))

    @app.get("/api/tasks/{task_id}/events")
    def get_task_events(task_id: str, tail: int = 100) -> dict[str, Any]:
        return TASK_MANAGER.get_events(task_id, tail=tail)

    @app.get("/api/tasks/{task_id}/export")
    def export_task(task_id: str, format: str = "csv"):
        file_path = TASK_MANAGER.download_path(task_id, format)
        if file_path is None:
            return {"task_id": task_id, "status": "format_not_found"}
        return FileResponse(file_path, filename=file_path.name)
else:
    app = None
