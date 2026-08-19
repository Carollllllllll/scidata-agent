from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import FileResponse
except ImportError:  # pragma: no cover - allows core agent tests without FastAPI installed.
    FastAPI = None  # type: ignore[assignment]
    File = Form = UploadFile = None  # type: ignore[assignment]
    FileResponse = None  # type: ignore[assignment]

from scidata_agent.agent.scidata_agent import SciDataAgent
from scidata_agent.config import get_settings, load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "runtime" / "uploads"
OUTPUT_DIR = BASE_DIR / "runtime" / "outputs"
TASK_RESULTS: dict[str, dict] = {}


if FastAPI is not None:
    app = FastAPI(title="SciData Agent API", version="0.1.0")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "service": "SciData Agent API",
            "qwen_configured": bool(settings.dashscope_api_key),
            "model": settings.qwen_model,
            "vl_model": settings.qwen_vl_model,
            "text_model_pool": settings.qwen_models,
            "vl_model_pool": settings.qwen_vl_models,
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

    @app.post("/api/analyze")
    async def analyze(
        research_question: str = Form(...),
        files: list[UploadFile] | None = File(default=None),
        max_pdf_pages: int = Form(8),
        max_arxiv_papers: int | None = Form(None),
        max_dynamic_text_blocks: int = Form(20),
        max_record_text_blocks: int = Form(20),
        max_figures_per_pdf: int = Form(6),
    ) -> dict:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_files: list[str] = []
        for uploaded in files or []:
            target = UPLOAD_DIR / uploaded.filename
            with target.open("wb") as handle:
                shutil.copyfileobj(uploaded.file, handle)
            saved_files.append(str(target))

        agent = SciDataAgent(output_dir=OUTPUT_DIR)
        result = agent.run(
            research_question,
            saved_files,
            max_pdf_pages=max_pdf_pages,
            auto_fetch_arxiv=True,
            max_arxiv_papers=max_arxiv_papers,
            max_dynamic_text_blocks=max_dynamic_text_blocks,
            max_record_text_blocks=max_record_text_blocks,
            max_figures_per_pdf=max_figures_per_pdf,
        )
        payload = result.model_dump(mode="json", by_alias=True)
        TASK_RESULTS[result.task_id] = payload
        return payload

    @app.post("/api/discover")
    async def discover(research_question: str = Form(...)) -> dict:
        agent = SciDataAgent(output_dir=OUTPUT_DIR)
        result = agent.run(research_question, files=[], auto_fetch_arxiv=False)
        payload = result.model_dump(mode="json", by_alias=True)
        TASK_RESULTS[result.task_id] = payload
        return payload

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> dict:
        return TASK_RESULTS.get(task_id, {"task_id": task_id, "status": "not_found"})

    @app.get("/api/tasks/{task_id}/export")
    def export_task(task_id: str, format: str = "csv"):
        result = TASK_RESULTS.get(task_id)
        if not result:
            return {"task_id": task_id, "status": "not_found"}
        export_key = "json" if format == "json" else format
        file_path = result["export_files"].get(export_key)
        if not file_path:
            return {"task_id": task_id, "status": "format_not_found"}
        return FileResponse(file_path, filename=Path(file_path).name)
else:
    app = None
