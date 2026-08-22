from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import Request
from uuid import uuid4

from scidata_agent.agent.schemas import (
    DiscoveredSource,
    SourceArtifact,
    SourceInsight,
    SourceTriageDecision,
    SourceType,
    TextBlock,
    UploadedFile,
)
from scidata_agent.tools.connectors.base import USER_AGENT
from scidata_agent.tools.source_triage import SMALL_FILE_BYTES
from scidata_agent.tools.url_safety import safe_urlopen


TEXT_EXTENSIONS = {".txt", ".md", ".json", ".xml"}
PARSEABLE_TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
TABLE_EXTENSIONS = PARSEABLE_TABLE_EXTENSIONS | {".json", ".xml"}
PDF_EXTENSIONS = {".pdf"}
DEFAULT_ARTIFACT_MAX_BYTES = 100 * 1024 * 1024


Downloader = Callable[[str, Path, int], None]
TextFetcher = Callable[[str, int, dict[str, str] | None], str]


def download_source_artifact(
    artifact: SourceArtifact,
    output_dir: Path,
    *,
    max_bytes: int | None = None,
    downloader: Downloader | None = None,
) -> Path:
    """Materialize one planner-selected artifact without hiding download errors.

    Existing local files are reused. Remote files are downloaded through the
    same URL-safety and content-validation path used by triage ingestion.
    ``max_bytes`` is a transport safety limit, not a limit on the number of
    research results or selected artifacts.
    """
    if artifact.local_path:
        local_path = Path(artifact.local_path).expanduser().resolve()
        if local_path.exists() and local_path.is_file():
            artifact.local_path = str(local_path)
            artifact.name = artifact.name or local_path.name
            artifact.size_bytes = local_path.stat().st_size
            artifact.status = "downloaded"
            return local_path
    if not artifact.url:
        raise ValueError("Selected artifact has no URL and no existing local_path.")

    limit = max_bytes if max_bytes is not None else _positive_env_int(
        "SCIDATA_ARTIFACT_MAX_BYTES", DEFAULT_ARTIFACT_MAX_BYTES
    )
    if limit <= 0:
        raise ValueError("Artifact download max_bytes must be greater than zero.")
    target_dir = output_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / _artifact_filename(artifact)
    downloader = downloader or _download_url
    downloader(artifact.url, target_path, limit)
    if not target_path.exists() or not target_path.is_file():
        raise RuntimeError(f"Downloader returned without creating file: {target_path}")
    artifact.local_path = str(target_path)
    artifact.name = artifact.name or target_path.name
    artifact.size_bytes = target_path.stat().st_size
    artifact.content_type = artifact.content_type or _content_type(target_path)
    artifact.status = "downloaded"
    artifact.failure_reason = None
    return target_path


def _artifact_filename(artifact: SourceArtifact) -> str:
    raw_name = Path(str(artifact.name or "")).name
    raw_name = raw_name.split("?", 1)[0].split("#", 1)[0]
    if not raw_name or raw_name in {".", ".."}:
        raw_name = Path(str(artifact.url or "")).name.split("?", 1)[0]
    if not raw_name:
        raw_name = artifact.artifact_id
    suffix = Path(raw_name).suffix.lower()
    suffix_by_type = {
        "pdf": ".pdf",
        "supplementary_pdf": ".pdf",
        "csv": ".csv",
        "tsv": ".tsv",
        "xlsx": ".xlsx",
        "json": ".json",
        "xml": ".xml",
        "html": ".html",
        "readme": ".md",
        "image": ".bin",
    }
    if not suffix and artifact.artifact_type in suffix_by_type:
        raw_name += suffix_by_type[artifact.artifact_type]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name)[:180] or artifact.artifact_id


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def ingest_triaged_sources(
    sources: list[DiscoveredSource],
    decisions: list[SourceTriageDecision],
    output_dir: Path,
    task_id: str,
    max_bytes: int = SMALL_FILE_BYTES,
    downloader: Downloader | None = None,
    text_fetcher: TextFetcher | None = None,
) -> tuple[list[UploadedFile], list[TextBlock], list[SourceInsight], list[str]]:
    """Perform lightweight multi-source ingestion based on triage decisions."""
    source_by_id = {source.source_id: source for source in sources}
    download_dir = output_dir / task_id / "downloads" / "multi_source"
    download_dir.mkdir(parents=True, exist_ok=True)
    downloader = downloader or _download_url
    text_fetcher = text_fetcher or _fetch_text

    uploaded_files: list[UploadedFile] = []
    text_blocks: list[TextBlock] = []
    insights: list[SourceInsight] = []
    logs: list[str] = []

    for decision in decisions:
        source = source_by_id.get(decision.source_id)
        if source is None or decision.recommended_action == "skip":
            continue

        metadata_insight = _metadata_insight(source, decision)
        insights.append(metadata_insight)
        text_blocks.extend(_insight_to_text_blocks(metadata_insight))

        if decision.recommended_action == "read_readme":
            readme_insight, readme_log = _read_github_readme(source, decision, text_fetcher)
            insights.append(readme_insight)
            text_blocks.extend(_insight_to_text_blocks(readme_insight))
            logs.append(readme_log)
            continue

        if decision.recommended_action == "read_file_manifest":
            manifest_insight = _file_manifest_insight(source, decision)
            insights.append(manifest_insight)
            text_blocks.extend(_insight_to_text_blocks(manifest_insight))
            continue

        if decision.recommended_action in {"download_pdf", "download_small_table", "download_small_supplement"}:
            if decision.provider == "arxiv":
                logs.append(f"Multi-source ingestion deferred arXiv PDF to arXiv downloader: {source.title}")
                continue
            file_url = _download_url_for_decision(source, decision)
            if not file_url:
                error = _error_insight(source, decision, "No downloadable URL was available for this triage action.")
                insights.append(error)
                text_blocks.extend(_insight_to_text_blocks(error))
                logs.append(f"Download skipped: source='{source.title}', reason=no downloadable URL.")
                continue

            target_path = download_dir / _download_filename(source, file_url, decision.recommended_action)
            try:
                downloader(file_url, target_path, max_bytes)
            except Exception as exc:
                error = _error_insight(source, decision, f"Download failed: {exc}")
                insights.append(error)
                text_blocks.extend(_insight_to_text_blocks(error))
                logs.append(f"Download failed: source='{source.title}', url='{file_url}', error={exc}")
                continue

            source.metadata.setdefault("downloaded_paths", []).append(str(target_path))
            source.metadata["downloaded_path"] = str(target_path)
            insight = SourceInsight(
                source_id=source.source_id,
                title=source.title,
                provider=decision.provider,
                source_type=source.source_type,
                insight_type="downloaded_file",
                content=f"Downloaded file for source '{source.title}': {target_path.name}\nURL: {file_url}",
                url=file_url,
                confidence=decision.relevance_score,
                metadata={
                    "path": str(target_path),
                    "action": decision.recommended_action,
                    "suffix": target_path.suffix.lower(),
                },
            )
            insights.append(insight)
            logs.append(f"Downloaded source file: source='{source.title}', file='{target_path.name}'.")

            if target_path.suffix.lower() in PARSEABLE_TABLE_EXTENSIONS | PDF_EXTENSIONS:
                uploaded_files.append(
                    UploadedFile(
                        filename=target_path.name,
                        path=target_path,
                        content_type=_content_type(target_path),
                    )
                )
            elif target_path.suffix.lower() in TEXT_EXTENSIONS:
                text = target_path.read_text(encoding="utf-8", errors="ignore")
                text_insight = SourceInsight(
                    source_id=source.source_id,
                    title=source.title,
                    provider=decision.provider,
                    source_type=source.source_type,
                    insight_type="source_summary",
                    content=f"Downloaded text-like supplement from {source.title}:\n{text}",
                    url=file_url,
                    confidence=decision.relevance_score,
                    metadata={"path": str(target_path), "action": decision.recommended_action},
                )
                insights.append(text_insight)
                text_blocks.extend(_insight_to_text_blocks(text_insight))

    return uploaded_files, text_blocks, insights, logs


def _metadata_insight(source: DiscoveredSource, decision: SourceTriageDecision) -> SourceInsight:
    metadata = source.metadata or {}
    content = {
        "title": source.title,
        "provider": decision.provider,
        "source_type": source.source_type,
        "url": source.url,
        "query": source.query,
        "description": source.description,
        "reason": source.reason,
        "doi": metadata.get("doi") or metadata.get("DOI"),
        "authors": metadata.get("authors") or metadata.get("creators"),
        "year": metadata.get("year") or metadata.get("publication_year"),
        "published": metadata.get("published") or metadata.get("publication_date") or metadata.get("published_date"),
        "venue": metadata.get("venue"),
        "pdf_url": metadata.get("pdf_url"),
        "open_access_url": metadata.get("open_access_url"),
        "topics": metadata.get("topics"),
        "keywords": metadata.get("keywords"),
        "triage_action": decision.recommended_action,
        "triage_reason": decision.reason,
    }
    return SourceInsight(
        source_id=source.source_id,
        title=source.title,
        provider=decision.provider,
        source_type=source.source_type,
        insight_type="metadata",
        content=json.dumps({key: value for key, value in content.items() if value not in (None, "", [], {})}, ensure_ascii=False, indent=2),
        url=source.url,
        confidence=decision.relevance_score,
        metadata={"triage_action": decision.recommended_action},
    )


def _file_manifest_insight(source: DiscoveredSource, decision: SourceTriageDecision) -> SourceInsight:
    files = source.metadata.get("files", [])
    return SourceInsight(
        source_id=source.source_id,
        title=source.title,
        provider=decision.provider,
        source_type=source.source_type,
        insight_type="file_manifest",
        content=json.dumps({"title": source.title, "files": files}, ensure_ascii=False, indent=2),
        url=source.url,
        confidence=decision.relevance_score,
        metadata={"file_count": len(files) if isinstance(files, list) else 0},
    )


def _read_github_readme(
    source: DiscoveredSource,
    decision: SourceTriageDecision,
    text_fetcher: TextFetcher,
) -> tuple[SourceInsight, str]:
    full_name = source.metadata.get("full_name")
    if not full_name:
        return (
            _error_insight(source, decision, "GitHub full_name is missing; README cannot be read."),
            f"GitHub README skipped: source='{source.title}', reason=missing full_name.",
        )
    readme_url = f"https://api.github.com/repos/{full_name}/readme"
    try:
        text = text_fetcher(readme_url, SMALL_FILE_BYTES, {"Accept": "application/vnd.github.raw"})
    except Exception as exc:
        return (
            _error_insight(source, decision, f"GitHub README fetch failed: {exc}"),
            f"GitHub README fetch failed: source='{source.title}', error={exc}",
        )
    insight = SourceInsight(
        source_id=source.source_id,
        title=source.title,
        provider=decision.provider,
        source_type=source.source_type,
        insight_type="readme",
        content=f"GitHub README for {source.title}:\n{text}",
        url=readme_url,
        confidence=decision.relevance_score,
        metadata={"full_name": full_name},
    )
    source.metadata["readme_url"] = readme_url
    source.metadata["readme_chars"] = len(text)
    return insight, f"GitHub README read: source='{source.title}', chars={len(text)}."


def _error_insight(source: DiscoveredSource, decision: SourceTriageDecision, message: str) -> SourceInsight:
    return SourceInsight(
        source_id=source.source_id,
        title=source.title,
        provider=decision.provider,
        source_type=source.source_type,
        insight_type="download_error",
        content=message,
        url=source.url,
        confidence=0.2,
        metadata={"action": decision.recommended_action},
    )


def _insight_to_text_blocks(insight: SourceInsight, max_chars: int = 4500) -> list[TextBlock]:
    chunks = _chunk_text(insight.content, max_chars=max_chars)
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            TextBlock(
                source_file=_safe_source_filename(insight),
                source_path=insight.url or insight.source_id,
                source_type=SourceType.UNKNOWN,
                page=None,
                text=chunk,
                chunk_id=f"{insight.insight_id}_{index}_{uuid4().hex[:6]}",
            )
        )
    return blocks


def _download_url_for_decision(source: DiscoveredSource, decision: SourceTriageDecision) -> str | None:
    if decision.recommended_action == "download_pdf":
        return source.metadata.get("pdf_url") or source.metadata.get("open_access_url") or source.url
    best_file = _best_small_file(source.metadata.get("files", []), prefer_table=decision.recommended_action == "download_small_table")
    if best_file:
        return (
            best_file.get("download_url")
            or best_file.get("url")
            or best_file.get("self")
            or (best_file.get("links") or {}).get("download")
            or (best_file.get("links") or {}).get("self")
        )
    return None


def _best_small_file(files: Any, prefer_table: bool) -> dict[str, Any] | None:
    if not isinstance(files, list):
        return None
    candidates = []
    for file_item in files:
        if not isinstance(file_item, dict):
            continue
        size = _file_size(file_item)
        if size is not None and size > SMALL_FILE_BYTES:
            continue
        extension = _file_extension(file_item)
        if prefer_table and extension not in TABLE_EXTENSIONS:
            continue
        if not prefer_table and extension not in TEXT_EXTENSIONS | PDF_EXTENSIONS:
            continue
        candidates.append((size or 0, file_item))
    return sorted(candidates, key=lambda item: item[0])[0][1] if candidates else None


def _download_filename(source: DiscoveredSource, url: str, action: str) -> str:
    suffix = _suffix_from_url(url)
    if not suffix:
        suffix = _suffix_from_selected_file(source, action)
    if not suffix:
        suffix = ".pdf" if action == "download_pdf" else ".dat"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", source.title).strip("_").lower()[:80] or source.source_id
    provider = str(source.metadata.get("provider") or "source").lower()
    return f"{provider}_{source.source_id}_{slug}{suffix}"


def _suffix_from_selected_file(source: DiscoveredSource, action: str) -> str:
    best_file = _best_small_file(
        source.metadata.get("files", []),
        prefer_table=action == "download_small_table",
    )
    if not best_file:
        return ""
    return _file_extension(best_file)


def _suffix_from_url(url: str) -> str:
    match = re.search(r"(\.[A-Za-z0-9]{2,5})(?:$|\?)", url.lower())
    return match.group(1) if match else ""


def _file_extension(file_item: dict[str, Any]) -> str:
    name = str(file_item.get("name") or file_item.get("key") or file_item.get("download_url") or file_item.get("url") or "")
    suffix = _suffix_from_url(name)
    if suffix:
        return suffix
    file_type = str(file_item.get("type") or "").lower()
    return f".{file_type}" if file_type in {"csv", "tsv", "xlsx", "xls", "json", "xml", "pdf", "txt", "md"} else ""


def _file_size(file_item: dict[str, Any]) -> int | None:
    for key in ["size", "filesize", "file_size"]:
        value = file_item.get(key)
        try:
            if value not in (None, ""):
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _download_url(url: str, target_path: Path, max_bytes: int) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    timeout = _positive_env_int("SCIDATA_ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS", 60)
    with safe_urlopen(request, timeout=timeout) as response:
        content = response.read(max_bytes + 1)
        content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
    if len(content) > max_bytes:
        raise RuntimeError(f"download exceeds max_bytes={max_bytes}")
    _validate_download_content(target_path.suffix.lower(), content, content_type)
    target_path.write_bytes(content)


def _fetch_text(url: str, max_bytes: int, headers: dict[str, str] | None = None) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with safe_urlopen(request, timeout=20) as response:
        content = response.read(max_bytes + 1)
        content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].lower()
    if len(content) > max_bytes:
        raise RuntimeError(f"text fetch exceeds max_bytes={max_bytes}")
    if content_type.startswith(("image/", "audio/", "video/")) or b"\x00" in content[:4096]:
        raise RuntimeError(f"response is not text content: content_type={content_type or 'unknown'}")
    return content.decode("utf-8", errors="ignore")


def _validate_download_content(suffix: str, content: bytes, content_type: str) -> None:
    if not content:
        raise RuntimeError("downloaded file is empty")
    if suffix == ".pdf" and not content.lstrip().startswith(b"%PDF-"):
        raise RuntimeError(f"downloaded content is not a PDF: content_type={content_type or 'unknown'}")
    if suffix == ".xlsx" and not content.startswith(b"PK"):
        raise RuntimeError("downloaded content is not an XLSX archive")
    if suffix == ".xls" and not content.startswith(bytes.fromhex("D0CF11E0A1B11AE1")):
        raise RuntimeError("downloaded content is not an XLS workbook")
    if suffix in {".csv", ".tsv", ".json", ".xml", ".txt", ".md"} and b"\x00" in content[:4096]:
        raise RuntimeError("downloaded text/table content appears to be binary")


def _content_type(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".csv":
        return "text/csv"
    if suffix in {".xlsx", ".xls"}:
        return "application/vnd.ms-excel"
    return None


def _safe_source_filename(insight: SourceInsight) -> str:
    provider = re.sub(r"[^A-Za-z0-9]+", "_", str(insight.provider or "source")).strip("_").lower()
    kind = re.sub(r"[^A-Za-z0-9]+", "_", insight.insight_type).strip("_").lower()
    title = re.sub(r"[^A-Za-z0-9]+", "_", insight.title).strip("_").lower()[:60] or insight.source_id
    return f"{provider}_{kind}_{title}.txt"


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[index:index + max_chars] for index in range(0, len(text), max_chars)]
