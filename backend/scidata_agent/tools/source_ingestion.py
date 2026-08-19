from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from scidata_agent.agent.schemas import (
    DiscoveredSource,
    SourceInsight,
    SourceTriageDecision,
    SourceType,
    TextBlock,
    UploadedFile,
)
from scidata_agent.tools.connectors.base import USER_AGENT
from scidata_agent.tools.source_triage import SMALL_FILE_BYTES


TEXT_EXTENSIONS = {".txt", ".md", ".json", ".xml"}
PARSEABLE_TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
TABLE_EXTENSIONS = PARSEABLE_TABLE_EXTENSIONS | {".json", ".xml"}
PDF_EXTENSIONS = {".pdf"}


Downloader = Callable[[str, Path, int], None]
TextFetcher = Callable[[str, int, dict[str, str] | None], str]


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
                    content=f"Downloaded text-like supplement from {source.title}:\n{text[:8000]}",
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
        content=f"GitHub README for {source.title}:\n{text[:12000]}",
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
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            continue
    return None


def _download_url(url: str, target_path: Path, max_bytes: int) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise RuntimeError(f"download exceeds max_bytes={max_bytes}")
    target_path.write_bytes(content)


def _fetch_text(url: str, max_bytes: int, headers: dict[str, str] | None = None) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urlopen(request, timeout=20) as response:
        content = response.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise RuntimeError(f"text fetch exceeds max_bytes={max_bytes}")
    return content.decode("utf-8", errors="ignore")


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
