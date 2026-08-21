from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import re
import shutil
import time
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from scidata_agent.agent.schemas import ArxivSearchPlan, DiscoveredSource, SourceDiscoveryPlan, SourceSearchRequest
from scidata_agent.tools.connectors.base import BaseConnector


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_PDF_READ_TIMEOUT_SECONDS = 30
DEFAULT_PDF_TOTAL_TIMEOUT_SECONDS = 600
DEFAULT_ARXIV_BATCH_TIMEOUT_SECONDS = 3600
DEFAULT_ARXIV_DOWNLOAD_WORKERS = 3
PDF_CHUNK_SIZE = 1024 * 1024


class ArxivConnectorError(RuntimeError):
    """Raised when arXiv search fails."""


class ArxivConnector(BaseConnector):
    name = "arxiv"
    supported_source_types = ("paper", "paper_search", "paper_metadata")

    def search(self, request: SourceSearchRequest) -> list[DiscoveredSource]:
        return search_arxiv(request.query, request.max_results)


def search_arxiv(query: str, max_results: int = 5, timeout: int = 20, retries: int = 2) -> list[DiscoveredSource]:
    """Search arXiv and return paper-like DiscoveredSource objects."""
    normalized_query = " ".join(query.split())
    if not normalized_query:
        return []
    search_query = normalized_query if _looks_like_arxiv_query(normalized_query) else f"all:{normalized_query}"

    params = urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max(1, min(max_results, 20)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    request = Request(
        f"{ARXIV_API_URL}?{params}",
        headers={"User-Agent": "SciDataAgent/0.1 (research data discovery; contact=local)"},
    )

    last_exc: Exception | None = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                xml_text = response.read()
            break
        except HTTPError as exc:  # pragma: no cover - exercised by integration/smoke tests.
            last_exc = exc
            if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504} or attempt >= attempts:
                raise ArxivConnectorError(f"arXiv API request failed after {attempts} attempt(s): {exc}") from exc
        except (TimeoutError, URLError) as exc:  # pragma: no cover - exercised by integration/smoke tests.
            last_exc = exc
            if attempt >= attempts:
                raise ArxivConnectorError(f"arXiv API request failed after {attempts} attempt(s): {exc}") from exc
        except Exception as exc:  # pragma: no cover - exercised by integration/smoke tests.
            raise ArxivConnectorError(f"arXiv API request failed: {exc}") from exc
        time.sleep(min(2 ** (attempt - 1), 8))
    else:  # pragma: no cover - defensive; loop raises on final failure.
        raise ArxivConnectorError(f"arXiv API request failed after {attempts} attempt(s): {last_exc}")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivConnectorError(f"arXiv API returned invalid XML: {exc}") from exc

    return [_entry_to_source(entry, normalized_query) for entry in root.findall("atom:entry", ATOM_NS)]


def _download_worker_count(max_workers: int | None) -> int:
    configured = max_workers
    if configured is None:
        try:
            configured = int(os.getenv("SCIDATA_PDF_DOWNLOAD_MAX_WORKERS", str(DEFAULT_ARXIV_DOWNLOAD_WORKERS)))
        except ValueError:
            configured = DEFAULT_ARXIV_DOWNLOAD_WORKERS
    return max(1, int(configured))


def _download_one_arxiv_source(
    source: DiscoveredSource,
    index: int,
    total: int,
    download_dir: Path,
    timeout: int,
    retries: int,
    downloader: Callable[[str, Path, int], None],
    total_timeout: int,
    progress_callback: Callable[[str, dict[str, Any]], None] | None,
    reuse_dirs: list[Path],
) -> tuple[Path | None, dict[str, Any]]:
    pdf_url = source.metadata.get("pdf_url")
    if not isinstance(pdf_url, str) or not pdf_url:
        return None, {
            "status": "skipped",
            "title": source.title,
            "note": f"arXiv PDF download skipped: no pdf_url for '{source.title}'.",
        }

    filename = build_pdf_filename(source)
    target_path = download_dir / filename
    reused_path = _find_reusable_pdf(target_path, reuse_dirs)
    copy_note: str | None = None
    if reused_path is not None and reused_path != target_path:
        try:
            shutil.copy2(reused_path, target_path)
            copy_note = (
                f"arXiv PDF copied from previous task: title='{source.title}', "
                f"file='{target_path.name}'."
            )
        except OSError:
            _remove_quietly(target_path)

    if _is_valid_pdf(target_path):
        note = copy_note or f"arXiv PDF reused: title='{source.title}', file='{target_path.name}'."
        _notify(progress_callback, "reused", {"index": index, "total": total, "title": source.title, "path": str(target_path)})
        return target_path, {"status": "reused", "title": source.title, "note": note}

    if target_path.exists():
        _remove_quietly(target_path)
    _notify(progress_callback, "started", {"index": index, "total": total, "title": source.title, "path": str(target_path)})
    try:
        _download_with_retry(
            pdf_url,
            target_path,
            timeout,
            retries,
            downloader,
            total_timeout=total_timeout,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        note = f"arXiv PDF download failed for '{source.title}' after {retries + 1} attempt(s): {exc}"
        _notify(progress_callback, "failed", {"index": index, "total": total, "title": source.title, "error": str(exc)})
        return None, {"status": "failed", "title": source.title, "note": note}

    note = f"arXiv PDF downloaded: title='{source.title}', file='{target_path.name}'."
    _notify(progress_callback, "completed", {"index": index, "total": total, "title": source.title, "path": str(target_path)})
    return target_path, {"status": "completed", "title": source.title, "note": note}


def download_arxiv_pdfs(
    plan: SourceDiscoveryPlan,
    download_dir: Path,
    max_papers: int = 30,
    timeout: int = DEFAULT_PDF_READ_TIMEOUT_SECONDS,
    retries: int = 2,
    allowed_source_ids: set[str] | None = None,
    downloader: Callable[[str, Path, int], None] | None = None,
    total_timeout: int = DEFAULT_PDF_TOTAL_TIMEOUT_SECONDS,
    batch_timeout: int = DEFAULT_ARXIV_BATCH_TIMEOUT_SECONDS,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    reuse_dirs: list[Path] | None = None,
    max_workers: int | None = None,
) -> list[Path]:
    """Download selected arXiv PDFs with per-file and batch deadlines.

    Existing valid PDFs are reused. Failed or timed-out downloads are isolated
    to the current source and do not block the remaining sources.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    downloader = downloader or download_pdf
    batch_started = time.monotonic()

    downloaded: list[Path] = []
    # Select at most the configured download cap before scheduling work. This
    # preserves the hard resource limit while allowing those downloads to run concurrently.
    selected_sources = select_arxiv_papers(
        plan,
        max_papers=max(0, max_papers),
        allowed_source_ids=allowed_source_ids,
    )
    if not selected_sources:
        return downloaded
    if batch_timeout > 0 and time.monotonic() - batch_started >= batch_timeout:
        message = f"arXiv PDF batch timeout reached before starting {len(selected_sources)} source(s)."
        plan.notes.append(message)
        _notify(progress_callback, "batch_timeout", {"index": 1, "total": len(selected_sources)})
        return downloaded

    workers = min(_download_worker_count(max_workers), len(selected_sources))
    plan.notes.append(
        f"arXiv PDF download scheduled {len(selected_sources)} source(s) with max_workers={workers}."
    )
    reuse_dirs = reuse_dirs or []
    if workers == 1:
        outcomes = [
            _download_one_arxiv_source(
                source,
                index,
                len(selected_sources),
                download_dir,
                timeout,
                retries,
                downloader,
                total_timeout,
                progress_callback,
                reuse_dirs,
            )
            for index, source in enumerate(selected_sources, start=1)
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arxiv-download") as executor:
            futures = [
                executor.submit(
                    _download_one_arxiv_source,
                    source,
                    index,
                    len(selected_sources),
                    download_dir,
                    timeout,
                    retries,
                    downloader,
                    total_timeout,
                    progress_callback,
                    reuse_dirs,
                )
                for index, source in enumerate(selected_sources, start=1)
            ]
            outcomes = [future.result() for future in futures]

    for source, (path, result) in zip(selected_sources, outcomes, strict=True):
        if result.get("note"):
            plan.notes.append(str(result["note"]))
        if path is not None:
            source.metadata["downloaded_path"] = str(path)
            downloaded.append(path)
    return downloaded


def download_pdf(
    url: str,
    target_path: Path,
    timeout: int = DEFAULT_PDF_READ_TIMEOUT_SECONDS,
    total_timeout: int = DEFAULT_PDF_TOTAL_TIMEOUT_SECONDS,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    """Stream one PDF with socket-read and total-deadline protection."""
    request = Request(url, headers={"User-Agent": "SciDataAgent/0.1 (arXiv PDF ingestion; contact=local)"})
    partial_path = target_path.with_name(target_path.name + ".part")
    started = time.monotonic()
    bytes_written = 0
    try:
        with urlopen(request, timeout=timeout) as response, partial_path.open("wb") as handle:
            while True:
                if total_timeout > 0 and time.monotonic() - started >= total_timeout:
                    raise TimeoutError(f"PDF total download timeout after {total_timeout}s")
                chunk = response.read(PDF_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
                _notify(progress_callback, "progress", {"path": str(target_path), "bytes": bytes_written})
        if not _is_valid_pdf(partial_path):
            raise ArxivConnectorError("downloaded content is not a valid PDF")
        partial_path.replace(target_path)
    except Exception:
        _remove_quietly(partial_path)
        raise


def select_arxiv_papers(
    plan: SourceDiscoveryPlan,
    max_papers: int | None = 30,
    allowed_source_ids: set[str] | None = None,
) -> list[DiscoveredSource]:
    """Select arXiv paper sources with downloadable PDFs."""
    papers: list[DiscoveredSource] = []
    for source in plan.candidate_sources:
        if allowed_source_ids is not None and source.source_id not in allowed_source_ids:
            continue
        if source.source_type != "paper":
            continue
        if source.metadata.get("provider") != "arxiv":
            continue
        if not source.metadata.get("pdf_url"):
            continue
        papers.append(source)
        if max_papers is not None and len(papers) >= max(0, max_papers):
            break
    return papers


def _download_with_retry(
    pdf_url: str,
    target_path: Path,
    timeout: int,
    retries: int,
    downloader: Callable[[str, Path, int], None],
    *,
    total_timeout: int = DEFAULT_PDF_TOTAL_TIMEOUT_SECONDS,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    last_exc: Exception | None = None
    attempts = max(1, retries + 1)
    for attempt in range(1, attempts + 1):
        try:
            if downloader is download_pdf:
                download_pdf(
                    pdf_url,
                    target_path,
                    timeout=timeout,
                    total_timeout=total_timeout,
                    progress_callback=progress_callback,
                )
            else:
                downloader(pdf_url, target_path, timeout)
            return
        except Exception as exc:
            last_exc = exc
            _remove_quietly(target_path)
            _remove_quietly(target_path.with_name(target_path.name + ".part"))
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    if last_exc:
        raise last_exc


def _is_valid_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 5:
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(5) == b"%PDF-"
    except OSError:
        return False


def _find_reusable_pdf(target_path: Path, reuse_dirs: list[Path]) -> Path | None:
    if _is_valid_pdf(target_path):
        return target_path
    for reuse_dir in reuse_dirs:
        candidate = reuse_dir / target_path.name
        if _is_valid_pdf(candidate):
            return candidate
    return None


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _notify(callback: Callable[[str, dict[str, Any]], None] | None, status: str, data: dict[str, Any]) -> None:
    if callback is not None:
        callback(status, data)


def build_pdf_filename(source: DiscoveredSource) -> str:
    """Build a stable readable filename for a downloaded arXiv PDF."""
    pdf_url = str(source.metadata.get("pdf_url") or "")
    parsed = urlparse(pdf_url)
    arxiv_id = parsed.path.rstrip("/").split("/")[-1] if parsed.path else ""
    arxiv_id = arxiv_id.replace(".", "_").replace("/", "_")
    title_slug = re.sub(r"[^A-Za-z0-9]+", "_", source.title).strip("_").lower()[:80]
    if arxiv_id:
        return f"arxiv_{arxiv_id}_{title_slug or 'paper'}.pdf"
    return f"arxiv_{title_slug or source.source_id}.pdf"


def enrich_with_arxiv_results(
    plan: SourceDiscoveryPlan,
    arxiv_plan: ArxivSearchPlan,
    max_results: int = 20,
    searcher: Callable[[str, int], list[DiscoveredSource]] | None = None,
) -> tuple[SourceDiscoveryPlan, str]:
    """Append real arXiv paper results using LLM-planned arXiv queries."""
    searcher = searcher or search_arxiv
    if not arxiv_plan.should_search_arxiv:
        plan.notes.append("arXiv search skipped by LLM arXiv Search Planner.")
        return plan, "skipped"
    if not arxiv_plan.queries:
        plan.notes.append("arXiv search skipped: LLM arXiv Search Planner returned no queries.")
        return plan, "skipped"

    existing_keys = {_source_key(source) for source in plan.candidate_sources}
    added = 0
    failed = 0
    searched = 0
    for query_spec in arxiv_plan.queries:
        query = normalize_arxiv_query(query_spec.query)
        if not query:
            continue
        searched += 1
        requested = min(max_results, query_spec.max_results)
        try:
            papers = searcher(query, requested)
        except Exception as exc:
            failed += 1
            plan.notes.append(f"arXiv search failed: query='{query}', error={exc}")
            continue

        query_added = 0
        for paper in papers:
            key = _source_key(paper)
            if key in existing_keys:
                continue
            paper.query = query
            paper.reason = _paper_reason(paper.reason, query_spec.purpose)
            paper.metadata.setdefault("arxiv_search_purpose", query_spec.purpose)
            paper.metadata.setdefault("arxiv_selection_criteria", arxiv_plan.selection_criteria)
            plan.candidate_sources.append(paper)
            existing_keys.add(key)
            added += 1
            query_added += 1
        plan.notes.append(
            f"arXiv search completed: query='{query}', purpose='{query_spec.purpose or ''}', added_papers={query_added}."
        )

    status = f"added={added},searched={searched},failed={failed}"
    if searched == 0:
        return plan, "skipped"
    if failed and added == 0:
        return plan, f"failed={failed}"
    return plan, status


def normalize_arxiv_query(query: str) -> str:
    """Normalize an LLM-provided arXiv query without adding domain-specific terms."""
    cleaned = " ".join(str(query).split())
    if not cleaned:
        return ""
    if _looks_like_arxiv_query(cleaned):
        return cleaned
    return f"all:{cleaned}"


def _entry_to_source(entry: ET.Element, query: str) -> DiscoveredSource:
    title = _text(entry, "atom:title") or "Untitled arXiv paper"
    summary = _text(entry, "atom:summary")
    arxiv_id_url = _text(entry, "atom:id")
    published = _text(entry, "atom:published")
    updated = _text(entry, "atom:updated")
    authors = [
        _text(author, "atom:name")
        for author in entry.findall("atom:author", ATOM_NS)
        if _text(author, "atom:name")
    ]
    links = _extract_links(entry)
    pdf_url = _pick_pdf_url(links)

    metadata: dict[str, Any] = {
        "provider": "arxiv",
        "authors": authors,
        "published": published,
        "updated": updated,
        "pdf_url": pdf_url,
        "links": links,
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "", [])}

    return DiscoveredSource(
        title=" ".join(title.split()),
        source_type="paper",
        url=arxiv_id_url,
        query=query,
        description=" ".join(summary.split()) if summary else None,
        reason="Matched by real arXiv API search.",
        confidence=0.72,
        metadata=metadata,
    )


def _text(element: ET.Element, path: str) -> str | None:
    found = element.find(path, ATOM_NS)
    if found is None or found.text is None:
        return None
    return found.text.strip()


def _extract_links(entry: ET.Element) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    for link in entry.findall("atom:link", ATOM_NS):
        attrs = {key: value for key, value in link.attrib.items() if value}
        if attrs:
            links.append(attrs)
    return links


def _pick_pdf_url(links: list[dict[str, str]]) -> str | None:
    for link in links:
        if link.get("title") == "pdf" or link.get("type") == "application/pdf":
            return link.get("href")
    return None


def _source_key(source: DiscoveredSource) -> str:
    return (source.url or source.title).strip().lower()


def _looks_like_arxiv_query(query: str) -> bool:
    return any(token in query for token in ["all:", "ti:", "abs:", "au:", "cat:", "submittedDate:", " AND ", " OR "])


def _paper_reason(reason: str | None, purpose: str | None) -> str:
    parts = []
    if reason:
        parts.append(reason)
    if purpose:
        parts.append(f"LLM-planned arXiv query purpose: {purpose}")
    return " ".join(parts) if parts else "Matched by LLM-planned arXiv API search."
