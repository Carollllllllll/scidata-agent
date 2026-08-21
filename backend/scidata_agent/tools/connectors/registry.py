from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from scidata_agent.agent.schemas import DiscoveredSource, MultiSourceSearchPlan, SourceSearchRequest
from scidata_agent.tools.connectors.arxiv import ArxivConnector
from scidata_agent.tools.connectors.base import BaseConnector, source_key
from scidata_agent.tools.connectors.crossref import CrossrefConnector
from scidata_agent.tools.connectors.figshare import FigshareConnector
from scidata_agent.tools.connectors.github import GitHubConnector
from scidata_agent.tools.connectors.openalex import OpenAlexConnector
from scidata_agent.tools.connectors.semantic_scholar import SemanticScholarConnector
from scidata_agent.tools.connectors.zenodo import ZenodoConnector


SearchFn = Callable[[SourceSearchRequest], list[DiscoveredSource]]
DEFAULT_SEARCH_MAX_WORKERS = 4
DEFAULT_SEARCH_RETRY_ROUNDS = 1
DEFAULT_SEARCH_CACHE_TTL_SECONDS = 24 * 60 * 60


class SearchResultCache:
    """Small atomic JSON cache for successful connector search responses."""

    def __init__(self, directory: Path, ttl_seconds: int | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if ttl_seconds is None:
            try:
                ttl_seconds = int(os.getenv("SCIDATA_SEARCH_CACHE_TTL_SECONDS", str(DEFAULT_SEARCH_CACHE_TTL_SECONDS)))
            except (TypeError, ValueError):
                ttl_seconds = DEFAULT_SEARCH_CACHE_TTL_SECONDS
        self.ttl_seconds = max(0, int(ttl_seconds))

    def _path(self, request: SourceSearchRequest) -> Path:
        payload = request.model_dump(mode="json")
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
        return self.directory / f"{key}.json"

    def get(self, request: SourceSearchRequest) -> list[DiscoveredSource] | None:
        path = self._path(request)
        try:
            if self.ttl_seconds and time.time() - path.stat().st_mtime > self.ttl_seconds:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [DiscoveredSource.model_validate(item) for item in payload.get("sources", [])]
        except (OSError, ValueError, TypeError):
            return None

    def put(self, request: SourceSearchRequest, sources: list[DiscoveredSource]) -> None:
        path = self._path(request)
        temporary = path.with_suffix(".tmp")
        payload = {"request": request.model_dump(mode="json"), "sources": [source.model_dump(mode="json") for source in sources]}
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _search_worker_count(max_workers: int | None) -> int:
    configured = max_workers
    if configured is None:
        try:
            configured = int(
                os.getenv("SCIDATA_SEARCH_MAX_WORKERS")
                or os.getenv("SCIDATA_CONNECTOR_WORKERS", str(DEFAULT_SEARCH_MAX_WORKERS))
            )
        except (TypeError, ValueError):
            configured = DEFAULT_SEARCH_MAX_WORKERS
    return max(1, int(configured))


def _search_one_request(
    request: SourceSearchRequest,
    connectors: dict[str, BaseConnector],
    searchers: dict[str, SearchFn],
    cache: SearchResultCache | None = None,
) -> tuple[SourceSearchRequest, list[DiscoveredSource], dict[str, Any]]:
    searcher = searchers.get(request.connector_name)
    connector = connectors.get(request.connector_name)
    if searcher is None and connector is None:
        return request, [], {
            "connector": request.connector_name,
            "query": request.query,
            "status": "failed",
            "error": "connector is not available",
            "attempts": 1,
            "added": 0,
        }
    if cache is not None:
        cached = cache.get(request)
        if cached is not None:
            return request, cached, {
                "connector": request.connector_name,
                "query": request.query,
                "status": "completed",
                "cache_hit": True,
                "attempts": 0,
                "added": 0,
            }
    try:
        sources = searcher(request) if searcher else connector.search(request)  # type: ignore[union-attr]
    except Exception as exc:
        return request, [], {
            "connector": request.connector_name,
            "query": request.query,
            "status": "failed",
            "error": str(exc),
            "attempts": 1,
            "added": 0,
        }
    if cache is not None:
        cache.put(request, sources)
    return request, sources, {
        "connector": request.connector_name,
        "query": request.query,
        "status": "completed",
        "cache_hit": False,
        "attempts": 1,
        "added": 0,
    }


def available_connectors() -> dict[str, BaseConnector]:
    connectors: list[BaseConnector] = [
        ArxivConnector(),
        OpenAlexConnector(),
        SemanticScholarConnector(),
        CrossrefConnector(),
        ZenodoConnector(),
        FigshareConnector(),
        GitHubConnector(),
    ]
    return {connector.name: connector for connector in connectors}


def execute_multi_source_search(
    plan: MultiSourceSearchPlan,
    searchers: dict[str, SearchFn] | None = None,
    max_workers: int | None = None,
    cache_dir: str | Path | None = None,
    retry_failed_requests: int | None = None,
) -> tuple[list[DiscoveredSource], dict[str, Any]]:
    if not plan.should_search:
        return [], {"status": "skipped", "searched": 0, "added": 0, "failed": 0, "connector_status": []}

    connectors = available_connectors()
    found: list[DiscoveredSource] = []
    existing_keys: set[str] = set()
    connector_status: list[dict[str, Any]] = []
    failed = 0
    requests = list(plan.search_requests)
    searchers_map = searchers or {}
    cache = SearchResultCache(Path(cache_dir)) if cache_dir else None
    if retry_failed_requests is None:
        try:
            retry_failed_requests = int(os.getenv("SCIDATA_SEARCH_RETRY_ROUNDS", str(DEFAULT_SEARCH_RETRY_ROUNDS)))
        except (TypeError, ValueError):
            retry_failed_requests = DEFAULT_SEARCH_RETRY_ROUNDS
    retry_failed_requests = max(0, int(retry_failed_requests))
    workers = min(_search_worker_count(max_workers), max(1, len(requests)))
    def run_requests(batch: list[SourceSearchRequest]):
        if workers == 1 or len(batch) <= 1:
            return [_search_one_request(request, connectors, searchers_map, cache) for request in batch]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="source-search") as executor:
            futures = [
                executor.submit(_search_one_request, request, connectors, searchers_map, cache)
                for request in batch
            ]
            # Consume in request order so downstream exports remain deterministic.
            return [future.result() for future in futures]

    outcomes = run_requests(requests)
    for retry_round in range(1, retry_failed_requests + 1):
        failed_requests = [request for request, _sources, status in outcomes if status["status"] == "failed"]
        if not failed_requests:
            break
        retry_outcomes = run_requests(failed_requests)
        retry_iter = iter(retry_outcomes)
        replacement = []
        for request, sources, status in outcomes:
            if status["status"] == "failed":
                _retry_request, retry_sources, retry_status = next(retry_iter)
                retry_status = {**retry_status, "retry_round": retry_round, "attempts": status.get("attempts", 1) + retry_status.get("attempts", 1)}
                replacement.append((request, retry_sources, retry_status))
            else:
                replacement.append((request, sources, status))
        outcomes = replacement

    for request, sources, status_entry in outcomes:
        if status_entry["status"] == "failed":
            failed += 1
            connector_status.append(status_entry)
            continue

        added = 0
        for source in sources:
            key = source_key(source)
            if key in existing_keys:
                continue
            source.query = source.query or request.query
            source.reason = source.reason or request.purpose
            source.metadata.setdefault("provider", request.connector_name)
            source.metadata.setdefault("search_purpose", request.purpose)
            source.metadata.setdefault("must_have", request.must_have)
            source.metadata.setdefault("nice_to_have", request.nice_to_have)
            found.append(source)
            existing_keys.add(key)
            added += 1
        connector_status.append({**status_entry, "added": added})

    searched = len(requests)

    status = "completed"
    if searched == 0:
        status = "skipped"
    elif failed and not found:
        status = "failed"
    elif failed:
        status = "partial"
    return (
        found,
        {
            "status": status,
            "searched": searched,
            "added": len(found),
            "failed": failed,
            "retry_rounds": retry_failed_requests,
            "cache_enabled": cache is not None,
            "max_workers": workers,
            "connector_status": connector_status,
        },
    )


def merge_sources(
    existing: list[DiscoveredSource],
    new_sources: list[DiscoveredSource],
) -> tuple[list[DiscoveredSource], int]:
    keys = {source_key(source) for source in existing}
    merged = list(existing)
    added = 0
    for source in new_sources:
        key = source_key(source)
        if key in keys:
            continue
        merged.append(source)
        keys.add(key)
        added += 1
    return merged, added
