from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import os
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


def _search_worker_count(max_workers: int | None) -> int:
    configured = max_workers
    if configured is None:
        try:
            configured = int(os.getenv("SCIDATA_SEARCH_MAX_WORKERS", str(DEFAULT_SEARCH_MAX_WORKERS)))
        except ValueError:
            configured = DEFAULT_SEARCH_MAX_WORKERS
    return max(1, int(configured))


def _search_one_request(
    request: SourceSearchRequest,
    connectors: dict[str, BaseConnector],
    searchers: dict[str, SearchFn],
) -> tuple[SourceSearchRequest, list[DiscoveredSource], dict[str, Any]]:
    searcher = searchers.get(request.connector_name)
    connector = connectors.get(request.connector_name)
    if searcher is None and connector is None:
        return request, [], {
            "connector": request.connector_name,
            "query": request.query,
            "status": "failed",
            "error": "connector is not available",
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
            "added": 0,
        }
    return request, sources, {
        "connector": request.connector_name,
        "query": request.query,
        "status": "completed",
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
    workers = min(_search_worker_count(max_workers), max(1, len(requests)))
    if workers == 1 or len(requests) <= 1:
        outcomes = [_search_one_request(request, connectors, searchers_map) for request in requests]
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="source-search") as executor:
            futures = [
                executor.submit(_search_one_request, request, connectors, searchers_map)
                for request in requests
            ]
            # Consume in request order so downstream exports remain deterministic.
            outcomes = [future.result() for future in futures]

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
