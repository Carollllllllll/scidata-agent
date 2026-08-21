from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
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


def _connector_workers(request_count: int) -> int:
    try:
        configured = int(os.getenv("SCIDATA_CONNECTOR_WORKERS", "4"))
    except (TypeError, ValueError):
        configured = 4
    return max(1, min(configured, max(1, request_count)))


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
) -> tuple[list[DiscoveredSource], dict[str, Any]]:
    if not plan.should_search:
        return [], {"status": "skipped", "searched": 0, "added": 0, "failed": 0, "connector_status": []}

    connectors = available_connectors()
    found: list[DiscoveredSource] = []
    existing_keys: set[str] = set()
    connector_status: list[dict[str, Any]] = []
    failed = 0
    searched = 0

    requests = list(plan.search_requests)
    searched = len(requests)
    jobs: list[tuple[SourceSearchRequest, Future[list[DiscoveredSource]] | None, str | None]] = []
    with ThreadPoolExecutor(
        max_workers=_connector_workers(len(requests)),
        thread_name_prefix="scidata-connector",
    ) as pool:
        for request in requests:
            searcher = (searchers or {}).get(request.connector_name)
            connector = connectors.get(request.connector_name)
            if searcher is None and connector is None:
                jobs.append((request, None, "connector is not available"))
                continue
            search_fn = searcher or connector.search  # type: ignore[union-attr]
            jobs.append((request, pool.submit(search_fn, request), None))

        # Consume futures in plan order. Requests run concurrently, while
        # deduplication and connector_status remain deterministic.
        results: list[tuple[SourceSearchRequest, list[DiscoveredSource] | None, str | None]] = []
        for request, future, preflight_error in jobs:
            if preflight_error is not None:
                results.append((request, None, preflight_error))
                continue
            try:
                results.append((request, future.result(), None))  # type: ignore[union-attr]
            except Exception as exc:
                results.append((request, None, str(exc)))

    for request, sources, error in results:
        if error is not None:
            failed += 1
            connector_status.append(
                {
                    "connector": request.connector_name,
                    "query": request.query,
                    "status": "failed",
                    "error": error,
                    "added": 0,
                }
            )
            continue

        added = 0
        for source in sources or []:
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
        connector_status.append(
            {
                "connector": request.connector_name,
                "query": request.query,
                "status": "completed",
                "added": added,
            }
        )

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
