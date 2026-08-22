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
from scidata_agent.tools.connectors.base import (
    BaseConnector,
    source_cluster_id,
    source_identity_keys,
)
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

        for source in sources:
            source.query = source.query or request.query
            source.reason = source.reason or request.purpose
            source.metadata.setdefault("provider", request.connector_name)
            source.metadata.setdefault("search_purpose", request.purpose)
            source.metadata.setdefault("must_have", request.must_have)
            source.metadata.setdefault("nice_to_have", request.nice_to_have)
        found, added = merge_sources(found, sources)
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
    merged = list(existing)
    key_index: dict[str, DiscoveredSource] = {}
    for source in merged:
        source.source_cluster_id = source_cluster_id(source)
        for key in source_identity_keys(source):
            key_index.setdefault(key, source)

    added = 0
    for source in new_sources:
        match = next((key_index[key] for key in source_identity_keys(source) if key in key_index), None)
        if match is not None:
            _merge_source_records(match, source)
            for key in source_identity_keys(source):
                key_index.setdefault(key, match)
            continue
        source.source_cluster_id = source_cluster_id(source)
        merged.append(source)
        for key in source_identity_keys(source):
            key_index.setdefault(key, source)
        added += 1
    return merged, added


def _merge_source_records(target: DiscoveredSource, incoming: DiscoveredSource) -> None:
    """Merge a provider record into the canonical source without losing provenance."""
    target.source_cluster_id = target.source_cluster_id or source_cluster_id(target)
    metadata = target.metadata
    incoming_metadata = incoming.metadata or {}

    source_records = metadata.get("source_records")
    if not isinstance(source_records, list):
        source_records = []
        metadata["source_records"] = source_records
    if not source_records:
        source_records.append(_source_record_snapshot(target, target.source_cluster_id))
    if not any(item.get("source_id") == incoming.source_id for item in source_records if isinstance(item, dict)):
        source_records.append(_source_record_snapshot(incoming, target.source_cluster_id))

    source_ids = metadata.get("source_ids")
    if not isinstance(source_ids, list):
        source_ids = [target.source_id]
        metadata["source_ids"] = source_ids
    if incoming.source_id not in source_ids:
        source_ids.append(incoming.source_id)
    providers = metadata.get("providers")
    if not isinstance(providers, list):
        providers = []
        metadata["providers"] = providers
    for provider in (metadata.get("provider"), incoming_metadata.get("provider")):
        if provider and provider not in providers:
            providers.append(provider)
    alternate_urls = metadata.get("alternate_urls")
    if not isinstance(alternate_urls, list):
        alternate_urls = []
        metadata["alternate_urls"] = alternate_urls
    for url in (incoming.url, incoming_metadata.get("pdf_url"), incoming_metadata.get("open_access_url")):
        if url and url != target.url and url not in alternate_urls:
            alternate_urls.append(url)

    conflicts = metadata.get("source_conflicts")
    if not isinstance(conflicts, list):
        conflicts = []
        metadata["source_conflicts"] = conflicts
    for field_name, incoming_value in incoming_metadata.items():
        if field_name in {"provider", "source_records", "source_ids", "providers", "alternate_urls", "source_conflicts"}:
            continue
        current_value = metadata.get(field_name)
        if current_value in (None, "", []):
            metadata[field_name] = incoming_value
        elif incoming_value not in (None, "", []) and _comparable_value(current_value) != _comparable_value(incoming_value):
            conflict = {
                "field": field_name,
                "values": [current_value, incoming_value],
                "source_ids": [target.source_id, incoming.source_id],
            }
            if conflict not in conflicts:
                conflicts.append(conflict)

    if incoming.description and not target.description:
        target.description = incoming.description
    if incoming.reason and incoming.reason not in (target.reason or ""):
        target.reason = " ".join(part for part in (target.reason, incoming.reason) if part)
    target.confidence = max(target.confidence, incoming.confidence)


def _comparable_value(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(sorted(_comparable_value(item) for item in value))
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=True)
    return " ".join(str(value).split()).strip().lower()


def _source_record_snapshot(source: DiscoveredSource, cluster_id: str) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "source_cluster_id": cluster_id,
        "provider": source.metadata.get("provider"),
        "source_type": source.source_type,
        "title": source.title,
        "url": source.url,
        "metadata": {
            key: value
            for key, value in (source.metadata or {}).items()
            if key not in {"source_records", "source_ids", "providers", "alternate_urls", "source_conflicts"}
        },
    }
